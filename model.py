import sys
import json
import numpy as np
import tensorflow as tf
from scipy.special import softmax


# Gaussian Error Linear Unit
def gelu(x):
    """
    - Relu와 같이 사용되는 활성화 함수로 모델의 출력을 제어
    - GeLU 함수는 입력값에 대해 부드러운 비선형성을 제공하며 더 복잡한 패턴을 학습 할 수 있도록 도와줌
    - ReLU는 음수 입력값에 대해 0을 출력하는 특성이 있으나 GeLU는 음수 입력값에 대해서도 0이 아닌 출력을 제공
    - GeLU는 NLP 모델에서 많이 사용되며 ReLU는 CNN 모델에서 많이 사용
    """

    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


# Linear
def linear(emb, w, b):
    """
    - 신경망의 각 레이어에서 입력값을 다음 레이어로 전달할 때 수행되는 기본 연산
    - 입력값에 가중치를 적용하고 편향을 더하여 선형 변환을 수행
    - 입력값의 차원 수와 가중치의 차원 수에 따라 출력값의 차원 수가 결정됨
    - 모델이 데이터의 특성을 학습하고 복잡한 패턴을 예측하는 데 사용됨
    - 일반적으로 신경망의 각 레이어에서 사용되며, 활성화 함수와 함께 사용되어 모델의 출력을 제어함
    """
    if b.shape[0] != w.shape[1]:
        raise ValueError("Bias dimension is not match with weight dimension")

    return np.matmul(emb, w) + b[np.newaxis, :]


# Layer Normalization
def normalize(x, g, b, eps=1e-10):
    """
    - normalize 함수는 입력값 x를 정규화하여 분포의 평균이 0이고 표준편차가 1이 되도록 함.
    - 정규화 과정은 입력값을 표준화하여 모델의 학습을 안정화하고, 가중치 업데이트를 더 효율적으로 할 수 있게 함
    - g와 b는 모델이 학습하는 동안 학습되는 가중치와 편향 파라미터로 각각 정규화된 값에 곱해지고 더해짐
    - eps는 분산이 0이 되는 것을 방지하기 위한 작은 값
    """
    # axis=-1은 마지막 축을 의미하며, keepdims=True는 축을 유지하도록 함
    # x = np.array([[1, 2, 3], [4, 5, 6]]) 이고 axis가 0일 경우 출력은 [2.5, 3.5, 4.5]가 됨
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    sqrtd = np.sqrt(var + eps)
    h = (x - mean) / sqrtd

    return h * g[np.newaxis, :] + b[np.newaxis, :]


# Attention Mask
def attention_mask(num_decoder_tokens, num_sequence_tokens):
    """
    - 디코더에서 입력값이 하나씩 늘어날 때마다 어텐션을 제한하는 마스크를 생성
    - 미래의 정보를 참조하지 못하도록 하여 모델의 학습을 안정화
    - 마스크 값이 True일 때 어텐션은 허용되고, False일 때는 어텐션이 제한됨
    """

    i = np.arrange(num_decoder_tokens)[:, None]
    j = np.arrange(num_sequence_tokens)
    mask = i >= j - num_sequence_tokens + num_decoder_tokens

    return mask


class gpt2Model:
    def __init__(self):
        # 모델이 처리 할 수 있는 총 단어의 수
        self.n_vocab = None
        # 모델이 처리 할 수 있는 최대 시컨스 길이
        self.n_ctx = None
        # 임베딩 차원 수
        self.n_embd = None
        # 모델의 어텐션 헤드 수
        self.n_head = None
        # 모델의 레이어 수
        # [질문] 레이어 수가 뭐지? 트랜스포머 모델에서 레이어는 무엇을 의미하나요?
        self.n_layer = None
        # 키 벡터 저장소
        self.k_memory = None
        # 밸류 벡터 저장소
        self.v_memory = None
        self.tensors = None
        self.shapes = None
        self.eps = 1e-6

    def load(self, dir_model):
        print("loading hypter paramenters")
        with open(dir_model + "/hparams.json", "r", encoding="utf-8") as f:
            hparams = json.load(f)

        self.n_vocab = hparams["n_vocab"]
        self.n_ctx = hparams["n_ctx"]
        self.n_embd = hparams["n_embd"]
        self.n_head = hparams["n_head"]
        self.n_layer = hparams["n_layer"]
        self.k_momory = np.zeros(
            shape=(self.n_layer, self.n_ctx, self.n_embd), dtype=np.float32
        )
        self.v_memory = np.zeros(
            shape=(self.n_layer, self.n_ctx, self.n_embd), dtype=np.float32
        )
        # 모델의 weight, bias 등 다양한 텐서 데이터를 저장하기 위한 딕셔너리
        self.tensors = {}
        # 각 텐서 데이터의 차원 정보를 저장하기 위한 딕셔너리
        self.shapes = {}

        list_vars = tf.train.list_variables(dir_model)
        for tensor_name, shape in list_vars:
            print("loading variable %s" % tensor_name)
            data = tf.train.load_variable(dir_model, tensor_name)
            # [질문] 해당 변수들만 트랜스포즈 하는 이유는? squeeze를 하는 이유는?
            # 모델을 만들때 애초에 스퀴즈해서 저장하면 되는거 아닌지?
            transpose_list = [
                "/attn/c_attn/w",
                "/attn/c_proj/w",
                "/mlp/c_fc/w",
                "/mlp/c_proj/w",
            ]
            if np.any([tensor_name.endswith(x) for x in transpose_list]):
                data = data.squeeze(0)
            shape = data.shape
            print("proccessing variable %s with shape %s" % (tensor_name, shape))
            # [질문] tensor_name에 model/ 경로가 있는지는 어떻게 알수있는지? 일반적인 다른 모델에서도 동일하게 적용되는 부분인지
            tensor_name = tensor_name.split("model/")[1]
            self.tensors[tensor_name] = data
        self.shapes = {key: tensor.shape for (key, tensor) in self.tensors.items()}
        print("processing done")

    def forward(self, tokens, n_past):
        # 헤드 수를 로컬 변수로 저장
        n_head = self.n_head
        # 모델의 임베딩 벡터의 차원수를 로컬 변수로 저장
        n_embd = self.n_embd
        # 입력 토큰의 길이를 계산
        n = len(tokens)
        # 토큰의 위치 인덱스를 저장할 배열을 초기화
        pos_indices = np.zeros(n, dtype=np.int32)
        # [질문] n_past가 무엇을 의미하는지 모르겠습니다.
        # [질문] 그리고 포지션 벡터에 i 번째임을 저장하는 거라 생각했는데 n_past를 더하는 이유가 뭔가요?
        for i in range(n):
            pos_indices[i] = i + n_past
        # [질문]n_past를 더해서 위치 벡터에는 단순히 i위치가 아니라 i+n_past의 값을 가질텐데 tensor에서 위치 임베딩 값을 가져오는게 정확한건지?
        # 그냥 i로 저장된 벡터를 tensors["wpe"][0,1,...,n-1]로 가져와야한다고 생각했는데 이 부분이 이해가 안갑니다
        pos_emb = self.tensors["wpe"][pos_indices]
        token_emb = self.tensors["wte"][tokens]
        input_layer = token_emb + pos_emb

        for l in range(self.n_layer):
            """
            attn input layer normalization
            h%d/ln_1/g shape [n_embd]
            h%d/ln_1/b shape [n_embd]
            """
            h = normalize(
                input_layer,
                self.tensors["h%d/ln_1/g" % l],
                self.tensors["h%d/ln_1/b" % l],
            )

            """
            self attention layer
            h%d/attn/c_attn/w [3 * n_embd, n_embd]
            h%d/attn/c_attn/b [3 * n_embd]
            """

            attn = linear(
                h,
                self.tensors["h%d/attn/c_attn/w" % l],
                self.tensors["h%d/attn/c_attn/b" % l],
            )
            # [질문] 어텐션 벡터를 그대로 복사해서 q,k,v로 사용하고 각각의 weight을 찾는 다고 생각했는데 3등분해서 쓰는 이유가?
            Q = attn[:, 0 * n_embd : 1 * n_embd]
            K_curr = attn[:, 1 * n_embd : 2 * n_embd]
            V_curr = attn[:, 2 * n_embd : 3 * n_embd]

            # [질문] 메모리에 저 값을 저장하는 이유와 K-curr의 벡터 차원과 메모리에 저장되는 차원이 다른거 같은데 제가 잘 못이해하고 있는걸까요?
            self.k_memory[l, n_past : n_past + n, :] = K_curr
            self.v_memory[l, n_past : n_past + n, :] = V_curr
            # [질문] 트랜스포즈를 해야 하는 이유를 명확하게 모르겠습니다. 결국 데이터의 순서?를 바꿔주는 것 같은데 이 부분이 이해가 잘 안됩니다.
            Q = Q.reshape(n, n_head, n_embd // n_head).transpose(1, 2, 0)
            K = (
                self.k_memory[l, : n_past + n, :]
                .reshape(n_past + n, n_head, n_embd // n_head)
                .transpose(1, 0, 2)
            )
            V = (
                self.v_memory[l, : n_past + n, :]
                .reshape(n_past + n, n_head, n_embd // n_head)
                .transpose(1, 0, 2)
            )
            W = np.einsum("hnd,hpd->hnp", Q, K) / np.sqrt(V.shape[-1])
            """
            Q와 K의 내적을 계산하고 그 결과를 정규화
            h: 헤드 수
            n: 현재 시퀀스 길이(쿼리의 차원)
            p: 이전 시퀀스와 현재 시퀀스의 길이 합(키의 차원)
            d: 각 헤드의 임베딩 차원
            """
            # 어텐션 마스크 생성
            mask = attention_mask(W.shape[1], W.shape[2])
            # 마스크가 0인 경우 -1 * 1e10을 해줌으로써 어텐션 점수를 매우 낮춤
            W = W - (1.0 - mask) * 1e10
            W = softmax(W, axis=-1)
            h = np.einsum("hab,hby->hay", W, V).transpose((1, 0, 2)).reshape(n, -1)

            # projection layer
            """
                h%d/attn/c_proj/w [n_embd, n_embd]
                h%d/attn/c_proj/b [n_embd]
            """
            h = linear(
                h,
                self.tensors["h%d/attn/c_proj/w" % l],
                self.tensors["h%d/attn/c_proj/b" % l],
            )

            # residual connection
            input_ff = input_layer + h
            """
            feed-foward input normalization
            h%d/ln_1/g shape [n_embd]
            h%d/ln_1/b shape [n_embd]
            """
            h = normalize(
                input_ff, self.tensors["h%d/ln_2/g" % l], self.tensors["h%d/ln_2/b" % l]
            )

            """
            feed-forward layer
            h%d/mlp/c_fc/w [n_embd, n_embd * 3]
            h%d/mlp/c_fc/b [n_embd * 3]
            h%d/mlp/c_proj/w [n_embd * 3, n_embd]
            h%d/mlp/c_proj/b [n_embd]
            """
            h = linear(
                h,
                self.tensors["h%d/mlp/c_fc/w" % l],
                self.tensors["h%d/mlp/c_fc/b" % l],
            )
            h = gelu(h)
            h = linear(
                h,
                self.tensors["h%d/mlp/c_proj/w" % l],
                self.tensors["h%d/mlp/c_proj/b" % l],
            )
            input_layer = h + input_ff

        # [질문] input_layer를 복사하는 이유는? 이 과정에서는 이미 연산이 끝난게 아닌가요?
        emb = input_layer.copy()
        # final normalization
        emb = normalize(emb, self.tensors["ln_f/g"], self.tensors["ln_f/b"])
        # head is tied with wte in gpt-2 model
        # 토큰 임베딩을 단어 확률로 변환
        # 마지막 토큰의 임베딩 값에 전체 토큰 임베딩 확률을 곱해서 다음 단어를 예측하는 확률을 계산
        lm_head = self.tensors["wte"].T
        logits = np.matmul(emb[-1], lm_head)
        return logits


if __name__ == "__main__":  # test
    dir_model = "./models/gpt-2-117M"
    model = gpt2Model()
    model.load(dir_model)
