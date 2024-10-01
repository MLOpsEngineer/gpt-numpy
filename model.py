import sys
import json
import numpy as np
import tensorflow as tf
from scipy.special import softmax


# Gaussian Error Linear Unit
def gelu(x):
    """
    - GeLU(가우시안 오류 선형 유닛)는 입력값에 부드러운 비선형성을 제공하는 활성화 함수입니다.
    - ReLU와 달리 음수 입력값에 대해서도 0이 아닌 출력을 제공합니다.
    - NLP 모델에서 많이 사용됩니다.
    """

    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


# Linear
def linear(emb, w, b):
    """
    - 입력값에 가중치를 곱하고 편향을 더하여 선형 변환을 수행합니다.
    - 입력 차원과 가중치 차원에 따라 출력 차원이 결정됩니다.
    - 신경망의 각 층에서 활성화 함수와 함께 사용되어 모델의 출력을 제어합니다.
    """
    #편향 b의 차원이 가중치 행렬 w의 열 차원과 일치하는지 확인
    if b.shape[0] != w.shape[1]:
        raise ValueError("Bias dimension is not match with weight dimension")

    return np.matmul(emb, w) + b[np.newaxis, :]


# Layer Normalization
def normalize(x, g, b, eps=1e-10):
    """
    - 입력값 x를 정규화하여 평균이 0이고 분산이 1이 되도록 합니다.
    - 정규화된 값에 학습 가능한 가중치 g를 곱하고 편향 b를 더합니다.
    - 모델의 학습을 안정화하고 가중치 업데이트를 효율적으로 수행할 수 있게 합니다.
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
    - 디코더의 각 시간 단계에서 미래 정보를 참조하지 못하도록 마스크를 생성합니다.
    - 마스크는 어텐션 점수를 계산할 때 사용됩니다.
    - True인 위치는 어텐션이 허용되고, False인 위치는 어텐션이 제한됩니다.
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
        # 트랜스포머의 블록의 수(디코더)
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
        # 현재 연산중인 부분만 처리
        # n_past는 이전에 모델이 처리한 토큰의 수를 나타냄
        # GPT-2는 자기회귀 모델로, 이전 컨텍스트를 기반으로 다음 단어를 예측함
        # 따라서 현재 토큰의 위치를 정확히 나타내기 위해 이전 토큰의 수를 고려해야 함
        # 만약 이전 토큰을 고려하지 않고 i만 사용하면, 모델은 전체 시퀀스에서의 정확한 위치 정보를 잃게 됨
        for i in range(n):
            pos_indices[i] = i + n_past

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
            #논문과 다르게 attention 값을 3등분하여 사용함
            Q = attn[:, 0 * n_embd : 1 * n_embd]
            K_curr = attn[:, 1 * n_embd : 2 * n_embd]
            V_curr = attn[:, 2 * n_embd : 3 * n_embd]

            #메모리에 각 이전에 연산한 결과를 저장해둠
            self.k_memory[l, n_past : n_past + n, :] = K_curr
            self.v_memory[l, n_past : n_past + n, :] = V_curr
            #Q는 (n_head, n, n_embd // n_head) 이렇게 바뀜
            Q = Q.reshape(n, n_head, n_embd // n_head).transpose(1, 0, 2)
            #K와 V는 (n_head, n_past + n, n_embd // n_head) 이렇게 바뀜
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
            W = np.einsum("hnd,hmd->hnp", Q, K) / np.sqrt(V.shape[-1])
            """
            Q와 K의 내적을 계산하고 그 결과를 정규화
            h: 헤드 수
            n: 현재 시퀀스 길이(쿼리의 차원)
            m: 이전 시퀀스와 현재 시퀀스의 길이 합(키의 차원)
            d: 각 헤드의 임베딩 차원
            """
            # 어텐션 마스크 생성
            mask = attention_mask(W.shape[1], W.shape[2])
            # 마스크가 0인 경우 -1 * 1e10을 해줌으로써 어텐션 점수를 매우 낮춤
            W = W - (1.0 - mask) * 1e10
            W = softmax(W, axis=-1)
            h = np.einsum("hnm,hmd->hnd", W, V).transpose((1, 0, 2)).reshape(n, -1)

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
