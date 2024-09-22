import sys
import json
import numpy as np
import tensorflow as tf
from scipy.special import softmax

#Gaussian Error Linear Unit
def gelu(x):
    '''
    - Relu와 같이 사용되는 활성화 함수로 모델의 출력을 제어
    - GeLU 함수는 입력값에 대해 부드러운 비선형성을 제공하며 더 복잡한 패턴을 학습 할 수 있도록 도와줌
    - ReLU는 음수 입력값에 대해 0을 출력하는 특성이 있으나 GeLU는 음수 입력값에 대해서도 0이 아닌 출력을 제공
    - GeLU는 NLP 모델에서 많이 사용되며 ReLU는 CNN 모델에서 많이 사용
    '''
    
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

#Linear
def linear(emb, w, b):
    '''
    - 신경망의 각 레이어에서 입력값을 다음 레이어로 전달할 때 수행되는 기본 연산
    - 입력값에 가중치를 적용하고 편향을 더하여 선형 변환을 수행
    - 입력값의 차원 수와 가중치의 차원 수에 따라 출력값의 차원 수가 결정됨
    - 모델이 데이터의 특성을 학습하고 복잡한 패턴을 예측하는 데 사용됨
    - 일반적으로 신경망의 각 레이어에서 사용되며, 활성화 함수와 함께 사용되어 모델의 출력을 제어함
    '''
    if b.shape[0] != w.shape[1]:
        raise ValueError("Bias dimension is not match with weight dimension")
    
    return np.matmul(emb, w) + b[np.newaxis, :]

#Layer Normalization
def normalize(x, g, b, eps=1e-10):
    '''
    - normalize 함수는 입력값 x를 정규화하여 분포의 평균이 0이고 표준편차가 1이 되도록 함.
    - 정규화 과정은 입력값을 표준화하여 모델의 학습을 안정화하고, 가중치 업데이트를 더 효율적으로 할 수 있게 함
    - g와 b는 모델이 학습하는 동안 학습되는 가중치와 편향 파라미터로 각각 정규화된 값에 곱해지고 더해짐
    - eps는 분산이 0이 되는 것을 방지하기 위한 작은 값
    '''
    #axis=-1은 마지막 축을 의미하며, keepdims=True는 축을 유지하도록 함
    #x = np.array([[1, 2, 3], [4, 5, 6]]) 이고 axis가 0일 경우 출력은 [2.5, 3.5, 4.5]가 됨
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    sqrtd = np.sqrt(var + eps)
    h = (x - mean) / sqrtd
    
    return h * g[np.newaxis, :] + b[np.newaxis, :]

#Attention Mask
def attention_mask(num_decoder_tokens, num_sequence_tokens):
    '''
    - 디코더에서 입력값이 하나씩 늘어날 때마다 어텐션을 제한하는 마스크를 생성
    - 미래의 정보를 참조하지 못하도록 하여 모델의 학습을 안정화
    - 마스크 값이 True일 때 어텐션은 허용되고, False일 때는 어텐션이 제한됨
    '''

    i = np.arrange(num_decoder_tokens)[:, None]
    j = np.arrange(num_sequence_tokens)
    mask = i >= j - num_sequence_tokens + num_decoder_tokens

    return mask

class gpt2Model:
    def __init__(self):
        #모델이 처리 할 수 있는 총 단어의 수
        self.n_vocab = None
        #모델이 처리 할 수 있는 최대 시컨스 길이
        self.n_ctx = None
        #임베딩 차원 수
        self.n_embd = None
        #모델의 어텐션 헤드 수
        self.n_head = None
        #모델의 레이어 수
        self.n_layer = None
        #키 벡터 저장소
        self.k_memory = None
        #밸류 벡터 저장소
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
        self.k_momory = np.zeros(shape=(self.n_layer, self.n_ctx, self.n_embd), dtype=np.float32)
        self.v_memory = np.zeros(shape=(self.n_layer, self.n_ctx, self.n_embd), dtype=np.float32)
        #모델의 weight, bias 등 다양한 텐서 데이터를 저장하기 위한 딕셔너리
        self.tensors = {}
        #각 텐서 데이터의 차원 정보를 저장하기 위한 딕셔너리
        self.shapes = {}

        

        
        