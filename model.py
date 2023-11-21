import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model # 512
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # 토큰을 임베딩 벡터로 매핑(길이 512로 통일)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # 논문 참고


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model  # positionalEmbedding도 Embedding vector과 같은 길이의 벡터
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # seq_len = max len of sentenc
        pe = torch.zeros(seq_len, d_model)  # matrix shape(seq_len, d_model)
        position = torch.arrange(0, seq_len, dtype=torch.float()).unsqueeze(1)  # vector shape(seq_len, 1)
        div = torch.exp(torch.arrange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div)  # 짝수 position: sin
        pe[:, 1::2] = torch.cos(position * div)  # 홀수 position: cos

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        

        self.register_buffer('pe', pe)  # 텐서 저장

    def forward(self, x): 
        # x.shape(batch, length of seq, d_model)
        # pe.shape(1, seq_len, d_model)
        x = x + (self.pe[:, :x.shape[1], :]).required_grad_(False)
        return self.dropout(x)
        
class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6 ):
        super().__init__()
        self.eps = eps
        # nn.Parameter -> parameter를 learnable하게
        self.gamma = nn.Parameter(torch.ones(1)) 
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x-mean) / (std+self.eps) + self.beta

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiheadAttention(nn.Module):

    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h  # # of attention heads
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h  # dim of attention head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod  # instance 호출 없이 멤버함수 사용 가능
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # (batch, h, d_k, seq_len)

        if mask is not None:   # mask: attention에서 상관없는 단어를 softmax에서 0으로 만들기 위함
            attention_scores.masked_fill_(mask==0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1)

        if dropout is not None:
            attention_scores = dropout(attention_scores) 

        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)  # -> (batch, h, seq_len, d_k), make matrix seq_len by d_k
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        # batch, sew_len의 dim은 유지, d_model을 h와 d_k로 나누기

        x, self.attention_scores = MultiheadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)  # -> (batch, seq_len, h, d_k)

        return self.w_o(x)



