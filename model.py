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

        # seq_len = max len of sentence
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
        # batch, seq_len의 dim은 유지, d_model을 h와 d_k로 나누기

        x, self.attention_scores = MultiheadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)  # -> (batch, seq_len, h, d_k)

        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiheadAttention, feed_forward_block: FeedForward, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])  # ResidualConnection 모듈 2개 리스트에 저장

    def forward(self, x, src_mask):   # mask for padding
        x = self.residual_connections[0](x, self.self_attention_block(x, x, x, src_mask))  # self attention
        x = self.feed_forward_block[1](x, self.feed_forward_block)
        return x 
    
class Encoder(nn.Module):  # N encoders

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiheadAttention, cross_attention_block: MultiheadAttention, feed_forward_block: FeedForward, dropout):
        super().__init__()
        self.self_attention_block = self.self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, self.self_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x 
    
class Decoder(nn.Module):  # N decoders

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

                                         
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim = -1)  # log softmax for stability


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbedding, tgt_embedding: InputEmbedding,
                src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, tgt, src_mask, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(encoder_output, src_mask, tgt_mask)
    
    def projection(self, x):
        return self.projection_layer(x)

# Initialize all the parameters and give to one transformer model
def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, 
                      d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048):
    # Embedding layers
    src_embedding = InputEmbedding(d_model, src_vocab_size)
    tgt_embedding = InputEmbedding(d_model, tgt_vocab_size)

    # Positioanl Embedding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Encoder Blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Decoder Blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiheadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.Modulelist(decoder_blocks))

    # Projection Layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # transformer
    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer


