# Imnplemented GPT-2 Small from scratch
import torch
import torch.nn as nn
import math
from typing import List
import tiktoken
import argparse

# load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

def tokenize(text:str)->List:
    return tokenizer.encode(text)

# Multi-Haed Attention Layer

class MultiHeadAttention(nn.Module):
    def __init__(self,din,dout,context_vec_legth,num_head,dropout = 0.1,kqv_bias = True):
        super().__init__()
        self.dout = dout
        self.w_key = nn.Linear(din,dout,bias=kqv_bias)
        self.w_query = nn.Linear(din,dout,bias=kqv_bias)
        self.w_value = nn.Linear(din,dout,bias=kqv_bias)
        self.out_proj = nn.Linear(dout, dout)  # Linear layer to combine head outputs

        self.dropout = nn.Dropout(dropout)
        self.head_dim = din//num_head
        self.num_head = num_head

        self.register_buffer("mask",torch.triu(torch.ones(context_vec_legth,context_vec_legth),diagonal=1))

    def forward(self,x:torch.Tensor):
        bs,num_token, dim = x.size()

        key = self.w_key(x)
        query = self.w_query(x)
        value = self.w_value(x)
        # print(self.num_head,self.head_dim)
        key = key.view(bs,num_token,self.num_head,self.head_dim)
        query = query.view(bs,num_token,self.num_head,self.head_dim)
        value = value.view(bs,num_token,self.num_head,self.head_dim)

        key = key.transpose(1,2)
        query = query.transpose(1,2)
        value = value.transpose(1,2)

        attention_score = query @ key.transpose(2,3)

        attention_score.masked_fill_(self.mask.bool()[:num_token,:num_token],-torch.inf)

        k_dim = key.size()[-1]
        att_weights = torch.softmax(attention_score/k_dim**0.5,axis=-1)

        att_weights = self.dropout(att_weights)
        context_vector = (att_weights @ value).transpose(1,2)
        context_vector = context_vector.contiguous().view(bs,num_token,self.dout)
        context_vector = self.out_proj(context_vector)
        return context_vector

# Layer Normalizatio Layer
class LayerNorm(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.eps = 1e-5
        self.gama = nn.Parameter(torch.ones(dim)) # scale 
        self.beta = nn.Parameter(torch.zeros(dim)) # shift
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        mean = x.mean(dim=-1,keepdim=True)

        var = x.var(dim=-1,keepdim=True,unbiased=False)
        return (x - mean)/(torch.sqrt(var + self.eps)) *self.gama + self.beta

# GELU(Gaussian Error Linear Unit) Activation function 
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5 * x * (1.0+ torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * torch.pow(x,3))))

# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.ffn = nn.Sequential(*[
            nn.Linear(emb_dim,4*emb_dim),
            GELU(),
            nn.Linear(4*emb_dim,emb_dim)
        ])
    def forward(self,x) -> torch.Tensor:
        return self.ffn(x)

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb_dim = config["emb_dim"]
        self.n_heads = config["n_heads"]
        self.drop_rate = config["drop_rate"]
        self.context_length = config["context_length"]
        self.qkv_bias = config["qkv_bias"]

        self.layer_norm1 = LayerNorm(self.emb_dim)
        self.layer_norm2 = LayerNorm(self.emb_dim)

        self.mha = MultiHeadAttention(din = self.emb_dim,
                                      dout = self.emb_dim,
                                      context_vec_legth=self.context_length,
                                      num_head=self.n_heads,
                                      dropout=self.drop_rate,
                                      kqv_bias=self.qkv_bias
                                      )
        self.ffn = FeedForward(emb_dim = self.emb_dim)
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)
        
    def forward(self,x:torch.Tensor)-> torch.Tensor:
        skip_conn = x
        out = self.layer_norm1(x)
        out = self.mha(out) 
        out = self.dropout1(out)
        out = out + skip_conn

        skip_conn = out
        out = self.layer_norm2(out)
        out = self.ffn(out) # Feed forward network
        out = self.dropout1(out)
        out = out + skip_conn # skip connectio

        return out

# GPT2 Model    
class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_em = nn.Embedding(config["vocab_size"],config["emb_dim"])
        self.pe = nn.Embedding(config["context_length"],config["emb_dim"])
        self.tf_blocks = nn.Sequential(*[
            TransformerBlock(config=config) for _ in range(config["n_layers"])
        ])
        self.layer_norm = LayerNorm(config["emb_dim"])
        self.out_layer = nn.Linear(config["emb_dim"],config["vocab_size"])
        self.dropout = nn.Dropout(config["drop_rate"])
        
    def forward(self,x):
        bs,num_tok = x.size()
        em_out = self.tok_em(x)
        pe_out = self.pe(torch.arange(num_tok))
        out = em_out + pe_out

        out = self.dropout(out)
        out = self.tf_blocks(out)

        out = self.layer_norm(out)
        logist = self.out_layer(out)
        return logist
    
def generating_text(model,tok_id,conext_len,max_new_token):
    for _ in range(max_new_token):
        tok_id = tok_id[:,:conext_len]
        with torch.no_grad():
            logist = model(tok_id)
        out = logist[:,-1:,]
        next_tok = torch.argmax(torch.softmax(out,dim=-1),dim=-1,)
        tok_id = torch.cat([tok_id,next_tok],dim=1)
    return tok_id

def number_of_trainable_parameter(model):
    return sum([i.numel() for i in model.parameters()])

def main():
    # parser = argparse.ArgumentParser()
    gpt2_config = {
        "vocab_size": 50257,    
        "context_length": 1024, 
        "emb_dim": 768,
        "n_heads": 12,      
        "n_layers": 12,        
        "drop_rate": 0.1,     
        "qkv_bias": False    
    }
    model = GPT2Model(gpt2_config)
    num_param = number_of_trainable_parameter(model)
    print(f"number of parameter - {round(num_param/1e6,2)} B")

    inp = torch.tensor(tokenize("Implement GPT from Scratch "))
    inp = inp.unsqueeze(0)
    output_token_ids = generating_text(model=model,tok_id=inp,conext_len=1024,max_new_token=10)

    print(tokenizer.decode(output_token_ids.squeeze(0).tolist()))

if __name__=="__main__":
    main()