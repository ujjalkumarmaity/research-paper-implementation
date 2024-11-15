# Imnplemented GPT-2 Small from scratch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
import tiktoken
import argparse

# load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

def tokenize(text:str)->List:
    return tokenizer.encode(text)

# Multi-Haed Attention Layer

class MultiHeadAttention1(nn.Module):
    def __init__(self,n_embd,n_ctx,num_head,attn_pdrop = 0.1,resid_pdrop = 0.1,qkv_bias = True):
        super().__init__()
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd,3*n_embd,bias=qkv_bias)
        self.c_proj = nn.Linear(n_embd, n_embd,bias=qkv_bias)  # Linear layer to combine head outputs

        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        self.head_dim = n_embd//num_head
        self.num_head = num_head

        self.register_buffer("masked_bias",torch.tril(torch.ones(n_ctx,n_ctx)).view(1,1,n_ctx,n_ctx))

    def forward(self,x:torch.Tensor):
        bs,num_token, e_dim = x.size()
        query, key, value = self.c_attn(x).split(self.n_embd,dim=2) # (bs,num_tok,dim)

        key = key.view(bs,num_token,self.num_head,self.head_dim)
        query = query.view(bs,num_token,self.num_head,self.head_dim)
        value = value.view(bs,num_token,self.num_head,self.head_dim)

        key = key.transpose(1,2)
        query = query.transpose(1,2)
        value = value.transpose(1,2)

        attention_score = (query @ key.transpose(2,3)) * (1.0 / math.sqrt(key.size(-1)))

        attention_score.masked_fill_(self.masked_bias[:,:,:num_token,:num_token]==0,float("-inf"))
        k_dim = key.size()[-1]
        att_weights = F.softmax(attention_score,dim=-1)
        
        att_weights = self.attn_dropout(att_weights)

        context_vector = (att_weights @ value)
        context_vector = context_vector.transpose(1,2).contiguous().view(bs,num_token,self.n_embd)
        out = self.resid_dropout(self.c_proj(context_vector))
        return out
    

# Layer Normalizatio Layer
class LayerNorm(nn.Module):
    def __init__(self,n_embd,eps=1e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd)) # scale 
        self.bias = nn.Parameter(torch.zeros(n_embd)) # shift
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        mean = x.mean(dim=-1,keepdim=True)

        var = x.var(dim=-1,keepdim=True,unbiased=False)
        return (x - mean)/(torch.sqrt(var + self.eps)) *self.weight + self.bias

# GELU(Gaussian Error Linear Unit) Activation function 
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5 * x * (1.0+ torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * torch.pow(x,3))))


# Feed Forward Network
# (mlp): GPT2MLP(
#   (c_fc): Conv1D()
#   (c_proj): Conv1D()
#   (act): NewGELUActivation()
#   (dropout): Dropout(p=0.1, inplace=False)
# )
class GPT2MLP(nn.Module):
    def __init__(self, n_embd,dropout = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embd,4*n_embd)
        self.c_proj = nn.Linear(4*n_embd,n_embd)
        self.act = GELU()
        self.dropout = nn.Dropout(p = dropout)

    def forward(self,x) -> torch.Tensor:
        return self.dropout(self.c_proj((self.act(self.c_fc(x)))))
    

# Transformer Block
class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config["n_embd"]
        self.n_head = config["n_head"]
        self.attn_pdrop = config["attn_pdrop"]
        self.resid_pdrop = config["resid_pdrop"]
        self.n_ctx = config["n_ctx"]
        self.qkv_bias = config["qkv_bias"]

        self.ln_1 = LayerNorm(n_embd = self.n_embd,eps = 1e-05)
        

        self.attn = MultiHeadAttention1(n_embd = self.n_embd,
                                      n_ctx=self.n_ctx,
                                      num_head=self.n_head,
                                      attn_pdrop=self.attn_pdrop,
                                      resid_pdrop=self.resid_pdrop,
                                      qkv_bias=self.qkv_bias
                                      )
        self.ln_2 = LayerNorm(n_embd = self.n_embd,eps = 1e-05)
        self.mlp = GPT2MLP(n_embd = self.n_embd,dropout=self.attn_pdrop)
        # self.dropout = nn.Dropout(self.drop_rate)

        
    def forward(self,x:torch.Tensor)-> torch.Tensor:
        skip_conn = x
        out = self.ln_1(x)
        out = self.attn(out) 
        # out = self.dropout1(out)
        out = out + skip_conn

        skip_conn = out
        out = self.ln_2(out)
        out = self.mlp(out) # Feed forward network
        # out = self.dropout1(out)
        out = out + skip_conn # skip connectio

        return out

# GPT2 Model    
class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config["vocab_size"],config["n_embd"]) # (wte): Embedding(50257, 768)
        self.wpe = nn.Embedding(config["n_ctx"],config["n_embd"]) # (wpe): Embedding(1024, 768)
        self.drop = nn.Dropout(config["embd_pdrop"])
        self.h = nn.ModuleList([
            GPT2Block(config=config) for _ in range(config["n_layer"])
        ])
        self.ln_f = LayerNorm(config["n_embd"])

        
        
    def forward(self,x):
        bs,num_tok = x.size()
        em_out = self.wte(x)
        pe_out = self.wpe(torch.arange(num_tok))
        out = em_out + pe_out

        out = self.drop(out)
        for block in self.h:
            out = block(out)

        out = self.ln_f(out)
        # logist = self.lm_head(out)
        return out

class CustomGPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config["n_embd"],config["vocab_size"],bias=False)
    def forward(self,x):
        out = self.transformer(x)
        logit = self.lm_head(out)
        return logit

    @classmethod
    def from_pretrained(cls,model_name:str):
        assert "gpt" in model_name
        from transformers import GPT2Config,GPT2LMHeadModel
        config = GPT2Config.from_pretrained(model_name).to_dict()
        config.update({"qkv_bias":True})
        model = CustomGPT2LMHeadModel(config=config)
        cs_sd = model.state_dict()

        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        hf_sd = hf_model.state_dict()

        keys_cs = [k for k in cs_sd.keys() if not k.endswith('attn.masked_bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for i in keys_cs:
            if any(i for t in transposed if i.endswith(t)):
                with torch.no_grad():
                    cs_sd[i].copy_(hf_sd[i].t())
            else:
                with torch.no_grad():
                    cs_sd[i].copy_(hf_sd[i])
        return model


    @torch.no_grad
    def generate(self,inputs,max_new_token = 10):
        for _ in range(max_new_token):
            logits = self(inputs)
            out = logits[:,-1,:] # take last output
            last_tok_prob = F.softmax(out,dim=-1)
            last_tok_id = torch.argmax(last_tok_prob,dim=-1).unsqueeze(0)
            inputs = torch.cat([inputs,last_tok_id],dim=1)
        return inputs

def number_of_trainable_parameter(model):
    return sum([i.numel() for i in model.parameters()])

def main():
    model = CustomGPT2LMHeadModel.from_pretrained("gpt2")
    num_param = number_of_trainable_parameter(model)
    print(f"number of parameter - {round(num_param/1e6,2)} B")

    inp = torch.tensor(tokenize("Implement GPT from Scratch "))
    inp = inp.unsqueeze(0)
    out = model.generate(inp,max_new_token=10)
    print(tokenizer.decode(out.squeeze(0).tolist()))

if __name__=="__main__":
    main()