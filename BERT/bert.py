"""HF Bert Model summary

BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(30522, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0-11): 12 x BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)
"""

import torch
import torch.nn as nn
import math
from transformers import BertTokenizer

class CustomBertEmbeddings(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,config.hidden_size,padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids",torch.arange(config.max_position_embeddings).expand(1,-1),persistent=False)
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(self,input_ids:torch.Tensor,token_type_ids:torch.Tensor = None,**kwargs):
        bs,T = input_ids.size()
        emb = self.word_embeddings(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.shape,dtype=torch.long,device=input_ids.device)
        emb += self.token_type_embeddings(token_type_ids)
        emb += self.position_embeddings(self.position_ids[:,:T])

        l_out = self.LayerNorm(emb)
        out = self.dropout(l_out)
        return out
    

class CustomBERTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query = nn.Linear(config.hidden_size,config.hidden_size)
        self.key = nn.Linear(config.hidden_size,config.hidden_size)
        self.value = nn.Linear(config.hidden_size,config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.num_attention_heads = config.num_attention_heads
        # assert config.hidden_size//self.num_attention_heads == 0
        self.head_dim = int(config.hidden_size / self.num_attention_heads)

    def forward(self,hidden_state:torch.Tensor,attention_mask:torch.Tensor):
        B,S,D = hidden_state.size()
        q = self.query(hidden_state).view(B,S,self.num_attention_heads,self.head_dim).transpose(2,1)
        k = self.key(hidden_state).view(B,S,self.num_attention_heads,self.head_dim).transpose(2,1)
        v = self.value(hidden_state).view(B,S,self.num_attention_heads,self.head_dim).transpose(2,1) # (B,NH,S,H_DIM)

        attention_score = torch.matmul(q , k.transpose(-1,-2))/math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_score.masked_fill(attention_mask == torch.tensor(False),float("-inf"))

        attention_weight = nn.functional.softmax(attention_score,dim=-1)
        attention_weight = self.dropout(attention_weight) # (B,NH,S,S)

        contex_vec = torch.matmul(attention_weight,v)
        # print(attention_score.shape)
        contex_vec = contex_vec.permute(0,2,1,3)
        contex_vec = contex_vec.reshape(hidden_state.size())
        return contex_vec
        
class CustomBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size,config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps,elementwise_affine=True)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    def forward(self,self_attention_out:torch.Tensor,hidden_state:torch.Tensor):
        return self.LayerNorm(self.dropout(self.dense(self_attention_out)) + hidden_state)
    

class CustomBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = CustomBERTSelfAttention(config=config)
        self.output = CustomBertSelfOutput(config=config)

    def forward(self,hidden_state:torch.Tensor,attention_mask:torch.Tensor = None):
        ct = self.self(hidden_state,attention_mask)
        attention_out = self.output(ct,hidden_state)
        return attention_out


class CustomBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size,config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
    def forward(self,hidden_state):
        return self.intermediate_act_fn(self.dense(hidden_state))
    
class CustomBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size,config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps,elementwise_affine=True)
        
    def forward(self,hidden_state,input_tensor):
        return self.LayerNorm(self.dropout(self.dense(hidden_state)) + input_tensor)


class CustomBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CustomBertAttention(config=config)
        self.intermediate = CustomBertIntermediate(config=config)
        self.output = CustomBertOutput(config=config)

    def forward(self,input_ids:torch.Tensor,attention_mask:torch.Tensor):
        ct_vec = self.attention(input_ids,attention_mask)
        im_out = self.intermediate(ct_vec)
        bert_out = self.output(im_out,ct_vec)
        return bert_out
    

class CustomBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([CustomBertLayer(config=config) for _ in range(config.num_hidden_layers)])
    def get_extended_attention_mask(self,attention_mask:torch.Tensor):
        if len(attention_mask.size())==2:
            return attention_mask[:,None,None,:]
        elif len(attention_mask.size())==3:
            return NotImplementedError()
        else:
            return attention_mask
    def forward(self,input_ids:torch.Tensor,attention_mask:torch.Tensor):
        attention_mask = self.get_extended_attention_mask(attention_mask=attention_mask)
        out = input_ids
        for layer in self.layer:
            out = layer(out,attention_mask)
        return out
    

class CustomBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size,config.hidden_size)
        self.activation = nn.Tanh()
    def forward(self,encoder_out:torch.Tensor):
        return self.activation(self.dense(encoder_out[:,0])) # take on;y CLS
    

class CustomBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = CustomBertEmbeddings(config=config)
        self.encoder = CustomBertEncoder(config=config)
        self.pooler = CustomBertPooler(config=config)
    def forward(self,input_ids:torch.Tensor,token_type_ids:torch.Tensor,attention_mask:torch.Tensor):
        em_ot = self.embeddings(input_ids,token_type_ids)
        encoder_out = self.encoder(em_ot,attention_mask)
        out = self.pooler(encoder_out)
        return {"last_hidden_state":encoder_out,"pooler_output":out}
    
    @classmethod
    def from_pretrained(cls,pretrained_model_name_or_path):
        from transformers import BertModel,BertConfig
        hf_model = BertModel.from_pretrained(pretrained_model_name_or_path)
        hf_model_weight = hf_model.state_dict()

        config = BertConfig.from_pretrained(pretrained_model_name_or_path)

        custom_model = CustomBertModel(config=config)
        custom_model_sd = custom_model.state_dict()
        for name,_ in custom_model_sd.items():
            with torch.no_grad():
                custom_model_sd[name].copy_(hf_model_weight[name])

        del hf_model_weight
        return custom_model

# def main():
#     model_name = "bert-base-uncased"
#     model = CustomBertModel.from_pretrained(model_name)
#     tokenizer = BertTokenizer.from_pretrained(model_name)
#     data = tokenizer(["Implemented bert from scratch"],return_tensors="pt")
#     print(data)
#     out = model(**data)
#     print(out.shape)

# if __name__=="__main__":
#     main()
        

