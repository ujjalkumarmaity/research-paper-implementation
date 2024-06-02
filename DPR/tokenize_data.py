from transformers import BertTokenizer,DistilBertTokenizer
from datasets import load_from_disk
from datasets import set_caching_enabled
import argparse
from datasets import load_dataset
import numpy as np
set_caching_enabled(False)

def load_hf_data(PATH):
    "load HF data"
    dataset = load_from_disk(PATH,keep_in_memory=True) # biencoder-nq-train-tokenize-data-bert.hf
    return dataset

def save_hf_data(dataset,PATH):
    "load HF data"
    dataset.save_to_disk(PATH)

class HFBertTokenizer():
    "initilize HF tokenizer"
    def __init__(self,tokenizer:BertTokenizer,max_length :int,pad_to_max:bool ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max
    def text_to_tensor(self,text,title=None,apply_max_len=True,add_special_tokens=True):
        if title:
            token_ids = self.tokenizer(title,text_pair=text,return_tensors='pt',max_length=self.max_length,truncation=True,padding="max_length")
        else:
            token_ids = self.tokenizer(text,return_tensors='pt',max_length=self.max_length,truncation=True,padding="max_length")

        return token_ids

    def get_tokenizer(self):
        return self.tokenizer
    
class HFDistilBertTokenizer(HFBertTokenizer):
    def __init__(self, tokenizer, max_length: int,pad_to_max:bool ) -> None:
        super().__init__(tokenizer, max_length,pad_to_max)

def tokenize_data(x,num_neg = 5 ,num_hard_neg = 5,hf_tokenizer = None):
    pos_ctxs = x['positive_ctxs']
    pos_ctx = pos_ctxs[np.random.choice(len(pos_ctxs))]
    neg_ctxs = x['negative_ctxs']# [:num_neg]
    hard_neg_ctxs = x['hard_negative_ctxs']# [:num_hard_neg]
    all_neg_ctxs = hard_neg_ctxs + neg_ctxs
#     random.shuffle(all_neg_ctxs
    all_neg_ctxs = all_neg_ctxs[:num_neg+num_hard_neg]
    
    q = x['question']
    q_tensor = hf_tokenizer.text_to_tensor(text=q)
    all_ctxs = [pos_ctx] + all_neg_ctxs
    neg_ctxs_title = ["" if i.get("title") is None else i.get("title") for i in all_ctxs]
    neg_ctxs_text = ["" if i.get("text") is None else i.get("text") for i in all_ctxs ]
    all_ctxs_tensor = hf_tokenizer.text_to_tensor(text=neg_ctxs_text,title=neg_ctxs_title)
    return {"q_input_ids":q_tensor['input_ids'],"q_attention_mask":q_tensor['attention_mask'],
            "all_ctxs_input_ids":all_ctxs_tensor['input_ids'],"all_ctxs_attention_mask":all_ctxs_tensor['attention_mask']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
    )
    args = parser.parse_args()

    dataset = load_dataset("json", data_files=args.data_path)
    num_hard_neg = 5
    num_neg = 5
    bert_tokenozer = BertTokenizer.from_pretrained('bert-base-uncased')
    hf_tokenizer = HFBertTokenizer(tokenizer=bert_tokenozer,max_length=200,pad_to_max = True)
    

    dataset = dataset.map(tokenize_data,
                                num_proc=2,
                                fn_kwargs={"num_hard_neg": num_hard_neg, "num_neg": num_neg,"hf_tokenizer":hf_tokenizer},
                                remove_columns=[ 'hard_negative_ctxs', 'question', 'negative_ctxs', 'positive_ctxs'])


    save_hf_data(dataset,".data/biencoder-nq-train-tokenize-data-bert.hf")

if __name__=="__main__":
    main()