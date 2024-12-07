# compare BERT HF and our custom model 
from transformers import BertModel,BertTokenizer
import unittest
from bert import CustomBertModel
import torch
class TestCustomBert(unittest.TestCase):
    def test_bert(self):
        model_name = "bert-base-uncased"
        hf_model = BertModel.from_pretrained(model_name)
        custom_model = CustomBertModel.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)

        x = tokenizer(["Implemented bert from scratch"],return_tensors="pt")
        print(x)
        hf_model.eval()
        custom_model.eval()
        
        hf_logist = hf_model(**x)
        cs_logist = hf_model(**x)

        self.assertTrue(torch.allclose(hf_logist["last_hidden_state"],cs_logist["last_hidden_state"]))
        self.assertTrue(torch.allclose(hf_logist["pooler_output"],cs_logist["pooler_output"]))

if __name__ == '__main__':
    unittest.main()