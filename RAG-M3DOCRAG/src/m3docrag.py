"""
M3DOCRAG (Multi-modalMulti page Multi-Document Retrieval-Augmented Generation
"""

import torch
import torch.nn as nn
from colpali_engine.models import ColPali, ColPaliProcessor
from PIL import Image
from colpali_engine.models import ColPaliProcessor
from torch.utils.data import DataLoader
from datasets import Dataset as HfDataset
import numpy as np
from transformers import AutoProcessor,AutoModelForImageTextToText
from typing import List
import argparse
from vector_store import VectorSearchFaiss
from config import cfg
from data import M3DocragDataSet


class DocEmbedding:
    vector_search_service:VectorSearchFaiss = None
    def __init__(self, dataset, retrieve_model_device_map = 'cuda:0'):
        self.model_name = cfg.COLPALI_MODEL# 'vidore/colpali-v1.2'
        self.dataset: HfDataset = dataset
        self.retrieve_model_device_map = retrieve_model_device_map

    def load_embedding_model(self):
        """ textual query q and page images P are projected into a shared multi-modal embedding space using ColPali """
        self.retrieval_model = ColPali.from_pretrained(
            cfg.COLPALI_BASE,
            torch_dtype=torch.bfloat16,
            device_map=self.retrieve_model_device_map
        ).eval()
        self.retrieval_model.load_adapter(self.model_name)
        self.retrieval_processor = ColPaliProcessor.from_pretrained(self.model_name)
    
    def build_index(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x:self.retrieval_processor.process_images([i['image'] for i in x])
        )
        page_embeddings = []
        for batch_doc in dataloader:
            with torch.no_grad():
                batch_doc = {k: v.to(self.retrieval_model.device) for k, v in batch_doc.items()}
                embeddings = self.retrieval_model(**batch_doc)
                mean_embedding = torch.mean(embeddings, dim=1).float().cpu().numpy()
                page_embeddings.extend(mean_embedding)
        
        actual_dim = torch.mean(embeddings, dim=1).shape[-1]
        self.vector_search_service = VectorSearchFaiss(embed_dim=actual_dim)
        
        self.vector_search_service.add_batch_embeddings(np.array(page_embeddings))

    def retrieve(self, query:str, k = 1)->List[HfDataset]:
        inputs = self.retrieval_processor.process_texts(query,)
        inputs = {k: v.to(self.retrieval_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            embeddings = self.retrieval_model(**inputs)
        embeddings = torch.mean(embeddings, dim=1).float().cpu().numpy()[0].reshape(1, -1)
        scores, indices = self.vector_search_service.search(embeddings=embeddings,k=k)
        retrieved_docs = [self.dataset[ind] for ind in indices]
        print(f"Retrieve Document Index - {indices}")
        return retrieved_docs
    
class M3DOCRAG:
    def __init__(self,qa_model_device = 'cuda:1'):
        self.qa_model_name = cfg.QA_MODEL
        self.qa_model_device = qa_model_device

    def load_qa_model(self):
        self.qa_processor = AutoProcessor.from_pretrained(self.qa_model_name)
        self.qa_model = AutoModelForImageTextToText.from_pretrained(self.qa_model_name,device_map=self.qa_model_device)
    
    def format_chat_template(self,query:str, retrieved_docs:List):
        messages = [{
            "role": "system",
                    "content": """You are a document analyzer that ONLY gives two types of responses:
                1. If you find the EXACT information: Respond with ONLY that specific information
                2. If you cannot find the EXACT information: Respond with EXACTLY and ONLY this phrase: "I cannot find this information in the provided document pages."

                DO NOT:
                - Explain your limitations
                - Talk about AI or models
                - Make assumptions
                - Give partial information
                - Provide multiple answers
                - Add any explanations"""
        }, 
        {
            "role": "user",
            "content": [
                *[{
                    "type": "image",
                    "image": doc['image']
                } for doc in retrieved_docs],
                {
                    "type": "text",
                    "text": f"IMPORTANT: Give ONLY the exact answer found in these document pages for: {query}"
                }
            ]
        }]
        return messages
    
    def generate_qa(self,query, retrieved_docs):
        message = self.format_chat_template(query=query, retrieved_docs=retrieved_docs)
        process_text = self.qa_processor.apply_chat_template(
                message, 
                tokenize=False, 
                add_generation_prompt=True
            )
        inputs = self.qa_processor(
            text=[process_text], images=[ind['image'] for ind in retrieved_docs], padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.qa_model.device)
        output_ids = self.qa_model.generate(**inputs, max_new_tokens=cfg.MAX_NEW_TOKENS)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.qa_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        # print(f"answer - {output_text}")
        return output_text

    def process_query(self,doc_embedding:DocEmbedding, query:str):
        retrieved_docs = doc_embedding.retrieve(query=query, k=1)
        output_text = self.generate_qa(query=query,retrieved_docs=retrieved_docs)
        return output_text

    def m3docrag_chat(self,doc_embedding):
        print("\n" + "="*80)
        print("M3DOCRAG Chat")
        print("="*80 + "\n")
        
        while True:
            try:
                query = input("\nYou: ").strip()
                if query.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    break
                if not query:
                    continue
                
                # Get answer
                answer = self.process_query(doc_embedding, query)
                print("\nAssistant: ", end="", flush=True)
                print(answer)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', choices=['sample','DocVQA'],default='sample')
    return args.parse_args()

def main():
    args = get_args()
    assert torch.cuda.is_available()
    assert torch.cuda.device_count()>1
    m3_dataset = M3DocragDataSet(dataset_name=args.dataset)
    dataset = m3_dataset.load_data()

    doc_embedding = DocEmbedding(dataset=dataset,retrieve_model_device_map='cuda:0' )
    doc_embedding.load_embedding_model()
    doc_embedding.build_index()

    m3rag = M3DOCRAG()
    m3rag.load_qa_model()
    m3rag.m3docrag_chat(doc_embedding)

if __name__=='__main__':
    main()
