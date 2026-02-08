from dotenv import load_dotenv
import os
load_dotenv()

class Config:
    OPENSEARCH_INDEX_NAME = 'm3docrag_embedding_index'
    OPENSEARCH_EMBED_DIM = 128
    # Model configs
    COLPALI_MODEL = 'vidore/colpali-v1.2'
    COLPALI_BASE = 'vidore/colpaligemma-3b-mix-448-base'
    QA_MODEL = 'Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4'
    
    # Processing configs
    BATCH_SIZE = 4
    MAX_NEW_TOKENS = 128
    RETRIEVAL_TOP_K = 1

cfg = Config()