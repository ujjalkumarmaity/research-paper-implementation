from config import cfg
from typing import Dict, List,Any
import faiss
class VectorSearchFaiss(object):
    def __init__(self, embed_dim = None):
        self.embed_dim = cfg.OPENSEARCH_EMBED_DIM if not embed_dim else embed_dim
        self.index = faiss.IndexFlatIP(self.embed_dim)        

    def add_batch_embeddings(self, embeddings: List[List[float]]):
        """
        Bulk index documents.
        Each document must have: 'embedding' (list of floats), 'text', 'source', etc.
        """
        self.index.add(embeddings)

    def search(self,embeddings,k):
        scores, indices = self.index.search(embeddings, k)
        return scores, indices


    