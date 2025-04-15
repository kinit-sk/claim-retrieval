import logging
import torch
import faiss
import numpy as np
from typing import Any, List

from src.retrievers.retriever import Retriever
from src.retrievers.vectorizers.vectorizer import Vectorizer
from src.datasets.dataset import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Faiss(Retriever):
    def __init__(self, 
                 name: str = 'faiss', 
                 top_k: int = 5,
                 vectorizer_document: Vectorizer = None,
                 vectorizer_query: Vectorizer = None,
                 device: str = 'cpu',
                 save_if_missing: bool = False,
                 dataset: Dataset = None, 
                 use_post: bool = False,
                 **kwargs: Any):
        
        super().__init__(name, top_k, dataset=dataset)
        self.vectorizer_document = vectorizer_document
        self.vectorizer_query = vectorizer_query
        self.device = device
        self.save_if_missing = save_if_missing
        self.document_embeddings = None
        self.index = None
        self.use_post = use_post

        self.create_index()

    def create_index(self):
        logger.info('Creating index with faiss.')
        
        self.calculate_embeddings()
        vectors = self.document_embeddings.cpu().numpy()
        ids = np.array(self.dataset.get_documents_ids())
        dim = vectors.shape[1]

        index_flat = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

        if self.device == 'cuda':
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        else:
            self.index = index_flat
        
        self.index.add_with_ids(vectors, ids)
        
        logger.info(f'Index with {self.index.ntotal} vectors created.')
    
    def calculate_embeddings(self):
        logger.info('Calculating embeddings for fact checks.')
        if self.dataset is None:
            documents = []
        else:
            documents = self.dataset.get_documents_texts()
            
            self.document_embeddings = self.vectorizer_document.vectorize(
                documents,
                save_if_missing=self.save_if_missing,
                normalize=True
            )

    def set_dataset(self, dataset: Dataset):
        self.dataset = dataset
        self.calculate_embeddings()
        self.create_index()

    def retrieve(self, query: dict) -> Any:
        if self.document_embeddings is None:
            self.calculate_embeddings()
            self.create_index()
        
        queries = []
        if self.use_post:
            queries = query['input']
        else:
            queries = query['prompt']
        
        logger.info(f'Calculating embeddings for {len(queries)} queries.')
        query_embeddings = self.vectorizer_query.vectorize(
            queries,
            save_if_missing=self.save_if_missing,
            normalize=True
        )
        query_vectors = query_embeddings.cpu().numpy()

        sims, retrieved_ids = self.index.search(query_vectors, self.top_k)
        
        retrieved_texts = [
            [self.dataset.get_document(doc_id) if doc_id >=0 else '' for doc_id in row]
            for row in retrieved_ids
        ]
        
        similarities = [
            [sim for sim in row]
            for row in sims
        ]
        
        if len(queries) > 1:
            return retrieved_texts, (retrieved_ids, similarities)
        else:
            return retrieved_texts[0], (retrieved_ids[0], similarities[0])