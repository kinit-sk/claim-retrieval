from typing import Any
from src.module import Module


class Retriever(Module):
    """
    The Retriever class is an abstract class that defines the interface for retrievers.
    
    Args:
        name (str): The name of the retriever.
        top_k (int): The number of documents to retrieve.
        dataset (Any): The knowledge base to use for retrieval.
    """
    def __init__(self, name: str = 'bm25', top_k: int = 5, dataset: Any = None, **kwargs):
        self.name = name
        self.top_k = top_k

        self.dataset = dataset

    def retrieve(self, query: str) -> Any:
        raise NotImplementedError
    
    def __call__(self, **kwargs: Any) -> Any:
        output = self.retrieve(**kwargs)
        query = kwargs['query']
        return self.convert_to_dict(query, output)

    def set_dataset(self, dataset: Any):
        self.dataset = dataset
    
    def convert_to_dict(self, query, output):
        return {
            'documents': output[0],
            'top_k': output[1],
            'query': query
        }
