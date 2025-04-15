from src.datasets.dataset import Dataset
from src.datasets.multiclaim.multiclaim_dataset import MultiClaimDataset
from src.datasets.multiclaim.multiclaim_metadata_dataset import MultiClaimMetadataDataset
from src.datasets.webreader.webreader import WebReader
from src.datasets.retrieved_documents import RetrievedDocuments

def dataset_factory(name, **kwargs) -> Dataset:
    """
    Factory function to create a dataset object.
    
    Args:
        name (str): The name of the dataset.
        
    Returns:
        Dataset: An instance of the dataset.
    """
    dataset = {
        'multiclaim': MultiClaimDataset,
        'multiclaim_metdata': MultiClaimMetadataDataset,
        'web': WebReader,
        'retrieved_documents': RetrievedDocuments,
    }[name](**kwargs)
    
    return dataset