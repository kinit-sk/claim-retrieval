from src.postprocess.postprocessor import Postprocessor
from src.postprocess.retriever_postprocessor import RetrieverPostprocessor

def postprocessor_factory(name: str, **kwargs) -> Postprocessor:
    """
    Factory function to create a postprocessor object.
    
    Args:
        name (str): The name of the postprocessor.
        
    Returns:
        Postprocessor: An instance of the postprocessor.
    """
    potsprocessor = {
        'retriever_postprocess': RetrieverPostprocessor,
    }[name]
    return potsprocessor(**kwargs)