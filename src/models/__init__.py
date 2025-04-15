from typing import Any

from src.models.model import Model
from src.models.hf_model import HFModel
from src.models.openai import OpenAIModel
from src.models.anthropic import AnthropicModel


def model_factory(model: str, **kwargs) -> Model:
    """
    Factory function to create a model object based on the model name.
    
    Args:
        model (str): The name of the model.
        
    Returns:
        Model: An instance of the model.
    """
    Model = {
        'hf_model': HFModel,
        'openai': OpenAIModel,
        'anthropic': AnthropicModel
    }[model]
    return Model(**kwargs)
