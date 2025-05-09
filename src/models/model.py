from src.module import Module
from typing import Any

class Model(Module):
    """
    The base class for all models.
    
    Args:
        name (str): The name of the model.
        max_new_tokens (int): The maximum number of new tokens.
    """
    def __init__(self, name: str, max_new_tokens: int, **kwargs):
        super().__init__(name='Model')
        self.max_new_tokens = max_new_tokens
        
    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Set the system prompt.
        
        Args:
            system_prompt (str): The system prompt.
        """
        self.system_prompt = system_prompt

    def infere(self, prompt: str):
        raise NotImplementedError
    
    def convert_to_dict(self, output: str) -> dict:
        return {
            'output': output,
        }
    
    def __call__(self, **kwargs: Any) -> Any:
        output = self.infere(**kwargs)
        return self.convert_to_dict(output)
