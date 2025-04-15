from typing import List, Union
from src.models.model import Model
import anthropic

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnthropicModel(Model):
    """
    A class to represent an Anthropic model.
    
    Attributes:
        name (str): The name of the model.
        max_new_tokens (int): The maximum number of tokens to generate.
        system_prompt (str): The system prompt to use.
    """
    def __init__(self, name: str = "claude-3-5-sonnet-20240620", max_new_tokens: int = 128, api_key='lm-studio', **kwargs):
        super().__init__(name='AnthropicModel', max_new_tokens=max_new_tokens)
        self.model_name = name
        self.client = None
        self.system_prompt = kwargs['system_prompt']
        self.api_key = api_key
        self.load()

    def load(self) -> 'AnthropicModel':
        """
        Load the OpenAI model.
        """
        self.client = anthropic.Anthropic(api_key=self.api_key)

        logging.log(
            logging.INFO, f'Created Anthropic client with {self.model_name}')

    def infere(self, prompt: Union[str, List[str]], max_new_tokens: int = None) -> str:
        """
        Generate text based on the prompt.
        
        Args:
            prompt (Union[str, List[str]]): The prompt to generate text from.
            
        Returns:
            str: The generated text.
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        if isinstance(prompt, str):
            prompt = [prompt]
        
        answers = []
        for p in prompt:
            if self.system_prompt:
                completion = self.client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": p}],
                    temperature=0.0,
                    max_tokens=self.max_new_tokens,
                )
            else:
                completion = self.client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": p}],
                    temperature=0.0,
                    max_tokens=self.max_new_tokens,
                )
            
            output = completion.content[0].text
            answers.append(output.strip())
        
        return '#####'.join(answers)
