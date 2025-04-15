from typing import List, Union
from src.models.model import Model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HFModel(Model):
    """
    A class to represent a Hugging Face model.
    
    Attributes:
        name (str): The name of the model.
        max_new_tokens (int): The maximum number of tokens to generate.
        do_sample (bool): Whether to use sampling.
        device_map (str): The device map.
        load_in_4bit (bool): Whether to load the model in 4-bit.
        load_in_8bit (bool): Whether to load the model in 8-bit.
        offload_folder (str): The offload folder.
        offload_state_dict (bool): Whether to offload the state dict.
        max_memory (Any): The maximum memory to use.
        system_prompt (str): The system prompt to use.    
    """
    def __init__(
        self, 
        name: str = 'google/gemma-7b-it',
        max_new_tokens: int = 128, 
        do_sample: bool = False, 
        device_map: str = 'auto', 
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        **kwargs
    ):
        super().__init__(name='HFModel', max_new_tokens=max_new_tokens)
        self.model_name = name
        self.tokenizer = None
        self.model = None
        self.device_map = device_map
        self.do_sample = do_sample
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.offload_folder = kwargs.get('offload_folder', None)
        self.offload_state_dict = kwargs.get('offload_folder', None)
        self.max_memory = kwargs.get('max_memory', None)
        self.system_prompt = kwargs['system_prompt']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load()
        
    def _load_model(self) -> None:
        """
        Load the model.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map = self.device_map,
            offload_folder = self.offload_folder,
            offload_state_dict = self.offload_state_dict,
            max_memory = self.max_memory
        )
        
    def load_quantized_model(self) -> None:
        """
        Load the quantized model.
        """
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            load_in_8bit=self.load_in_8bit,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map=self.device_map, 
            quantization_config=quantization_config
        )

    def load(self) -> 'HFModel':
        """
        Load the Hugging Face model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.load_in_8bit or self.load_in_4bit:
            self.load_quantized_model()
        else:
            self._load_model()

        logging.log(
            logging.INFO, f'Loaded model and tokenizer from {self.model_name}')
        
    def _is_chat(self) -> bool:
        """
        Check if the model is a chat model.
        """
        return hasattr(self.tokenizer, 'chat_template')
    
    def _get_system_role(self) -> str:
        """
        Get the system role.
        """
        if 'gemma' in self.model_name:
            return None
        else:
            return 'system'
    
    def _terminators(self) -> List[int]:
        """
        Get the terminators.
        """
        if 'Llama-3' in self.model_name:
            return [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        else:
            return [
                self.tokenizer.eos_token_id
            ]

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
            if self._is_chat():
                if self.system_prompt and self._get_system_role() == 'system':
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": p}
                    ]
                elif self.system_prompt:
                    messages = [
                        {"role": "user", "content": f'{self.system_prompt}\n\n{p}'}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": p}
                    ]
                
                inputs = self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True,
                    return_tensors='pt'
                ).to(self.device)

            else:
                if self.system_prompt is not None:
                    p = f'{self.system_prompt}\n\n{p}'

                inputs = self.tokenizer(
                    p, 
                    return_tensors='pt'
                ).to(self.device)['input_ids']
            
            generated_ids = self.model.generate(
                input_ids=inputs,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self._terminators(),
                do_sample=self.do_sample,
            )
            
            decoded_input = self.tokenizer.batch_decode(
                inputs, 
                skip_special_tokens=True
            )[0]
            
            decoded = self.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            decoded = decoded[len(decoded_input):]
            answers.append(decoded.strip())
        
        return '#####'.join(answers)
