from src.module import Module
from src.config import RAGConfig
from typing import Any, Union, List

from src.retrievers import retriever_factory
from src.models import model_factory
from src.prompts import prompt_factory
from src.postprocess import postprocessor_factory
from src.datasets import dataset_factory
from src.datasets.dataset import Dataset
from src.datasets.retrieved_documents import RetrievedDocuments
from src.config import Retriever, Prompt, LLM, Postprocessor, Dataset as DatasetModule
from src.retrievers.retriever import Retriever as RetrieverModule

from copy import deepcopy


class Pipeline(Module):
    """
    Pipeline class to hold the RAG pipeline
    
    Args:
        path: path to the configuration file
        rag_config: configuration of the RAG pipeline
    """
    def __init__(self, path: str = None, rag_config: dict = None):
        super().__init__('Pipeline')
        self.path = path
        self.rag_config = rag_config
        self.modules = []
        self.load()
        
    def get_dataset(self, dataset: Any) -> Dataset:
        dataset = dataset.__dict__
        name = dataset.pop('name')

        if name == 'retrieved_documents':
            return None

        return dataset_factory(
            name, **dataset
        ).load()
        
    def get_module(self, module: Union[Retriever, Prompt, LLM, Postprocessor, DatasetModule]) -> Module:
        """
        Get the module based on the module type
        
        Args:
            module: configuration of the module
            
        Returns:
            Module: the created module
        """
        if isinstance(module, Retriever):
            dataset = self.get_dataset(module.dataset)
            rest = {
                k: v for k, v in module.__dict__.items() if k not in ['name', 'dataset']
            }
            return retriever_factory(module.name, dataset=dataset, **rest)
        elif isinstance(module, Prompt):
            name = module.type
            rest = {
                k: v for k, v in module.__dict__.items() if k not in ['type']
            }
            return prompt_factory(name, **rest)
        elif isinstance(module, LLM):
            rest = {
                k: v for k, v in module.__dict__.items() if k not in ['model_name', 'model']
            }
            return model_factory(module.model, name=module.model_name, **rest)
        elif isinstance(module, Postprocessor):
            rest = {
                k: v for k, v in module.__dict__.items() if k not in ['name']
            }
            return postprocessor_factory(module.name, **rest)
        elif isinstance(module, DatasetModule):
            return self.get_dataset(module)
        else:
            raise ValueError(f'Unknown module type: {type(module)}')

    def _convert_kwargs(self, module: Union[Retriever, Prompt, LLM, Postprocessor], kwargs: dict) -> dict:
        if isinstance(module, RetrieverModule):
            self.retrieved_documents = kwargs['documents']
            self.retrieved_documents_ids = kwargs['top_k']

        return kwargs

    def __call__(self, **kwargs) -> Any:
        for idx, module in enumerate(self.modules):
            if isinstance(module, RetrieverModule) and idx > 0 and isinstance(self.modules[idx - 1], RetrieverModule):
                if len(kwargs['documents']) == 0:
                    continue
                dataset = RetrievedDocuments(
                    name='retrieved_documents',
                    documents=kwargs['documents'],
                    ids=kwargs['top_k']
                )
                kwargs = {
                    "query": kwargs['query']
                }
                module.set_dataset(dataset)

            kwargs = module(**kwargs)
            kwargs = self._convert_kwargs(module, kwargs)

        return kwargs
    
    def _load_modules(self) -> None:
        """
        Load the modules from the configuration
        """
        steps = deepcopy(self.config.steps)

        self.modules = []
        for step in steps:
            module = self.get_module(step)
            self.modules.append(module)
            
    def set_modules(self, modules: List[Union[Retriever, Prompt, LLM, Postprocessor]]) -> None:
        """
        Set the modules of the pipeline manually.
        
        Args:
            modules: list of modules
        """
        self.modules = modules

    def load(self) -> None:
        if self.rag_config:
            self.config = RAGConfig.from_dict(self.rag_config)
            self._load_modules()
        elif self.path:
            self.config = RAGConfig.load_config(self.path)
            self._load_modules()
