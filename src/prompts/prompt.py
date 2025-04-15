from typing import Any


class Prompt:
    """
    Prompt class to hold the prompt for the model
    """
    def __init__(self, **kwargs):
        self.prompt = None
        
        if 'template' in kwargs:
            self.from_template(kwargs['template'])

    def get_prompt(self, **kwargs) -> dict:
        return {
            'prompt': self.format(**kwargs)
        }
    
    def from_template(self, template: str, **kwargs):
        self.template = template
        
    def format(self, **kwargs) -> str:
        """
        Format the prompt using the saved prompt template.
        """
        return self.template.format(**kwargs)

    def __call__(self, **kwargs) -> Any:
        return self.get_prompt(
            **kwargs
        )

        
class PointwisePrompt(Prompt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_prompt(self, **kwargs) -> dict:
        documents = kwargs.get('documents', [])
        query = kwargs.get('query', '')

        return {
            'prompt': [
                self.template.format(document=document, query=query)
                for document in documents
            ]
        }


class UniversalPrompt(Prompt):
    def __init__(self, **kwargs):
        pass

    def get_prompt(self, **kwargs) -> dict:
        documents = kwargs.get('documents', [])
        post = kwargs.get('query', '')
        
        docs = ' '.join(
            [f'{doc}' for doc in documents])

        return {
            'prompt': f'{docs}\n\n{post}'  # nopep8
        }