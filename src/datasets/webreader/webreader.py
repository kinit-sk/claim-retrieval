import logging
import os
from os.path import join as join_path
import pandas as pd
from typing import Any

from src.datasets.dataset import Dataset
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebReader(Dataset):
    """
    WebReader class is used to read the content from the given url.
    
    Args:
        url (str): The link to the website.
        
    Returns:
        WebReader: An instance of the WebReader class.
    """
    our_dataset_path = join_path('.','datasets', 'multiclaim')
    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id_to_documents = None
        self.id_to_url = None
        self.load()
    
    def load(self) -> 'WebReader':
        """
        Load the dataset.
        
        Returns:
            WebReader: An instance of the WebReader class.
        """
        if os.path.exists(f'{self.our_dataset_path}/fact_checks_preprocessed.csv'):
            logger.info('Dataset exists.')
            df = pd.read_csv(f'{self.our_dataset_path}/fact_checks_preprocessed.csv')
    
            self.id_to_documents = {
                row['url']: {'rating': row['rating_category']}
                for index, row in df.iterrows()
            }
            self.id_to_url = {
                row['fact_check_id']: row['url'] for index, row in df.iterrows()
            }
            
        return self
    
    def _read_data(self, url: str) -> str:
        logger.info('Reading data.')
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            text = response.text
            
            data = BeautifulSoup(text, 'html.parser')
            text = data.get_text()
            return text
        
        return None
        
    def get_document(self, url: str) -> str:
        """
        Get the article from the url.
        
        Returns:
            str: The article content.
        """
        return self._read_data(url)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        documents = kwds['documents']
        ids = kwds['top_k']
        query = kwds['query']
        
        articles = []
        ratings = []
        for idx in ids:
            url = self.id_to_url[idx]
            article = self.get_document(url)
            rating = self.id_to_documents[url]['rating']
            articles.append(article)
            ratings.append(rating)
        
        self.fact_check_ids = ids
        self.fact_check_claims = documents
        self.fact_check_articles = articles
        self.ratings = ratings
        
        return {
            'documents': articles,
            'top_k': ids,
            'query': query,
            'fact-checks': documents,
            'ratings': ratings
        }
