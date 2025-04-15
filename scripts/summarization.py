import logging
import pandas as pd
from tqdm import tqdm
import argparse
from datasets import load_dataset
import os
import time
# This is necessary to use the code from the rag sub-module
import sys
sys.path.append("rag")

from src.models.hf_model import HFModel
from src.models.openai import OpenAIModel
from src.models.anthropic import AnthropicModel
from src.prompts.prompt import Prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./datasets/afp-sum.csv')
    parser.add_argument('--output', type=str, default='./results/summarization/')
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--load_in_4bit', action='store_true', default=False, help='Load the model in 4-bit.')
    parser.add_argument('--load_in_8bit', action='store_true', default=False, help='Load the model in 8-bit.')
    
    args = parser.parse_args()
    
    # check if path exists
    if os.path.exists(args.dataset):
        df = pd.read_csv(args.dataset)
        
        texts = df['processed_text'].tolist()
        summaries = df['summary'].tolist()
    else:
        dataset = load_dataset(args.dataset)
        texts = dataset['validation']['text']
        summaries = dataset['validation']['sum']
        
    prompts = {
        'article-last': 'Create a 3-5 sentence summary of the following article, focusing on the main idea. Provide only the summary in English without any additional text.\nArticle:{text}\nSummary:',
        'article-first': 'Article:{text}\n\nCreate a 3-5 sentence summary of the article, focusing on the main idea. Provide only the summary in English without any additional text.\nSummary:'
    }
    
    model_name = args.model
    if 'claude' in model_name:
        API_KEY = os.environ.get('ANTHROPIC_API_KEY')
        model = AnthropicModel(
            name=model_name,
            api_key=API_KEY,
            max_new_tokens=1024,
            system_prompt="Provide the summary in English.",
        )
    elif 'gpt-4o' in model_name:
        API_KEY = os.environ.get('OPENAI_API_KEY')
        AZURE_ENDPOINT = os.environ.get('OPENAI_AZURE_ENDPOINT')
        model = OpenAIModel(
            name=model_name,
            api_key=API_KEY,
            max_new_tokens=1024,
            system_prompt="Provide the summary in English.",
            azure_endpoint=AZURE_ENDPOINT
        )
    else:
        model = HFModel(
            name=model_name,
            max_new_tokens=1024,
            do_sample=False,
            device_map='auto',
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            system_prompt="Provide the summary in English."
        )
    
    dataset = args.dataset.split('/')[-1].split('.')[0]
    model_name = model_name.split('/')[-1]
    
    for key, value in prompts.items():
        prompt = Prompt(
            template = value,
        )
        
        df_generated = pd.DataFrame()
        generated_summaries = []
        prompts = []
        
        index = 0
        for text, summary in tqdm(zip(texts, summaries), total=len(texts)):
            created_prompts = prompt(text=text)
            try:
                model_output = model(prompt=created_prompts['prompt'], max_new_tokens=1024)
                prompts.append(created_prompts['prompt'])
                generated_summaries.append(model_output['output'])
            except Exception as e:
                print(e)
                model_output = {
                    'output': None
                }
            
            df_generated = pd.concat([df_generated, pd.DataFrame({
                'text': [text],
                'prompt': [created_prompts['prompt']],
                'summary': [summary],
                'generated_summary': [model_output['output']],
            })])
            
            index += 1
            if index % 50 == 0:
                df_generated.to_csv(f'{args.output}/{dataset}_{model_name}_{key}.csv', index=False)
            if 'claude' in model_name or 'gpt-4o' in model_name:
                time.sleep(25)
            
        df_generated.to_csv(f'{args.output}/{dataset}_{model_name}_{key}.csv', index=False)
