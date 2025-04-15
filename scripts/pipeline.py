import pandas as pd
from tqdm import tqdm
import argparse
import re
import time
import os

import sys
sys.path.append("rag")

from src.evaluation.utils import get_evaluation_subset
from src.retrievers import retriever_factory
from src.datasets import dataset_factory
from src.prompts import prompt_factory
from src.models import model_factory
from src.config import RAGConfig

def get_args():
    parser = argparse.ArgumentParser(description='Arguments for the script')
    parser.add_argument(
        '--config', type=str, default='./configs/pipeline.yaml', help='Model name')
    parser.add_argument('--language', type=str, default=None, help='Language')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    language = args.language
    
    config = RAGConfig.load_config(args.config)
        
    retriever_config = config.steps[0].__dict__
    knowledgebase_config = config.steps[1].__dict__
    prompt_relevant_config = config.steps[2].__dict__
    prompt_summary_config = config.steps[3].__dict__
    model_config = config.steps[4].__dict__
    prompt_overall_config = config.steps[5].__dict__
    prompt_explanation_config = config.steps[6].__dict__
    
    retriever_knowledgebase = dataset_factory(**retriever_config['dataset'].__dict__).load()
    del retriever_config['dataset']
    
    retriever = retriever_factory(**retriever_config, dataset=retriever_knowledgebase)
    web = dataset_factory(**knowledgebase_config)
    
    prompt_name = prompt_relevant_config['type']
    del prompt_relevant_config['type']
    prompt_relevant = prompt_factory(name=prompt_name, **prompt_relevant_config)
    
    prompt_name = prompt_summary_config['type']
    del prompt_summary_config['type']
    prompt_summary = prompt_factory(name=prompt_name, **prompt_summary_config)
    
    model_name = model_config['model_name']
    del model_config['model_name']
    
    model = model_factory(**model_config, name=model_name)
    model.set_system_prompt(model_config.get('system_prompt', None))
    
    prompt_name = prompt_overall_config['type']
    del prompt_overall_config['type']
    prompt_overall = prompt_factory(name=prompt_name, **prompt_overall_config)
    
    prompt_name = prompt_explanation_config['type']
    del prompt_explanation_config['type']
    prompt_explanation = prompt_factory(name=prompt_name, **prompt_explanation_config)
        
    dataset = dataset_factory('multiclaim', language=language).load()
    dataset_en = dataset_factory('multiclaim', language=language, version='english').load()
    id_to_post = get_evaluation_subset(dataset, language=language, previous_ids=None)
    
    os.makedirs(f'./results/final/pipeline', exist_ok=True)
    csv_path = f'./results/final/pipeline/{args.output_dir}.csv'
    df = pd.DataFrame(columns=['post_id', 'fact_check_ids', 'post', 'fact_check_claims', 'articles', 'ratings', 'summaries', 'relevant_prompt', 'relevant_claims', 'relevant_claims_ids', 'explanation_prompt', 'explanation'])
    
    for post_id, post in tqdm(list(id_to_post.items())):
        fact_check_claims, fc_ids = retriever.retrieve(query=post)
        results = web(documents=fact_check_claims, top_k=fc_ids, query=post)
        
        fact_check_articles = results['documents']
        ratings_en = results['ratings']
        
        retrieved_items = [
            f'Claim ID: {idx + 1}\nFact-checked claim: {dataset_en.get_document(fc_id)}'
            for idx, fc_id in enumerate(fc_ids)
        ]
        retrieved_part = ("\n\n").join(retrieved_items)
        
        model.set_system_prompt('"You are a fact-checker tasked to determine the veracity of the given claim. Your answer should by only in English."')
        p_relevant = prompt_relevant(post_text=post, retrieved_part=retrieved_part)
        prompts = p_relevant['prompt'].replace('\\n','\n')
        results = model(prompt=prompts, max_new_tokens=1024)
        relevant_output = results['output']
        
        relevant_claims_ids = re.findall(r'"claim_id": (.*),', relevant_output)
        
        try:
            relevant_claims_ids = [fc_ids[int(claim_id.replace('"', '').replace(',','')) - 1] for claim_id in relevant_claims_ids]
            relevant_claims_ids = list(set(relevant_claims_ids))
        except:
            df = pd.concat([df, pd.DataFrame({
                'post_id': [post_id],
                'post': [post],
                'fact_check_ids': [fc_ids],
                'fact_check_claims': [fact_check_claims],
                'ratings': [ratings_en],
                'relevant_prompt': [p_relevant['prompt'].replace('\\n','\n')],
                'relevant_claims': [relevant_output],
                'relevant_claims_ids': [""],
            })])
            df.to_csv(csv_path, index=False)
            continue
        
        if len(relevant_claims_ids) == 0:
            df = pd.concat([df, pd.DataFrame({
                'post_id': [post_id],
                'post': [post],
                'fact_check_ids': [fc_ids],
                'fact_check_claims': [fact_check_claims],
                'ratings': [ratings_en],
                'relevant_prompt': [p_relevant['prompt'].replace('\\n','\n')],
                'relevant_claims': [relevant_output],
                'relevant_claims_ids': [relevant_claims_ids],
            })])
            df.to_csv(csv_path, index=False)
            continue
            
                
        model.set_system_prompt(model_config.get('system_prompt', None))
        fact_check_articles_relevant = [fact_check_articles[fc_ids.index(fc_id)] for fc_id in relevant_claims_ids]
        p = prompt_summary(documents=fact_check_articles_relevant, query=post)
        prompts = [pr.replace('\\n','\n') for pr in p['prompt']]
        results = model(prompt=prompts, max_new_tokens=1024)
        summaries = results['output'].split('#####')
        
        model.set_system_prompt('"You are a fact-checker tasked to determine the veracity of the given claim. Your answer should by only in English."')
        sums = [
            f'Retrieved claim: {dataset_en.get_document(fc_id)}\nSummary: {summary}'
            for fc_id, summary in zip(relevant_claims_ids, summaries)
        ]
        
        # p_overall = prompt_overall(summaries=("\n\n").join(sums))
        # prompts = p_overall['prompt'].replace('\\n','\n')
        # results = model(prompt=prompts, max_new_tokens=1024)
        # overall_summary = results['output']
        
        ratings_en_relevant = [ratings_en[fc_ids.index(fc_id)] for fc_id in relevant_claims_ids]
        retrieved_items = [
            f'Fact-checked claim: {dataset_en.get_document(fc_id)}\nSummary of the fact-check article: {summary}\nRating of the fact-checked claim: {rating}'
            for fc_id, summary, rating in zip(relevant_claims_ids, summaries, ratings_en_relevant)
        ]
        retrieved_part = ("\n\n").join(retrieved_items)
        p_explanation = prompt_explanation(post_text=post, retrieved_part=retrieved_part)
        prompts = p_explanation['prompt'].replace('\\n','\n')
        results = model(prompt=p_explanation['prompt'], max_new_tokens=1024)
        explanation = results['output']
        
        df = pd.concat([df, pd.DataFrame({
            'post_id': [post_id],
            'post': [post],
            'fact_check_ids': [fc_ids],
            'fact_check_claims': [fact_check_claims],
            'summaries': [summaries],
            'articles': [fact_check_articles],
            'ratings': [ratings_en],
            'relevant_prompt': [p_relevant['prompt'].replace('\\n','\n')],
            'relevant_claims': [relevant_output],
            'relevant_claims_ids': [relevant_claims_ids],
            # 'overal_summary': [overall_summary],
            'explanation': [explanation],
            # 'overal_summary_prompt': [p_overall['prompt'].replace('\\n','\n')],
            'explanation_prompt': [p_explanation['prompt'].replace('\\n','\n')],
        })])
        
        df.to_csv(csv_path, index=False)
