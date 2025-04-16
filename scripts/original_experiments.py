import os
import yaml
import logging
import json
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

from src.evaluation.evaluate_faiss import evaluate_post_fact_check_pairs, evaluate_multiclaim, process_results
from src.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_experiment(config, date_range: tuple = None, domain: str = None, entity: str = None, language: str = None, output_dir: str = None):

    pipeline = Pipeline(rag_config=config)
    dataset = pipeline.modules[0].dataset
    
    if date_range is not None:
        start_date, end_date = date_range
        year, month, _ = start_date.split('/')
    
        os.makedirs(f'./results/{output_dir}/oiginal-date', exist_ok=True)
        csv_path = f'./results/{output_dir}/oiginal-date/{year}-{month}.csv' if output_dir is not None else None
        output_path = f'./results/{output_dir}/oiginal-date/{year}-{month}-results.csv'
    elif domain is not None:
        os.makedirs(f'./results/{output_dir}/original-domain', exist_ok=True)
        csv_path = f'./results/{output_dir}/original-domain/{domain}.csv' if output_dir is not None else None
        output_path = f'./results/{output_dir}/original-domain/{domain}-results.csv'
    elif entity is not None:
        os.makedirs(f'./results/{output_dir}/original-entity', exist_ok=True)
        csv_path = f'./results/{output_dir}/original-entity/{entity}.csv' if output_dir is not None else None
        output_path = f'./results/{output_dir}/original-entity/{entity}-results.csv'
    elif language is not None:
        os.makedirs(f'./results/{output_dir}/original-language', exist_ok=True)
        csv_path = f'./results/{output_dir}/original-language/{language}.csv' if output_dir is not None else None
        output_path = f'./results/{output_dir}/original-language/{language}-results.csv'

        
    generator = evaluate_post_fact_check_pairs(
        evaluate_multiclaim(
            dataset, 
            pipeline, 
            csv_path=csv_path
        ),
        dataset
    )

    results = process_results(
            generator, default_rank=1000, csv_path=output_path, all_posts=True)

if __name__ == '__main__':
    df_fcs = pd.read_csv('./datasets/multiclaim/fact_checks_preprocessed.csv')
    df_fcs['published_at'] = pd.to_datetime(df_fcs['published_at'])
    df_fcs['year'] = df_fcs['published_at'].apply(lambda x: x.year)
    df_fcs['month'] = df_fcs['published_at'].apply(lambda x: x.month)
    
    tuples = df_fcs.groupby(['year', 'month']).size()[df_fcs.groupby(['year', 'month']).size() > 100].index.tolist()
    domains = list(df_fcs['domain'].unique())
    
    date_ranges = []
    for year, month in tuples:
        start = datetime(int(year), int(month), 1)
        end = start + relativedelta(months=1, days=-1)
        date_ranges.append((start.strftime('%Y/%m/%d'), end.strftime('%Y/%m/%d')))
    
    models = [
        'multilingual-e5-large',
    ]
    
    experiments = ['language', 'domain', 'date', 'entity']
    
    
    for experiment in experiments:
        if experiment == 'language':
            languages = ['spa', 'eng', 'por', 'fra', 'msa', 'deu', 'ara', 'tha', 'hbs', 'kor', 'pol', 'slk', 'nld', 'ron', 'ell', 'ces', 'bul', 'hun', 'hin', 'mya']
            
            for model in models:
                for language in languages:
                    with open(f'./configs/{model}.yaml', 'r') as file:
                        config = yaml.safe_load(file)

                    output_dir = f'{model}'

                    # config['steps'][0]['retriever']['dataset']['language'] = language
                    config['steps'][0]['retriever']['dataset']['fact_check_language'] = language
                    run_experiment(config, language=language, output_dir=output_dir)
        elif experiment == 'date':
            for model in models:
                for date_range in date_ranges:
                    with open(f'./configs/{model}.yaml', 'r') as file:
                        config = yaml.safe_load(file)

                    output_dir = f'{model}'

                    config['steps'][0]['retriever']['dataset']['date_range'] = date_range
                    run_experiment(config, date_range=date_range, output_dir=output_dir)
                    
        elif experiment == 'domain':
            for model in models:
                for domain in domains:
                    with open(f'./configs/{model}.yaml', 'r') as file:
                        config = yaml.safe_load(file)

                    output_dir = f'{model}'

                    config['steps'][0]['retriever']['dataset']['domain'] = domain
                    run_experiment(config, domain=domain, output_dir=output_dir)
                    
        elif experiment == 'entity':
            with open('./datasets/multiclaim/named_entities.txt', 'r') as file:
                entities = file.readlines()
                entities = [entity.strip() for entity in entities]
                
            for model in models:
                for entity in entities:
                    with open(f'./configs/{model}.yaml', 'r') as file:
                        config = yaml.safe_load(file)

                    output_dir = f'{model}'

                    config['steps'][0]['retriever']['dataset']['entity'] = entity
                    run_experiment(config, entity=entity, output_dir=output_dir)
