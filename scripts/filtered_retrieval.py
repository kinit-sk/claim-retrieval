import os
import yaml
import logging
import json
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from src.evaluation.evaluate_faiss import evaluate_post_fact_check_pairs, evaluate_multiclaim_retrieval, process_results
from src.pipeline import Pipeline
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_experiment(config, output_dir: str = None, prompt_template: str = None, experiment_type: str = None, identifier: str = None):
    pipeline = Pipeline(rag_config=config)
    dataset = pipeline.modules[0].dataset
    
    os.makedirs(f'./results/{output_dir}/embedding-retrieval-{experiment_type}-0_8-sims', exist_ok=True)
    csv_path = f'./results/{output_dir}/embedding-retrieval-{experiment_type}-0_8-sims/{identifier}.csv' if output_dir is not None else None 

    if 'model_name' in config["steps"][0]["retriever"]:
        logger.info(f'Evaluating model "{config["steps"][0]["retriever"]["model_name"]}"')
    else:
        logger.info(f'Evaluating model "{config["steps"][0]["retriever"]["name"]}"')
    
    generator = evaluate_post_fact_check_pairs(
        evaluate_multiclaim_retrieval(
            dataset, 
            pipeline,
            prompt_template=prompt_template,
            csv_path=csv_path
        ),
        dataset
    )

    output_path = f'./results/{output_dir}/embedding-retrieval-{experiment_type}-0_8-sims/{identifier}-results.csv'

    results = process_results(
            generator, default_rank=1000, csv_path=output_path)
    
    results["number_of_pairs"] = len(dataset.fact_check_post_mapping)
    results["number_of_posts"] = len(dataset.id_to_post)
    results["number_of_fact_checks"] = len(dataset.id_to_documents)
    results["top_k"] = config["steps"][0]["retriever"]["top_k"]
    if 'model_name' in config["steps"][0]["retriever"]:
        results["model"] = config["steps"][0]["retriever"]["model_name"]
    else:
        results["model"] = config["steps"][0]["retriever"]["name"]

    print(results)

    metrics_path = f'./results/{output_dir}/embedding-retrieval-{experiment_type}-0_8-sims/{identifier}-metrics.json'
    with open(metrics_path, "w") as metrics_json:
        json.dump(results, metrics_json)


if __name__ == '__main__':
    df_fcs = pd.read_csv('./datasets/multiclaim/fact_checks_preprocessed.csv')
    
    models = ['multilingual-e5-large-2step']
    
    experiments = ['language', 'domain', 'date', 'entity']
    
    prompt_templates = {
        'language': 'Retrieve the results only in <<language>>.',
        'domain': 'Retrieve the results only from the <<domain>>.',
        'date': 'Retrieve the results published between <<start_date>> and <<end_date>>.',
        'entity': 'Retrieve the results that contain <<named_entity>>.'
    }
    
    for experiment in experiments:
        if experiment == 'language':
            languages = ['Spanish', 'English', 'Portuguese', 'French', 'Malay', 'German', 'Arabic', 'Thai', 'Serbo-Croatian', 'Korean', 'Polish', 'Slovak', 'Dutch', 'Romanian', 'Greek', 'Czech', 'Bulgarian', 'Hungarian', 'Hindi', 'Burmese']
            
            for model in models:
                for language in languages:
                    with open(f'./configs/{model}.yaml', 'r') as file:
                        config = yaml.safe_load(file)
                    prompt_template = prompt_templates[experiment]
                    prompt_template = prompt_template.replace('<<language>>', language)
                        
                    output_dir = f'{model}'
                    run_experiment(config, output_dir=output_dir, prompt_template=prompt_template, identifier=language, experiment_type=experiment)
        elif experiment == 'domain':
            domains = list(df_fcs['domain'].unique())
            existing_paths = glob.glob('./results/multilingual-e5-large-2step/embedding-retrieval-domain-0_8-sims/*.csv')
            existing_domains = [path.split('/')[-1].replace('.csv', '') for path in existing_paths if not path.endswith('-results.csv')]
            
            domains = [domain for domain in domains if domain not in existing_domains]
            print(len(domains))
            
            for model in models:
                for domain in domains:
                    if domain == 'scroll.in':
                        continue
                    with open(f'./configs/{model}.yaml', 'r') as file:
                        config = yaml.safe_load(file)
                    prompt_template = prompt_templates[experiment]
                    prompt_template = prompt_template.replace('<<domain>>', domain)
                        
                    output_dir = f'{model}'
                    run_experiment(config, output_dir=output_dir, prompt_template=prompt_template, identifier=domain, experiment_type=experiment)
        elif experiment == 'date':
            df_fcs['published_at'] = pd.to_datetime(df_fcs['published_at'])
            df_fcs['year'] = df_fcs['published_at'].apply(lambda x: x.year)
            df_fcs['month'] = df_fcs['published_at'].apply(lambda x: x.month)
            tuples = df_fcs.groupby(['year', 'month']).size()[df_fcs.groupby(['year', 'month']).size() > 100].index.tolist()
    
            date_ranges = []
            for year, month in tuples:
                start = datetime(int(year), int(month), 1)
                end = start + relativedelta(months=1, days=-1)
                date_ranges.append((start.strftime('%Y/%m/%d'), end.strftime('%Y/%m/%d')))
                
            for model in models:
                for date_range in date_ranges:
                    with open(f'./configs/{model}.yaml', 'r') as file:
                        config = yaml.safe_load(file)
                    prompt_template = prompt_templates[experiment]
                    prompt_template = prompt_template.replace('<<start_date>>', date_range[0])
                    prompt_template = prompt_template.replace('<<end_date>>', date_range[1])
                    
                    year, month, _ = date_range[0].split('/')
                        
                    output_dir = f'{model}'
                    run_experiment(config, output_dir=output_dir, prompt_template=prompt_template, identifier=f'{year}-{month}', experiment_type=experiment)
        elif experiment == 'entity':
            with open('./datasets/multiclaim/named_entities.txt', 'r') as file:
                entities = file.readlines()
                entities = [entity.strip() for entity in entities]
                
            print(len(entities))
            
            for model in models:
                for entity in entities:
                    with open(f'./configs/{model}.yaml', 'r') as file:
                        config = yaml.safe_load(file)
                    prompt_template = prompt_templates[experiment]
                    prompt_template = prompt_template.replace('<<named_entity>>', entity)
                        
                    output_dir = f'{model}'
                    run_experiment(config, output_dir=output_dir, prompt_template=prompt_template, identifier=entity, experiment_type=experiment)
