import os
import yaml
import logging
import json

from src.evaluation.evaluate_faiss import evaluate_post_fact_check_pairs, evaluate_multiclaim, process_results
from src.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_experiment(config, language: str = None, output_dir: str = None):

    pipeline = Pipeline(rag_config=config)
    dataset = pipeline.modules[0].dataset
    
    os.makedirs(f'./results/{output_dir}', exist_ok=True)
    csv_path = f'./results/{output_dir}/{language}.csv' if output_dir is not None else None 

    if 'model_name' in config["steps"][0]["retriever"]:
        logger.info(f'Evaluating model "{config["steps"][0]["retriever"]["model_name"]}" for language "{language}"')
    else:
        logger.info(f'Evaluating model "{config["steps"][0]["retriever"]["name"]}" for language "{language}"')
    
    generator = evaluate_post_fact_check_pairs(
        evaluate_multiclaim(
            dataset, 
            pipeline, 
            csv_path=csv_path
        ),
        dataset
    )

    output_path = f'./results/{output_dir}/{language}-results.csv'

    results = process_results(
            generator, default_rank=1000, csv_path=output_path)
    
    results["language"] = language
    results["number_of_pairs"] = len(dataset.fact_check_post_mapping)
    results["number_of_posts"] = len(dataset.id_to_post)
    results["number_of_fact_checks"] = len(dataset.id_to_documents)
    if 'model_name' in config["steps"][0]["retriever"]:
        results["model"] = config["steps"][0]["retriever"]["model_name"]
    else:
        results["model"] = config["steps"][0]["retriever"]["name"]

    print(results)

    metrics_path = f'./results/{output_dir}/{language}-metrics.json'
    with open(metrics_path, "w") as metrics_json:
        json.dump(results, metrics_json)


if __name__ == '__main__':
    
    models = [
        'bm25',
        'all-distilroberta-v1',
        'all-MiniLM-L6-v2',
        'all-MiniLM-L12-v2',
        'all-mpnet-base-v2',
        'bge-m3',
        'distiluse-base-multilingual-cased-v2',
        'gte-large-en-v1.5',
        'gtr-t5-large',
        'LaBSE',
        'multilingual-e5-small',
        'multilingual-e5-base',
        'multilingual-e5-large',
        'paraphrase-multilingual-MiniLM-L12-v2',
        'paraphrase-multilingual-mpnet-base-v2',
    ]

    languages = ['spa', 'eng', 'por', 'fra', 'msa', 'deu', 'ara', 'tha', 'hbs', 'kor', 'pol', 'slk', 'nld', 'ron', 'ell', 'ces', 'bul', 'hun', 'hin', 'mya']
    
    for model in models:
        for language in languages:
            with open(f'./configs/tems/{model}.yaml', 'r') as file:
                config = yaml.safe_load(file)

            output_dir = f'{model}'

            config['steps'][0]['retriever']['dataset']['language'] = language
            run_experiment(config, language, output_dir=output_dir)

        # All
        config['steps'][0]['retriever']['dataset']['language'] = None
        run_experiment(config, 'all', output_dir=output_dir)
