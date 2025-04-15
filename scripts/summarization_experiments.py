import os

# TODO: Need to fill in the environment variables
ANTHROPIC_API_KEY = ''
OPENAI_API_KEY = ''
OPENAI_AZURE_ENDPOINT = ''

os.environ['ANTHROPIC_API_KEY'] = ANTHROPIC_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['OPENAI_AZURE_ENDPOINT'] = OPENAI_AZURE_ENDPOINT

models = [
    # (model_name, quantized)
    ('CohereForAI/c4ai-command-r-plus-4bit', None),
    ('meta-llama/Meta-Llama-3.1-70B-Instruct', 'load_in_4bit'),
    ('meta-llama/Llama-3.3-70B-Instruct', 'load_in_4bit'),
    ('mistralai/Mistral-Large-Instruct-2407', 'load_in_4bit'),
    ('Qwen/Qwen2-72B-Instruct', 'load_in_4bit'),
    ('Qwen/Qwen2.5-0.5B-Instruct', None),
    ('Qwen/Qwen2.5-1.5B-Instruct', None),
    ('Qwen/Qwen2.5-3B-Instruct', None),
    ('Qwen/Qwen2.5-7B-Instruct', None),
    ('Qwen/Qwen2.5-72B-Instruct', 'load_in_4bit'),
    ('claude-3-5-sonnet-20240620', None),
    ('gpt-4o', None),
    
    ('meta-llama/Llama-3.2-1B-Instruct', None),
    ('meta-llama/Llama-3.2-1B-Instruct', 'load_in_4bit'),
    ('meta-llama/Llama-3.2-1B-Instruct', 'load_in_8bit'),
    
    ('meta-llama/Llama-3.2-3B-Instruct', None),
    ('meta-llama/Llama-3.2-3B-Instruct', 'load_in_4bit'),
    ('meta-llama/Llama-3.2-3B-Instruct', 'load_in_8bit'),
    
    ('meta-llama/Meta-Llama-3.1-70B-Instruct', None),
]

for model in models:
    
    quantization = '' if model[1] is None else f' --{model[1]}'
    
    os.makedirs(f'./results/summarization/{quantization}', exist_ok=True)
    os.system(f'python -m scripts.summarization --model {model[0]} --dataset ./datasets/sample100.csv --output ./results/summarization/{quantization} {quantization}')
