import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("rag")
from src.datasets import dataset_factory
from src.models import HFModel


df = pd.read_csv('./datasets/multiclaim/sampled_posts.csv')

multiclaim = dataset_factory(
    name='multiclaim',
    crosslingual=False,
    fact_check_language=None,
    language=None,
    post_language=None,
    split=None,
    version='original',
).load()

model = HFModel(
    name='meta-llama/Llama-3.3-70B-Instruct',
    # name="mistralai/Mistral-Large-Instruct-2407",
    max_new_tokens=1024,
    device_map='auto',
    load_in_4bit=True,
    offload_folder=None,
    max_memory=None,
    system_prompt="You are a professional fact-checker tasked to determine the veracity of the given claim. Your answer should be one of the following: True, False, or Unverifiable. If you are not sure, please say 'Unverifiable'.",
)

df['post_text'] = ''
df['prediction'] = ''
df.reset_index(drop=True, inplace=True)

for index, row in tqdm(df.iterrows(), total=len(df)):
    post_id = row['post_id']
    post = multiclaim.id_to_post[post_id]
    
    df.at[index, 'post_text'] = post
    
    prompt = f"""Given social media post, your task is to determine the veracity of the post using your own knowledge. After prediction, you should provide a short explanation of your reasoning.

Your answer should be formatted as follows:
Answer: [True/False/Unverifiable]
Explanation: [Your explanation here]

Post: {post}
Answer:"""

    result = model(prompt=prompt, max_new_tokens=1024)
    output = result['output']
    
    df.at[index, 'prediction'] = output
    
df.to_csv('./results/final/pipeline/sampled_posts-llama3_3-70b.csv', index=False)
