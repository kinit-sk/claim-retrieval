import pandas as pd
from tqdm import tqdm

import sys
sys.path.append("rag")

from src.retrievers.vectorizers.sentence_transformer_vectorizer import SentenceTransformerVectorizer


def main():
    vct = SentenceTransformerVectorizer(
        dir_path='./cache/me5-large',
        model_handle='intfloat/multilingual-e5-large'
    )

    df = pd.read_csv('./baseline.csv')
    df['similarity'] = None
    similarities = []

    vct.vectorize(
        df['post_text'].tolist(),
        save_if_missing=True,
        normalize=True
    )

    vct.vectorize(
        df['fact_check_text'].tolist(),
        save_if_missing=True,
        normalize=True
    )

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        post = row['post_text']
        claim = row['fact_check_text']
        
        post_embedding = vct.vectorize(
            [post],
            save_if_missing=True,
            normalize=True
        )
        
        claim_embedding = vct.vectorize(
            [claim],
            save_if_missing=True,
            normalize=True
        )
        
        similarity = post_embedding @ claim_embedding.T
        similarities.append(similarity.item())
        
    df['similarity'] = similarities

    df.to_csv('./me5-large.csv', index=False)
    

if __name__ == '__main__':
    main()