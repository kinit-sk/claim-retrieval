import pandas as pd
import os


def get_evaluation_subset(dataset, language=None, previous_ids=None):
    subset_path = os.path.join(
        '.', 'datasets', 'multiclaim', 'sampled_posts.csv')

    id_to_post = dataset.id_to_post
    df = pd.read_csv(subset_path)
    if language is not None:
        df = df[df['language'] == language]
    
    id_to_post = {
        id: post for id, post in id_to_post.items() if id in df['post_id'].values
    }
    print(f'Number of {language} posts: {len(id_to_post)}')
    if language and len(id_to_post) > 100:
        id_to_post = dict(list(id_to_post.items())[:100])
    
    if previous_ids is not None:
        id_to_post = {id: post for id, post in id_to_post.items()
                      if id not in previous_ids}

    return id_to_post
