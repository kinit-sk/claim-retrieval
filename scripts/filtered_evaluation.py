from scipy.stats import spearmanr, kendalltau
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta

def get_ranking(ordered_list):
    """
    Converts an ordered list of document IDs into a dictionary of ranks.
    """
    return {doc_id: rank for rank, doc_id in enumerate(ordered_list, start=1)}

def align_rankings(ground_truth, predicted):
    """
    Aligns rankings by creating a common set of document IDs.
    - Documents not in the predicted list are given the lowest rank (len(predicted) + 1).
    """
    all_docs = list(set(ground_truth) | set(predicted))  # Union of both lists
    gt_ranking = get_ranking(ground_truth)
    pred_ranking = get_ranking(predicted)
    
    # Assign maximum rank + 1000 for missing documents
    max_rank = max(len(ground_truth), len(predicted)) + 1000
    aligned_gt_ranks = [gt_ranking.get(doc, max_rank) for doc in all_docs]
    aligned_pred_ranks = [pred_ranking.get(doc, max_rank) for doc in all_docs]

    return aligned_gt_ranks, aligned_pred_ranks

def spearman_rank_correlation(ground_truth, predicted):
    gt_ranks, pred_ranks = align_rankings(ground_truth, predicted)
    return spearmanr(gt_ranks, pred_ranks).correlation

def kendall_tau(ground_truth, predicted):
    gt_ranks, pred_ranks = align_rankings(ground_truth, predicted)
    return kendalltau(gt_ranks, pred_ranks).correlation

def calculate_corrleation(ground_truth, predicted):
    """
    Calculate the Spearman rank correlation between two lists of document IDs.
    """
    spearman = spearman_rank_correlation(ground_truth, predicted)
    tau = kendall_tau(ground_truth, predicted)
    return spearman, tau

def get_language(ocr, text):
    if text != '':
        return text[2][0][0]
    
    if len(ocr) > 0:
        return ocr[0][2][0][0]
    else:
        return None

df_factcheck = pd.read_csv('./datasets/multiclaim/fact_checks_preprocessed.csv')
df_posts = pd.read_csv('./datasets/multiclaim/posts.csv')
df_posts['ocr'] = df_posts['ocr'].apply(lambda x: eval(str(x)))
df_posts['text'] = df_posts['text'].apply(lambda x: '' if pd.isna(x) else eval(str(x)))
df_posts['language'] = df_posts.apply(lambda x: get_language(x['ocr'], x['text']), axis=1)

## Language evaluation
lang_codes = ['spa', 'eng', 'por', 'fra', 'msa', 'deu', 'ara', 'tha', 'hbs', 'kor', 'pol', 'slk', 'nld', 'ron', 'ell', 'ces', 'bul', 'hun', 'hin', 'mya']
languages = ['Spanish', 'English', 'Portuguese', 'French', 'Malay', 'German', 'Arabic', 'Thai', 'Serbo-Croatian', 'Korean', 'Polish', 'Slovak', 'Dutch', 'Romanian', 'Greek', 'Czech', 'Bulgarian', 'Hungarian', 'Hindi', 'Burmese']

for code, language in tqdm(zip(lang_codes, languages), total=len(lang_codes)):
    df_true = pd.read_csv(f'./results/multilingual-e5-large/original-language/{code}.csv')
    df_true['fact_check_ids'] = df_true['fact_check_ids'].apply(eval)
    df_predicted = pd.read_csv(f'./results/multilingual-e5-large-2step/embedding-retrieval-language-0_8-sims/{language}.csv')
    df_predicted['fact_check_ids'] = df_predicted['fact_check_ids'].apply(eval)
    
    df_predicted['spearman'] = ''
    df_predicted['kendalltau'] = ''
    df_predicted['post_language'] = ''
    df_predicted['common_fcs'] = ''

    for index, row in df_predicted.iterrows():
        true = df_true[df_true['post_id'] == row['post_id']]['fact_check_ids'].values[0]
        pred = row['fact_check_ids']
        true_fcs = set(true)
        pred_fcs = set(pred)
        common_fcs = true_fcs.intersection(pred_fcs)
        common_fcs = len(common_fcs)
        
        df_predicted.at[index, 'common_fcs'] = common_fcs / len(true_fcs)
        
        spearman, kendall = calculate_corrleation(true, pred)
        df_predicted.at[index, 'spearman'] = spearman
        df_predicted.at[index, 'kendalltau'] = kendall
        
        post_lang = df_posts[df_posts['post_id'] == row['post_id']]['language'].values[0]
        df_predicted.at[index, 'post_language'] = post_lang
        
    df_predicted.to_csv(f'./results/multilingual-e5-large-2step/embedding-retrieval-language-0_8-sims/{language}.csv', index=False)

## Domain evaluation
domains = list(df_factcheck['domain'].unique())
is_done = True

for domain in tqdm(domains, total=len(domains)):
    if domain == "scroll.in":
        is_done = False

    if is_done or domain == "scroll.in":
        continue
    df_true = pd.read_csv(f'./results/multilingual-e5-large/original-domain/{domain}.csv')
    df_true['fact_check_ids'] = df_true['fact_check_ids'].apply(eval)
    df_predicted = pd.read_csv(f'./results/multilingual-e5-large-2step/embedding-retrieval-domain-0_8-sims/{domain}.csv')
    df_predicted['fact_check_ids'] = df_predicted['fact_check_ids'].apply(eval)
    
    df_predicted['spearman'] = ''
    df_predicted['kendalltau'] = ''
    df_predicted['post_language'] = ''
    df_predicted['common_fcs'] = ''

    for index, row in df_predicted.iterrows():
        true = df_true[df_true['post_id'] == row['post_id']]['fact_check_ids'].values[0]
        pred = row['fact_check_ids']
        true_fcs = set(true)
        pred_fcs = set(pred)
        common_fcs = true_fcs.intersection(pred_fcs)
        common_fcs = len(common_fcs)
        
        df_predicted.at[index, 'common_fcs'] = common_fcs / len(true_fcs)
        
        spearman, kendall = calculate_corrleation(true, pred)
        df_predicted.at[index, 'spearman'] = spearman
        df_predicted.at[index, 'kendalltau'] = kendall
        
        post_lang = df_posts[df_posts['post_id'] == row['post_id']]['language'].values[0]
        df_predicted.at[index, 'post_language'] = post_lang
        
    df_predicted.to_csv(f'./results/multilingual-e5-large-2step/embedding-retrieval-domain-0_8-sims/{domain}.csv', index=False)


## Date evaluation
df_factcheck['published_at'] = pd.to_datetime(df_factcheck['published_at'])
df_factcheck['year'] = df_factcheck['published_at'].apply(lambda x: x.year)
df_factcheck['month'] = df_factcheck['published_at'].apply(lambda x: x.month)
tuples = df_factcheck.groupby(['year', 'month']).size()[df_factcheck.groupby(['year', 'month']).size() > 100].index.tolist()

date_ranges = []
for year, month in tuples:
    start = datetime(int(year), int(month), 1)
    end = start + relativedelta(months=1, days=-1)
    date_ranges.append((start.strftime('%Y/%m/%d'), end.strftime('%Y/%m/%d')))

for date_range in tqdm(date_ranges, total=len(date_ranges)):
    date = date_range[0][:7].replace('/', '-')
    df_true = pd.read_csv(f'./results/multilingual-e5-large/original-date/{date}.csv')
    df_true['fact_check_ids'] = df_true['fact_check_ids'].apply(eval)
    df_predicted = pd.read_csv(f'./results/multilingual-e5-large-2step/embedding-retrieval-date-0_8-sims/{date}.csv')
    
    df_predicted['fact_check_ids'] = df_predicted['fact_check_ids'].apply(eval)
    
    df_predicted['spearman'] = ''
    df_predicted['kendalltau'] = ''
    df_predicted['post_language'] = ''
    df_predicted['common_fcs'] = ''

    for index, row in df_predicted.iterrows():
        true = df_true[df_true['post_id'] == row['post_id']]['fact_check_ids'].values[0]
        pred = row['fact_check_ids']
        true_fcs = set(true)
        pred_fcs = set(pred)
        common_fcs = true_fcs.intersection(pred_fcs)
        common_fcs = len(common_fcs)
        
        df_predicted.at[index, 'common_fcs'] = common_fcs / len(true_fcs)
        
        spearman, kendall = calculate_corrleation(true, pred)
        df_predicted.at[index, 'spearman'] = spearman
        df_predicted.at[index, 'kendalltau'] = kendall
        
        post_lang = df_posts[df_posts['post_id'] == row['post_id']]['language'].values[0]
        df_predicted.at[index, 'post_language'] = post_lang
        
    df_predicted.to_csv(f'./results/multilingual-e5-large-2step/embedding-retrieval-date-0_8-sims/{date}.csv', index=False)


## Entity evaluation
with open('./datasets/multiclaim/named_entities.txt', 'r') as file:
    entities = file.readlines()
    entities = [entity.strip() for entity in entities]
    
print(len(entities))

for entity in tqdm(entities, total=len(entities)):
    df_true = pd.read_csv(f'./results/multilingual-e5-large/original-entity/{entity}.csv')
    df_true['fact_check_ids'] = df_true['fact_check_ids'].apply(eval)
    df_predicted = pd.read_csv(f'./results/multilingual-e5-large-2step/embedding-retrieval-entity-0_8-sims/{entity}.csv')
    
    df_predicted['fact_check_ids'] = df_predicted['fact_check_ids'].apply(eval)
    
    df_predicted['spearman'] = ''
    df_predicted['kendalltau'] = ''
    df_predicted['post_language'] = ''
    df_predicted['common_fcs'] = ''

    for index, row in df_predicted.iterrows():
        true = df_true[df_true['post_id'] == row['post_id']]['fact_check_ids'].values[0]
        pred = row['fact_check_ids']
        true_fcs = set(true)
        pred_fcs = set(pred)
        common_fcs = true_fcs.intersection(pred_fcs)
        common_fcs = len(common_fcs)
        
        df_predicted.at[index, 'common_fcs'] = common_fcs / len(true_fcs)
        
        spearman, kendall = calculate_corrleation(true, pred)
        df_predicted.at[index, 'spearman'] = spearman
        df_predicted.at[index, 'kendalltau'] = kendall
        
        post_lang = df_posts[df_posts['post_id'] == row['post_id']]['language'].values[0]
        df_predicted.at[index, 'post_language'] = post_lang
        
    df_predicted.to_csv(f'./results/multilingual-e5-large-2step/embedding-retrieval-entity-0_8-sims/{entity}.csv', index=False)
