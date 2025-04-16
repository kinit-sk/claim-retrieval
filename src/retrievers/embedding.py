import logging
from typing import Any, List, Generator, Tuple
from nltk.tokenize import sent_tokenize
import torch
from torch_scatter import scatter
import nltk
from tqdm import tqdm

from src.datasets.cleaning import replace_stops, replace_whitespaces
from src.retrievers.retriever import Retriever
from src.retrievers.vectorizers.vectorizer import Vectorizer

nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def slice_text(text, window_type, window_size, window_stride=None) -> List[str]:
    """
    Split a `text` into parts using a sliding window. The windows slides either across characters or sentences, based on the value of `window_tyoe`.

    Attributes:
        text: str  Text that is to be splitted into windows.
        window_type: str  Either `sentence` or `character`. The basic unit of the windows.
        window_size: int  How many units are in a window.
        window_stride: int  How many units are skipped each time the window moves.
    """

    text = replace_whitespaces(text)

    if window_stride is None:
        window_stride = window_size

    if window_size < window_stride:
        logger.warning(
            f'Window size ({window_size}) is smaller than stride length ({window_stride}). This will result in missing chunks of text.')

    if window_type == 'sentence':
        text = replace_stops(text)
        sentences = sent_tokenize(text)
        return [
            ' '.join(sentences[i:i+window_size])
            for i in range(0, len(sentences), window_stride)
        ]

    elif window_type == 'character':
        return [
            text[i:i+window_size]
            for i in range(0, len(text), window_stride)
        ]


def gen_sliding_window_delimiters(post_lengths: List[int], max_size: int) -> Generator[Tuple[int, int], None, None]:
    """
    Calculate where to split the sequence of `post_lenghts` so that the individual batches do not exceed `max_size`
    """
    range_length = start = cur_sum = 0

    for post_length in post_lengths:
        if (range_length + post_length) > max_size:  # exceeds memory
            yield (start, start + range_length)
            start = cur_sum
            range_length = post_length
        else:  # memory still avail in current split
            range_length += post_length
        cur_sum += post_length

    if range_length > 0:
        yield (start, start + range_length)


class Embedding(Retriever):
    def __init__(
            self,
            name: str = 'embedding',
            top_k: int = 5,
            vectorizer_document: Vectorizer = None,
            vectorizer_query: Vectorizer = None,
            sliding_window: bool = False,
            sliding_window_pooling: str = 'max',
            sliding_window_size: int = None,
            sliding_window_stride: int = None,
            sliding_window_type: str = None,
            query_split_size: int = 1,
            dtype: torch.dtype = torch.float32,
            device: str = 'cpu',
            save_if_missing: bool = False,
            dataset: Any = None,
            use_post: bool = False,
            **kwargs: Any
    ):
        super().__init__(name, top_k, dataset=dataset)
        self.vectorizer_document = vectorizer_document
        self.vectorizer_query = vectorizer_query
        self.sliding_window = sliding_window
        self.sliding_window_pooling = sliding_window_pooling
        self.sliding_window_size = sliding_window_size
        self.sliding_window_stride = sliding_window_stride
        self.sliding_window_type = sliding_window_type
        self.query_split_size = query_split_size
        self.dtype = dtype
        self.device = device
        self.save_if_missing = save_if_missing
        self.document_embeddings = None
        self.use_post = use_post

    def calculate_embeddings(self):
        # logger.info('Calculating embeddings for fact checks')
        documents = self.dataset.get_documents_texts()
        
        document_embeddings = self.vectorizer_document.vectorize(
            documents,
            save_if_missing=self.save_if_missing,
            normalize=True
        )

        document_embeddings = document_embeddings.transpose(
            0, 1)  # Rotate for matmul

        self.document_embeddings = document_embeddings.to(
            device=self.device, dtype=self.dtype)

    def dataset(self, dataset: Any):
        self.dataset = dataset
        self.calculate_embeddings()

    def retrieve(self, query: dict, post_split_size: int = 32) -> Any:
        if self.document_embeddings is None:
            self.calculate_embeddings()
            
        queries = []
        if self.use_post:
            queries = query['input']
        else:
            queries = query['prompt']

        if self.sliding_window:
            logger.info('Splitting query into windows.')
            window = slice_text(
                queries,
                self.sliding_window_type,
                self.sliding_window_size,
                self.sliding_window_stride
            )

            logger.info('Calculating embeddings for the windows')
            query_embeddings = self.vectorizer_query.vectorize(
                list(window),
                save_if_missing=self.save_if_missing,
                normalize=True
            )

            query_lengths = [len(post) for post in list(window)]
            segment_array = torch.tensor([
                i
                for i, num_windows in enumerate(query_lengths)
                for _ in range(num_windows)
            ])

            delimiters = list(gen_sliding_window_delimiters(
                query_lengths, self.query_split_size))
        else:
            # logger.info('Calculating embeddings for query')
            query_embeddings = self.vectorizer_query.vectorize(
                queries,
                save_if_missing=self.save_if_missing,
                normalize=True
            )
            delimiters = [(0, 1)]

        top_k_results = []
        top_k_sims_results = []
        top_k_texts = []
        # logger.info('Calculating similarity for data splits')
        for start_id in tqdm(range(0, len(queries), post_split_size)):
            end_id = start_id + post_split_size

            sims = torch.mm(
                query_embeddings[start_id:end_id].to(
                    device=self.device, dtype=self.dtype),
                self.document_embeddings
            )

            sorted_ids = torch.argsort(sims, descending=True, dim=1)
            sorted_similarities = torch.sort(sims, descending=True, dim=1).values
            if self.top_k is None:
                top_k = sorted_ids[:, :].tolist()
                top_k_sims = sims[:, top_k].tolist()
            else:
                if isinstance(self.top_k, float):
                    top_k = []
                    top_k_sims = []
                    for row_sorted_ids, row_sorted_sims in zip(sorted_ids, sorted_similarities):
                        mask = row_sorted_sims > self.top_k
                        top_k.append(row_sorted_ids[mask].tolist())
                        top_k_sims.append(row_sorted_sims[mask].tolist())
                else:
                    top_k = sorted_ids[:, :self.top_k].tolist()
                    top_k_sims = sims[:, top_k].tolist()
            
            top_k = [
                self.dataset.map_topK(row)
                for row in top_k
            ]        
            
            top_k_texts.extend([
                [
                    self.dataset.get_document(int(fc_id))
                    for fc_id in row
                ]
                for row in top_k
            ])
            top_k_results.extend(top_k)
            top_k_sims_results.extend([
                [
                    sim for sim in row
                ]
                for row in top_k_sims
            ])

        if len(queries) > 1:
            return top_k_texts, top_k_results
        else:
            return top_k_texts[0], top_k_results[0]
