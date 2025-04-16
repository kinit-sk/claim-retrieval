import logging
from datetime import datetime
from os.path import join as join_path
from typing import Iterable, Optional, Any

import pandas as pd

from src.datasets.custom_types import Language, is_in_distribution, combine_distributions
from src.datasets.multiclaim.multiclaim_dataset import MultiClaimDataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiClaimMetadataDataset(MultiClaimDataset):
    """MultiClaim dataset extended with metadata for each fact-check.

    Class for multiclaim dataset that can load different variants. It requires data from `data/datasets/multiclaim`.

    Initialization Attributes:
        crosslingual: bool  If `True`, only crosslingual pairs (fact-check and post in different languages) are loaded.
        fact_check_fields: Iterable[str]  List of fields used to generate the final `str` representation for fact-checks. Supports `claim` and `title`.
        fact_check_language: Optional[Language]  If a `Language` is specified, only fact-checks with that language are selected.
        language: Optional[Language]  If a `Language` is specified, only fact-checks and posts with that language are selected.
        post_language: Optional[Language]  If a `Language` is specified, only posts with that language are selected.
        split: `train`, `test` or `dev`. `None` means all the samples.
        version: 'original' or 'english'. Language version of the dataset.

        Also check `Dataset` attributes.

        After `load` is called, following attributes are accesible:
            fact_check_post_mapping: list[tuple[int, int]]  List of Factcheck-Post id pairs.
            id_to_documents: dict[int, str]  Factcheck id -> Factcheck text
            id_to_post: dict[int, str]  Post id -> Post text


    Methods:
        load: Loads the data from the csv files. Populates `id_to_documents`, `id_to_post` and `fact_check_post_mapping` attributes.
    """

    our_dataset_path = join_path('.','datasets', 'multiclaim')
    csvs_loaded = False

    def __init__(
        self,
        crosslingual: bool = False,
        fact_check_fields: Iterable[str] = ('claim', ),
        fact_check_language: Optional[Language] = None,
        language: Optional[Language] = None,
        post_language: Optional[Language] = None,
        split: Optional[str] = None,
        version: str = 'original',
        domain: Optional[str] = None,
        entity: Optional[str] = None,
        date_range: Optional[tuple] = None,
        use_metadata: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert all(field in ('claim', 'title') for field in fact_check_fields)
        assert split in (None, 'dev', 'test', 'train')
        assert version in ('english', 'original')

        self.crosslingual = crosslingual
        self.fact_check_fields = fact_check_fields
        self.fact_check_language = fact_check_language
        self.language = language
        self.post_language = post_language
        self.split = split
        self.version = version
        self.domain = domain
        self.entity = entity
        self.date_range = date_range
        self.use_metadata = use_metadata
        
    def format_language(self, language):
        return {
            'eng': 'English (EN)',
            'ara': 'Arabic (AR)',
            'por': 'Portuguese (PT)',
            'fra': 'French (FR)',
            'nld': 'Dutch (NL)',
            'msa': 'Malay (MS)',
            'ben': 'Bengali (BN)',
            'pol': 'Polish (PL)',
            'spa': 'Spanish (ES)',
            'kor': 'Korean (KO)',
            'ita': 'Italian (IT)',
            'rus': 'Russian (RU)',
            'hbs': 'Serbo-Croatian (SH)',
            'hun': 'Hungarian (HU)',
            'tur': 'Turkish (TR)',
            'cat': 'Catalan (CA)',
            'deu': 'German (DE)',
            'hin': 'Hindi (HI)',
            'dan': 'Danish (DA)',
            'co': 'Corsican (CO)',
            'jw': 'Javanese (JV)',
            'ell': 'Greek (EL)',
            'uz': 'Uzbek (UZ)',
            'ukr': 'Ukrainian (UK)',
            'mkd': 'Macedonian (MK)',
            'heb': 'Hebrew (HE)',
            'tel': 'Telugu (TE)',
            'mal': 'Malayalam (ML)',
            'zho': 'Chinese (ZH)',
            'sin': 'Sinhala (SI)',
            'tam': 'Tamil (TA)',
            'khm': 'Khmer (KM)',
            'sqi': 'Albanian (SQ)',
            'asm': 'Assamese (AS)',
            'tha': 'Thai (TH)',
            'nor': 'Norwegian (NO)',
            'fas': 'Persian (FA)',
            'mya': 'Burmese (MY)',
            'tgl': 'Tagalog (TL)',
            'bul': 'Bulgarian (BG)',
            'tg': 'Tajik (TG)',
            'slk': 'Slovak (SK)',
            'aze': 'Azerbaijani (AZ)',
            'ka': 'Georgian (KA)',
            'kk': 'Kazakh (KK)',
            'ron': 'Romanian (RO)',
            'ces': 'Czech (CS)',
            'zu': 'Zulu (ZU)',
            'snd': 'Sindhi (SD)',
            'hi-Latn': 'Hindi (HI)',
            'yo': 'Yoruba (YO)',
            'gl': 'Galician (GL)',
            'kn': 'Kannada (KN)',
            'ga': 'Irish (GA)',
            'fin': 'Finnish (FI)',
            'fil': 'Filipino (PH)',
            'gu-Latn': 'Gujarati (GU)',
            'sl': 'Slovenian (SL)',
            'jpn': 'Japanese (JA)',
            'mg': 'Malagasy (MG)',
            'su': 'Sundanese (SU)',
            'ay': 'Aymara (AY)',
            'gu': 'Gujarati (GU)',
            'zh-Latn': 'Chinese (ZH)',
            'qu': 'Quechua (QU)',
            'af': 'Afrikaans (AF)',
            'la': 'Latin (LA)',
            'so': 'Somali (SO)',
            'sw': 'Swahili (SW)',
            'eu': 'Basque (EU)',
            'jv': 'Javanese (JV)',
            'et': 'Estonian (ET)',
            'ceb': 'Cebuano (CEB)',
            'rw': 'Kinyarwanda (RW)',
            'lg': 'Ganda (LG)',
            'ha': 'Hausa (HA)',
            'kri': 'Krio (KRI)',
            'eo': 'Esperanto (EO)',
            'nso': 'Northern Sotho (NSO)',
            'sv': 'Swedish (SV)',
            'mt': 'Maltese (MT)',
            'kn-Latn': 'Kannada (KN)',
            'sm': 'Samoan (SM)',
            'mr-Latn': 'Marathi (MR)',
            'lv': 'Latvian (LV)',
            'ts': 'Tsonga (TS)',
            'mi': 'Maori (MI)',
            'lb': 'Luxembourgish (LB)',
            'tk': 'Turkmen (TK)',
            'und': 'Undetermined (UN)',
            'tt': 'Tatar (TT)',
            'sd': 'Sindhi (SD)',
            'bh': 'Bihari (BH)',
            'mr': 'Marathi (MR)',
            'sa': 'Sanskrit (SA)',
            'pa': 'Punjabi (PA)',
            'ta-Latn': 'Tamil (TA)',
        }.get(language, language)

    def format_fact_check(self, claim, language, date, organization):
        
        if not self.use_metadata:
            return self.clean_text(claim[self.version == 'english'])
        
        if pd.isna(date) or date == '':
            formatted_date = 'Not available'
        else:
            if isinstance(date, str):
                date_obj = datetime.fromisoformat(date)
                formatted_date = date_obj.strftime("%Y/%m/%d")
            else:
                formatted_date = date.strftime("%Y/%m/%d")
        
        formatted_langauge = self.format_language(language)
        
        return f"""Fact-checked claim: {self.clean_text(claim[self.version == 'english'])}
Language: {formatted_langauge}
Published date: {formatted_date}
Fact-checking organization: {organization}"""

    def get_fact_check(self, fc_id: int, use_metadata: bool = False) -> Any:
        if use_metadata:            
            return self.id_to_formatted_fc[fc_id]
            
        return self.id_to_fc[fc_id]

    def load(self):

        self.maybe_load_csvs()

        df_posts = self.df_posts.copy()
        df_fact_checks = self.df_fact_checks.copy()
        df_fact_check_post_mapping = self.df_fact_check_post_mapping.copy()

        if self.split:
            split_post_ids = set(self.split_post_ids(self.split))
            df_posts = df_posts[df_posts.index.isin(split_post_ids)]
            logger.info(f'Filtering by split: {len(df_posts)} posts remaining.')  # nopep8

        # Filter fact-checks by the language detected in claim
        if self.language or self.fact_check_language:
            df_fact_checks = df_fact_checks[df_fact_checks['claim'].apply(
                # claim[2] is the language distribution
                lambda claim: is_in_distribution(
                    self.language or self.fact_check_language, claim[2])
            )]
            logger.info(
                f'Filtering fact-checks by language: {len(df_fact_checks)} posts remaining.')

        # Filter posts by the language detected in the combined distribution.
        # There was a slight bug in the paper version of post language filtering and in effect we have slightly more posts per language
        # in the paper. The original version did not take into account that the sum of percentages in a distribution does not have to be equal to 1.
        if self.language or self.post_language:
            def post_language_filter(row):
                texts = [
                    text
                    for text in [row['text']] + row['ocr']
                    if text  # Filter empty texts
                ]
                distribution = combine_distributions(texts)
                return is_in_distribution(self.language or self.post_language, distribution)

            df_posts = df_posts[df_posts.apply(post_language_filter, axis=1)]
            logger.info(f'Filtering posts by language: {len(df_posts)} posts remaining.')  # nopep8

        # Create mapping variable
        post_ids = set(df_posts.index)
        fact_check_ids = set(df_fact_checks.index)
        fact_check_post_mapping = set(
            (fact_check_id, post_id)
            for fact_check_id, post_id in df_fact_check_post_mapping.itertuples(index=False, name=None)
            if fact_check_id in fact_check_ids and post_id in post_ids
        )
        logger.info(f'Mappings remaining: {len(fact_check_post_mapping)}.')

        # Leave only crosslingual samples
        if self.crosslingual:

            crosslingual_mapping = set()
            for fact_check_id, post_id in fact_check_post_mapping:

                # Here, we assume that all fact-check claims have exactly one language assigned (via Google Translate)
                # We will have to rework this part if we want to support multiple detected languages.
                fact_check_language = df_fact_checks.loc[fact_check_id, 'claim'][2][0][0]

                post_texts = [
                    text
                    for text in [df_posts.loc[post_id, 'text']] + df_posts.loc[post_id, 'ocr']
                    if text
                ]
                post_distribution = combine_distributions(post_texts)

                if not is_in_distribution(fact_check_language, post_distribution):
                    crosslingual_mapping.add((fact_check_id, post_id))

            fact_check_post_mapping = crosslingual_mapping
            logger.info(f'Crosslingual mappings remaining: {len(fact_check_post_mapping)}')  # nopep8

        # Filtering posts if any crosslingual or language filter were applied
        remaining_post_ids = set(post_id for _, post_id in fact_check_post_mapping)
        # df_posts = df_posts[df_posts.index.isin(remaining_post_ids)]
        logger.info(f'Filtering posts.')
        logger.info(f'Posts remaining: {len(df_posts)}')
        
        if self.domain:
            df_fact_checks = df_fact_checks[df_fact_checks['domain'] == self.domain]
            logger.info(f'Filtering fact-checks by domain: {len(df_fact_checks)} fcs remaining.')

        if self.entity:
            # check if self.entity is in the fact-checks, self.entity is string
            df_fact_checks = df_fact_checks[df_fact_checks['claim'].str.contains(self.entity)]
            logger.info(f'Filtering fact-checks by entity: {len(df_fact_checks)} fcs remaining.')

        if self.date_range:
            start_date, end_date = self.date_range
            start_date = datetime.strptime(start_date, '%Y/%m/%d')
            end_date = datetime.strptime(end_date, '%Y/%m/%d')
            df_fact_checks = df_fact_checks[(df_fact_checks['published_at'] >= start_date) & (df_fact_checks['published_at'] <= end_date)]

        # Create object attributesid_to_post
        self.fact_check_post_mapping = list(fact_check_post_mapping)
        
        self.id_to_formatted_fc = {
            fact_check_id: self.format_fact_check(claim, language, date, org)
            for fact_check_id, claim, language, date, org in zip(df_fact_checks.index, df_fact_checks['claim'], df_fact_checks['language'], df_fact_checks['published_at'], df_fact_checks['domain'])
        }

        self.id_to_documents = {
            fact_check_id: self.format_fact_check(claim, language, date, org)
            for fact_check_id, claim, language, date, org in zip(df_fact_checks.index, df_fact_checks['claim'], df_fact_checks['language'], df_fact_checks['published_at'], df_fact_checks['domain'])
        }
        
        self.id_to_fc = {
            fact_check_id: self.clean_text(claim[self.version == 'english'])
            for fact_check_id, claim in zip(df_fact_checks.index, df_fact_checks['claim'])
        }
        

        self.id_to_post = dict()
        for post_id, post_text, ocr in zip(df_posts.index, df_posts['text'], df_posts['ocr']):
            texts = list()
            if post_text:
                texts.append(post_text[self.version == 'english'])
            for ocr_text in ocr:
                texts.append(self.maybe_clean_ocr(
                    ocr_text[self.version == 'english']))
            self.id_to_post[post_id] = self.clean_text(' '.join(texts))

        return self
