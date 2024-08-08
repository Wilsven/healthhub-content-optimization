"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.6
"""

from typing import Any

import numpy as np
import pandas as pd
from alive_progress import alive_bar
from content_optimization.pipelines.feature_engineering.utils import (
    pool_embeddings,
    split_into_chunks,
)
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseTfidfVectorizer
from nltk.tokenize import sent_tokenize
from pytictoc import TicToc
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


def extract_keywords(
    merged_data: pd.DataFrame,
    cfg: dict[str, Any],
    only_confirmed_option: list[str],
    all_option: list[str],
    model: str,
    spacy_pipeline: str,
    stop_words: str,
    workers: int,
    use_mmr: bool,
    diversity: float,
    top_n: int,
) -> pd.DataFrame:
    """
    Extract keywords using KeyBERT model based on the provided parameters and
    return the DataFrame with the added `keybert_keywords` column containing the keywords.

    Args:
        merged_data (pd.DataFrame): The DataFrame containing the merged data.
        cfg (dict[str, Any]): The configuration dictionary containing the options to subset the merged data.
        only_confirmed_option (list[str]): The list of confirmed content categories if option is `only_confirmed`.
        all_option (list[str]): The list of all content categories if option is `all`.
        model (str): Use a custom embedding model. See https://maartengr.github.io/KeyBERT/guides/embeddings.html
        spacy_pipeline (str): The spaCy pipeline to be used for part-of-speech tagging. Standard is the 'en' pipeline.
        stop_words (str): The stop words to be used for keyphrase extraction.
        workers (int): Number of workers for spaCy part-of-speech tagging. To use all workers, set it to -1.
        use_mmr (bool): Whether to use Maximal Marginal Relevance (MMR) for keyphrase extraction.
        diversity (float): The diversity parameter for keyphrase extraction.
        top_n (int): The number of top keywords to extract.

    Returns:
         pd.DataFrame: The dataframe with the extracted keywords.
    """
    option = cfg["option"]
    contributor = cfg["contributor"]  # TODO: To allow for options other than HPB
    to_remove = cfg["to_remove"]

    # Subset the merged data based on the content categories provided as option
    if option == "only_confirmed":
        assert set(only_confirmed_option).issubset(
            set(all_option)
        ), "Invalid option(s). Please ensure selected content categories exist."
        filtered_data = merged_data.query("content_category in @only_confirmed_option")
    elif option == "all":
        filtered_data = merged_data.copy()
    else:
        assert (
            option in all_option
        ), "Invalid option. Please ensure selected content category exists."
        filtered_data = merged_data.query("content_category == @option")

    # To remove flagged articles or not and to subset by contributor
    if to_remove:
        filtered_data = filtered_data.query(
            f"pr_name == '{contributor}' and to_remove == {not to_remove}"
        ).reset_index(drop=True)
    else:
        filtered_data = filtered_data.query(f"pr_name == '{contributor}'").reset_index(
            drop=True
        )

    # Extract the raw content body text
    docs = filtered_data["extracted_content_body"].to_list()

    kw_model = KeyBERT(model)
    vectorizer = KeyphraseTfidfVectorizer(
        spacy_pipeline, stop_words=stop_words, workers=workers
    )

    # Marginally more performant
    # See: https://github.com/MaartenGr/KeyBERT/issues/156
    with TicToc():
        counts = vectorizer.fit(docs)
        vectorizer.fit = lambda *args, **kwargs: counts

        # If keyphrase vectorizer is specified, `keyphrase_ngram_range` is ignored
        keywords = kw_model.extract_keywords(
            docs,
            use_mmr=use_mmr,
            diversity=diversity,
            top_n=top_n,
            vectorizer=vectorizer,
        )

    # We iterate through the keywords, and reverse the order of the keywords
    # from the closest to the most distant and taking only the keywords themselves,
    # ignoring the distances
    keywords = [[kw[0] for kw in kws[::-1]] for kws in keywords]
    # Store keywords in new column
    filtered_data_with_keywords = filtered_data.copy()
    filtered_data_with_keywords[f"keywords_{model}"] = keywords

    return filtered_data_with_keywords


def generate_embeddings(
    filtered_data_with_keywords: pd.DataFrame,
    model: str,
    owner: str,
    trust_remote_code: bool,
    pooling_strategy: str,
    columns_to_keep_emb: list[str],
    columns_to_emb: list[str],
) -> pd.DataFrame:
    """
    Generates embeddings on columns specified in columns_to_emb and
    returns a DataFrame with added embeddings columns

    Args:
        filtered_data_with_keywords (pd.DataFrame): The filtered DataFrame with extracted keywords.
        model (str): Embedding model.
        owner (str): Owner of embedding model.
        trust_remote_code (bool): Specifies if trust_remote_code is required.
        pooling_strategy (str): Pooling strategy of chunk embeddings.
        columns_to_emb (list[str]): List of column names to generate embeddings.

    Returns:
        pd.DataFrame: The DataFrame with the generated embeddings.
    """

    df_filtered = filtered_data_with_keywords.copy()

    df_filtered = df_filtered.loc[:, columns_to_keep_emb]
    df_filtered["keywords_all-MiniLM-L6-v2"] = df_filtered[
        "keywords_all-MiniLM-L6-v2"
    ].apply(lambda x: " ".join(x))

    # Load the tokenizer and model
    if trust_remote_code:
        sentence_transformer = SentenceTransformer(
            f"{owner}/{model}", trust_remote_code=True
        )
    else:
        sentence_transformer = SentenceTransformer(f"{owner}/{model}")
    tokenizer = AutoTokenizer.from_pretrained(f"{owner}/{model}")
    max_length = sentence_transformer.max_seq_length

    embedding_dict = {col: [] for col in columns_to_emb}

    embeddings_data = df_filtered.copy()

    for col_name, embedding_list in embedding_dict.items():
        with alive_bar((embeddings_data["id"].nunique()), force_tty=True) as bar:
            print(f"Generating embeddings for {col_name}")
            for id in embeddings_data["id"].unique():
                text = embeddings_data.query("id == @id")[col_name].values[0]

                if not text:
                    # Store empty array
                    dim = sentence_transformer.get_sentence_embedding_dimension()
                    embeddings = np.empty((dim,), dtype=np.float32)
                else:
                    # Step 1: Split the article into sentences
                    sentences = sent_tokenize(text)

                    # Step 2: Tokenize sentences and split into chunks of max 256 tokens
                    chunks = split_into_chunks(sentences, max_length, tokenizer)

                    # Step 3: Encode each chunk to get their embeddings
                    chunk_embeddings = [
                        sentence_transformer.encode(chunk) for chunk in chunks
                    ]

                    # Step 4: Aggregate chunk embeddings to form a single embedding for the entire article
                    embeddings = pool_embeddings(
                        chunk_embeddings, strategy=pooling_strategy
                    )

                indices = embeddings_data.query("id == @id").index.values

                for _ in range(len(indices)):
                    embedding_list.append(embeddings)

                bar()

    for col_name, embedding_list in embedding_dict.items():
        embedding_col = f"{col_name}_embeddings"
        embeddings_data[embedding_col] = embedding_list

    return embeddings_data


def combine_embeddings_by_weightage(
    embeddings_data: pd.DataFrame,
    title_weight: float,
    article_category_names_weight: float,
    category_description_weight: float,
    extracted_content_body_weight: float,
    keywords_weight: float,
) -> pd.DataFrame:
    """
    Generates weighted embeddings on based on the provided parameters and
    returns a DataFrame with added 'combined_embeddings' column containing
    the weighted embeddings.

    Args:
        embeddings_data (pd.DataFrame): The DataFrame with the generated embeddings.
        title_weight (float): Weightage of title embeddings to use to compute combined embeddings.
        article_category_names_weight (float): Weightage of article_category_names embeddings to use to compute combined embeddings.
        category_description_weight (float): Weightage of category_description embeddings to use to compute combined embeddings.
        extracted_content_body_weight (float): Weightage of extracted_content_body embeddings to use to compute combined embeddings.
        keywords_weight (float): Weightage of keywords embeddings to use to compute combined embeddings.

    Returns:
        pd.DataFrame: The DataFrame with the weighted embeddings.
    """
    embeddings_df = embeddings_data.copy()

    embeddings_df["combined_embeddings"] = (
        embeddings_df["title_embeddings"] * title_weight
        + embeddings_df["article_category_names_embeddings"]
        * article_category_names_weight
        + embeddings_df["category_description_embeddings"] * category_description_weight
        + embeddings_df["extracted_content_body_embeddings"]
        * extracted_content_body_weight
        + embeddings_df["keywords_all-MiniLM-L6-v2_embeddings"] * keywords_weight
    )

    weighted_embeddings = embeddings_df[
        [
            "id",
            "title",
            "full_url",
            "extracted_content_body",
            "category_description",
            "keywords_all-MiniLM-L6-v2",
            "title_embeddings",
            "article_category_names_embeddings",
            "category_description_embeddings",
            "extracted_content_body_embeddings",
            "keywords_all-MiniLM-L6-v2_embeddings",
            "combined_embeddings",
        ]
    ]
    return weighted_embeddings
