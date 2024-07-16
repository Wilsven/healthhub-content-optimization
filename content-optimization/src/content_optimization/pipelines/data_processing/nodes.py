"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.6
"""

import os
import re
from typing import Any, Callable

import pandas as pd
from tqdm import tqdm

from .extractor import HTMLExtractor
from .utils import (
    flag_articles_to_remove_after_extraction,
    flag_articles_to_remove_before_extraction,
    select_and_rename_columns,
)


def standardize_columns(
    all_contents: dict[str, Callable[[], Any]],
    columns_to_add_cfg: dict[str, list[str]],
    columns_to_keep_cfg: dict[str, list[str]],
    default_columns: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Standardizes the columns of multiple dataframes in a dictionary.

    This function takes in a dictionary of dataframes, where each dataframe
    is associated with a filename. It then standardizes the columns of each dataframe
    by performing the following steps:

    1. Get the content category from the filename.
    2. Load the dataframe using the provided partition function.
    3. Standardize the column names by selecting and renaming the columns.
    4. Mark articles with no content or with dummy content in the `to_remove` column.
    5. Add the standardized dataframe to the `all_contents_standardized` dictionary.

    The function returns a dictionary mapping content categories to the standardized dataframes.

    Args:
        all_contents (dict[str, Callable[[], Any]]):
            A dictionary containing the raw `partitions.PartitionedDataset`
            where the keys are the filenames and the values loads the raw excel data as
            `pandas.DataFrame`.

        columns_to_add_cfg (dict[str, list[str]]):
            A dictionary mapping content categories to lists of column names to add.

        columns_to_keep_cfg (dict[str, list[str]]):
            A dictionary mapping content categories to lists of column names to keep.

        default_columns (list[str]):
            A list of default column names to rename the columns of the dataframes to.

    Returns:
        dict[str, pd.DataFrame]:
            A dictionary that contains the standardized dataframes stored as partitioned
            parquet files, where the keys are the content categories and the values are
            the corresponding dataframes.
    """
    all_contents_standardized = {}

    pbar = tqdm(all_contents.items())

    for filename, partition_load_func in pbar:
        # Get content category from filename
        content_category = re.sub(r"export-published-", "", filename.split("_")[0])
        pbar.set_description(f"Standardizing: {content_category}")

        # Load partition data
        df = partition_load_func()

        # Standardize column names
        columns_to_add = columns_to_add_cfg.get(content_category, None)
        columns_to_keep = columns_to_keep_cfg.get(content_category, None)

        # Standardize columns
        df = select_and_rename_columns(
            df, columns_to_add, columns_to_keep, default_columns, content_category
        )

        # Strip all whitespaces across all strings in dataframe
        # See: https://github.com/Wilsven/healthhub-content-optimization/issues/53
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        # Mark articles with no content, was rejected by Excel due to a "Value
        # exceeded maximum cell size" error or with dummy content in `to_remove` column
        df = flag_articles_to_remove_before_extraction(df)

        all_contents_standardized[content_category] = df

    return all_contents_standardized


def add_contents(
    all_contents_standardized: dict[str, Callable[[], Any]],
    missing_contents: dict[str, Callable[[], Any]],
) -> dict[str, Callable[[], Any]]:
    excel_errors = {}
    all_contents_added = {}

    # Fetch all txt files from 01_raw to correct the Excel error
    for file_path, load_func in missing_contents.items():
        text = load_func()
        # Extract friendly url that is used as filename
        friendly_url = file_path.split("/")[-1]
        # Store in dictionary
        excel_errors[friendly_url] = text

    pbar = tqdm(all_contents_standardized.items())

    for content_category, partition_load_func in pbar:
        pbar.set_description(f"Adding: {content_category}")
        df = partition_load_func()
        for friendly_url, text in excel_errors.items():
            # Fetch rows where `friendly_url` column value must match filename
            article_index = df.index[df["friendly_url"] == friendly_url]
            # Skip if the index is empty
            if article_index.empty:
                continue
            # Assign Raw HTML content to `content_body` column
            df.loc[article_index, "content_body"] = text
            # TODO: Confirm if you wish to remove Multilingual content
            # Change remove_type if the content body is in a language other than English
            if re.search(r"(malay|tamil|chinese)", friendly_url):
                df.loc[article_index, "remove_type"] = "Multilingual"
            # Whitelist remaining content
            else:
                df.loc[article_index, "to_remove"] = False
                df.loc[article_index, "remove_type"] = None

        all_contents_added[content_category] = df

    return all_contents_added


def extract_data(
    all_contents_added: dict[str, Callable[[], Any]],
    word_count_cutoff: int,
    whitelist: list[int],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """
    Extracts data from processed content and stores it in parquet files
    and text files.

    Args:
        all_contents_added (dict[str, Callable[[], Any]]):
            A dictionary containing the standardized `partitions.PartitionedDataset`
            where the keys are the content categories and the values loads the
            standardized parquet data as `pandas.DataFrame`.

        word_count_cutoff (int): The minimum number of words in an article
            to be considered before flagging for removal.

        whitelist (list[int]): The list of article IDs to keep. See https://bitly.cx/IlwNV.

    Returns:
        tuple[dict[str, pd.DataFrame], dict[str, str]]:
            A tuple containing two dictionaries. The first dictionary contains
            the extracted data stored as partitioned parquet files, where the keys
            are the content categories and the values are the corresponding dataframes.
            The second dictionary contains the extracted text stored as partitioned
            text files, where the keys are the file paths and the values are the extracted
            text.
    """
    all_contents_extracted = {}  # to store as partitioned parquet files
    all_extracted_text = {}  # to store as partitioned text files

    pbar = tqdm(all_contents_added.items())

    for content_category, partition_load_func in pbar:
        pbar.set_description(f"Extracting: {content_category}")
        # Load partition data
        df = partition_load_func()

        # Initialise new columns in dataframe to store extracted data
        df["has_table"] = False
        df["has_image"] = False
        df["related_sections"] = None
        df["extracted_tables"] = None
        df["extracted_links"] = None
        df["extracted_headers"] = None
        df["extracted_img_alt_text"] = None
        df["extracted_content_body"] = ""

        for index, row in df.iterrows():
            # Skip extraction for those articles flagged for removal unless whitelisted
            if row["to_remove"]:
                # Check if the article is in the whitelist
                if row["id"] not in whitelist:
                    continue
                else:
                    # Whitelist article
                    df.at[index, "to_remove"] = False

            # Replace all forward slashes with hyphens to avoid saving as folders
            title = re.sub(r"\/", "-", row["title"]).strip()

            # Get the HTML content for extraction and relevant data for logging
            content_name = row["content_name"]
            full_url = row["full_url"]
            html_content = row["content_body"]

            # Extract text from HTML using the HTMLExtractor Class
            extractor = HTMLExtractor(
                content_name, content_category, full_url, html_content
            )
            has_table = extractor.check_for_table()
            has_image = extractor.check_for_image()
            related_sections = extractor.extract_related_sections()
            extracted_tables = extractor.extract_tables()
            extracted_links = extractor.extract_links()
            extracted_headers = extractor.extract_headers()
            extracted_img_alt_text = extractor.extract_alt_text_from_img()
            extracted_content_body = extractor.extract_text()

            # Store extracted data into the dataframe
            df.at[index, "has_table"] = has_table
            df.at[index, "has_image"] = has_image
            df.at[index, "related_sections"] = related_sections
            df.at[index, "extracted_tables"] = extracted_tables
            df.at[index, "extracted_links"] = extracted_links
            df.at[index, "extracted_headers"] = extracted_headers
            df.at[index, "extracted_img_alt_text"] = extracted_img_alt_text
            df.at[index, "extracted_content_body"] = extracted_content_body

            # Substitute forbidden characters for filenames with _
            title = re.sub(r'[<>:"/\\|?*]', "_", title)

            # Truncate title to 25 characters and append the id
            # See: https://github.com/Wilsven/healthhub-content-optimization/issues/42
            title = title[:25] + f"_{row['id']}"

            # Store text files in its own folder named `content_category`
            all_extracted_text[os.path.join(content_category, title)] = (
                extracted_content_body
            )

        # After extraction, we flag to remove articles with no content,
        # duplicated content, duplicated URL or below word count cutoff
        df = flag_articles_to_remove_after_extraction(df, word_count_cutoff, whitelist)

        # Store dataframes in a parquet file named `content_category`
        all_contents_extracted[content_category] = df

    return all_contents_extracted, all_extracted_text


def merge_data(all_contents_extracted: dict[str, Callable[[], Any]]) -> pd.DataFrame:
    """
    Merge the data from multiple partitioned dataframes into a single `pandas.DataFrame`.

    Parameters:
        all_contents_extracted (dict[str, Callable[[], Any]]):
            A dictionary containing the `partitions.PartitionedDataset`
            where the values loads the parquet data as `pandas.DataFrame`.

    Returns:
        pd.DataFrame: The merged dataframe.
    """
    merged_df = pd.DataFrame()
    for _, partition_load_func in all_contents_extracted.items():
        # Load partition data
        df = partition_load_func()
        merged_df = pd.concat([merged_df, df], axis=0, ignore_index=True)

    return merged_df
