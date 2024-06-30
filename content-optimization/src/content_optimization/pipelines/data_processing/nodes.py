"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.6
"""

import os
import re
from typing import Any, Callable

import pandas as pd
from alive_progress import alive_bar
from content_optimization.utils.columns import select_and_rename_columns
from content_optimization.utils.extract import HTMLExtractor


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
            where the keys are the filenames and the values loads the raww excel data as
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

    with alive_bar(
        len(all_contents), title="Standardizing columns", force_tty=True
    ) as bar:
        for filename, partition_load_func in all_contents.items():
            # Get content category from filename
            content_category = re.sub(r"export-published-", "", filename.split("_")[0])
            bar.text(f"Standardizing for {content_category}")

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

            # Mark articles with no content or with dummy content in `to_remove` column
            df["to_remove"] = df["content_body"].apply(
                lambda x: (
                    False
                    if pd.notna(x) and re.search(r"(<[div|p|h2].*?>)", str(x))
                    else True
                )
            )

            all_contents_standardized[content_category] = df
            bar()

    return all_contents_standardized


def extract_data(
    all_contents_standardized: dict[str, Callable[[], Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """
    Extracts data from processed content and stores it in parquet files
    and text files.

    Args:
        all_contents_standardized (dict[str, Callable[[], Any]]):
            A dictionary containing the standardized `partitions.PartitionedDataset`
            where the keys are the content categories and thevalues loads the
            standardized parquet data as `pandas.DataFrame`.

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

    with alive_bar(
        len(all_contents_standardized), title="Extracting data", force_tty=True
    ) as bar:
        for content_category, partition_load_func in all_contents_standardized.items():
            bar.text(f"Extracting: {content_category}")

            # Load partition data
            df = partition_load_func()

            # Initialise new columns in dataframe to store extracted data
            df["has_table"] = None
            df["has_image"] = None
            df["related_sections"] = None
            df["extracted_links"] = None
            df["extracted_headers"] = None
            df["extracted_content_body"] = None

            for index, row in df.iterrows():
                # Skip extraction for those articles flagged for removal
                if row["to_remove"]:
                    continue

                # Replace all forward slashes with hyphens to avoid saving as folders
                title = re.sub(r"\/", "-", row["title"]).strip()

                # Get the HTML content
                html_content = row["content_body"]

                # Extract text from HTML using the HTMLExtractor Class
                extractor = HTMLExtractor(html_content)
                has_table = extractor.check_for_table()
                has_image = extractor.check_for_image()
                related_sections = extractor.extract_related_sections()
                extracted_links = extractor.extract_links()
                extracted_headers = extractor.extract_headers()
                extracted_content_body = extractor.extract_text()

                # Store extracted data into the dataframe
                df.at[index, "has_table"] = has_table
                df.at[index, "has_image"] = has_image
                df.at[index, "related_sections"] = related_sections
                df.at[index, "extracted_links"] = extracted_links
                df.at[index, "extracted_headers"] = extracted_headers
                df.at[index, "extracted_content_body"] = extracted_content_body

                # If `extracted_content_body` is empty, we update flag to remove
                if extracted_content_body == "":
                    df.at[index, "to_remove"] = True

                # Substitute forbidden characters for filenames with _
                title = re.sub(r'[<>:"/\\|?*]', "_", title)

                # Truncate title to 25 characters and append the id
                # See: https://github.com/Wilsven/healthhub-content-optimization/issues/42
                title = title[:25] + f"_{row['id']}"

                # Store text files in its own folder named `content_category`
                all_extracted_text[os.path.join(content_category, title)] = (
                    extracted_content_body
                )

            # Store dataframes in a parquet file named `content_category`
            all_contents_extracted[content_category] = df
            bar()

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
