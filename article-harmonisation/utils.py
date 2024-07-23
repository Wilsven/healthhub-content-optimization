    
import pyarrow.parquet as pq
import os


ROOT = os.getcwd()
MERGED_DATA_DIRECTORY = (
    f"{ROOT}/content-optimization/data/03_primary/merged_data.parquet/2024-07-18T08.05.43.213Z/merged_data.parquet"
)

CONTENT_BODY = "extracted_content_body"
EXTRACTED_HEADERS = "extracted_headers"
KEY_PARQUET_INFO = ["title", "article_category_names", EXTRACTED_HEADERS, CONTENT_BODY]
TO_REMOVE = ["", " "]

table = pq.read_table(MERGED_DATA_DIRECTORY)


def concat_headers_to_content(extracted_article_content,article_list):
    final_configured_articles = []
    # for loop iterating through each 
    for content in extracted_article_content:
        for article_content in article_list:
            if article_content in str(content):
                idx = extracted_article_content.index(content)
                str_content = content.as_py()
                article_headers = list(table[EXTRACTED_HEADERS][idx])
                split_content = []
                for header_details in article_headers:
                    header = header_details[0].as_py()
                    if split_content == []:
                        split_content.extend(str_content.split(header))
                    else:
                        last_content = split_content.pop()
                        split_content.extend(last_content.split(header))
                    split_content[-1] = "Keypoint: "+ header + "\n" + split_content[-1][1:]
                for content in split_content:
                    if content in TO_REMOVE:
                        split_content.pop(split_content.index(content))
                final_configured_articles.append(split_content)
                
    return final_configured_articles
