"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.6
"""

from content_optimization.pipelines.feature_engineering.nodes import (
    combine_embeddings_by_weightage,
    extract_keywords,
    generate_embeddings,
)
from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=extract_keywords,
                inputs=[
                    "merged_data",
                    "params:cfg",
                    "params:selection_options.only_confirmed",
                    "params:selection_options.all",
                    "params:keywords.model",
                    "params:keywords.spacy_pipeline",
                    "params:keywords.stop_words",
                    "params:keywords.workers",
                    "params:keywords.use_mmr",
                    "params:keywords.diversity",
                    "params:keywords.top_n",
                ],
                outputs="filtered_data_with_keywords",
                name="extract_keywords_node",
            ),
            node(
                func=generate_embeddings,
                inputs=[
                    "filtered_data_with_keywords",
                    "params:embeddings.model",
                    "params:embeddings.owner",
                    "params:embeddings.trust_remote_code",
                    "params:embeddings.pooling_strategy",
                    "params:columns_to_keep_emb",
                    "params:columns_to_emb",
                ],
                outputs="embeddings_data",
                name="generate_embeddings_node",
            ),
            node(
                func=combine_embeddings_by_weightage,
                inputs=[
                    "embeddings_data",
                    "params:embeddings_weightage.title",
                    "params:embeddings_weightage.article_category_names",
                    "params:embeddings_weightage.category_description",
                    "params:embeddings_weightage.extracted_content_body",
                    "params:embeddings_weightage.keywords",
                ],
                outputs="weighted_embeddings",
                name="combine_embeddings_by_weightage",
            ),
        ]
    )
