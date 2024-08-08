"""
This is a boilerplate pipeline 'clustering'
generated using Kedro 0.19.6
"""

from content_optimization.pipelines.clustering.nodes import (
    cluster_viz,
    generate_clusters,
    generate_subclusters,
    merge_ground_truth_to_data,
    update_edges_dataframe,
)
from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=merge_ground_truth_to_data,
                inputs=[
                    "ground_truth_data",
                    "params:content_contributor",
                    "weighted_embeddings",
                ],
                outputs="merged_df_with_groundtruth",
                name="merge_ground_truth_to_data_node",
            ),
            node(
                func=generate_clusters,
                inputs=[
                    "merged_df_with_groundtruth",
                    "params:neo4j_config",
                    "params:sim_weightage.weight_title",
                    "params:sim_weightage.weight_cat",
                    "params:sim_weightage.weight_desc",
                    "params:sim_weightage.weight_body",
                    "params:sim_weightage.weight_combined",
                    "params:sim_weightage.weight_kws",
                    "params:set_threshold",
                ],
                outputs=[
                    "pred_cluster",
                    "clustered_nodes",
                    "unclustered_nodes",
                    "metrics",
                    "initial_cluster_size",
                ],
                name="generate_clusters_node",
            ),
            node(
                func=generate_subclusters,
                inputs=[
                    "weighted_embeddings",
                    "pred_cluster",
                    "params:umap_parameters",
                ],
                outputs=["updated_pred_cluster", "final_cluster_size"],
                name="generate_subclusters_node",
                # ["final_predicted_cluster","final_cluster_size","final_metrics","final_neo4j_clustered_data","final_neo4j_unclustered_data"]
            ),
            node(
                func=update_edges_dataframe,
                inputs=["clustered_nodes", "updated_pred_cluster"],
                outputs=["final_unclustered_nodes", "final_clustered_nodes"],
                name="update_edges_dataframe_node",
            ),
            node(
                func=cluster_viz,
                inputs=[
                    "final_clustered_nodes",
                    "final_unclustered_nodes",
                ],
                outputs=None,
                name="cluster_viz_node",
            ),
        ]
    )
