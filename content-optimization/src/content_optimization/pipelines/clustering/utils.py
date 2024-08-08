import logging

import hdbscan
import nltk

# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from .ctfidf import CTFIDFVectorizer

logging.basicConfig(level=logging.INFO)

nltk.download("wordnet")
stopwords = list(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clear_db(tx):
    logging.info("Clearing database")
    tx.run("MATCH (n) DETACH DELETE n")


def create_graph_nodes(tx, doc):
    # logging.info("Create nodes")
    tx.run(
        """
    CREATE (d:Article {
        id: $id,
        title: $title,
        url: $url,
        content: $content,
        meta_desc: $meta_description,
        vector_body: $vector_body,
        vector_title: $vector_title,
        vector_category: $vector_category,
        vector_desc: $vector_desc,
        vector_kws: $vector_keywords,
        vector_combined: $vector_combined,
        ground_truth: $ground_truth
    })""",
        id=doc["id"],
        title=doc["title"],
        url=doc["full_url"],
        content=doc["extracted_content_body"],
        meta_description=doc["category_description"],
        vector_title=doc["title_embeddings"],
        vector_category=doc["article_category_names_embeddings"],
        vector_desc=doc["category_description_embeddings"],
        vector_body=doc["extracted_content_body_embeddings"],
        vector_combined=doc["combined_embeddings"],
        vector_keywords=doc["keywords_all-MiniLM-L6-v2_embeddings"],
        ground_truth=doc["ground_truth_cluster"],
    )


def calculate_similarity(tx, vector_name):
    print(f"Calculate Similarity For {vector_name}")
    query = f"""
    MATCH (a:Article), (b:Article)
    WHERE a.id < b.id
    RETURN a.id AS node_1_id, b.id AS node_2_id,
           gds.similarity.cosine(a.{vector_name}, b.{vector_name}) AS similarity
    """
    result = tx.run(query)
    return {
        (record["node_1_id"], record["node_2_id"]): record["similarity"]
        for record in result
    }


def fetch_ground_truth(session):
    print("Fetching Ground Truth")
    query = """
    MATCH (a:Article)
    RETURN a.id AS id, a.ground_truth AS ground_truth
    """
    result = session.run(query)
    ground_truth = {record["id"]: record["ground_truth"] for record in result}
    return ground_truth


def combine_similarities(
    session,
    weight_title,
    weight_cat,
    weight_desc,
    weight_body,
    weight_combined,
    weight_kws,
):
    print("Combining Similarity")
    ground_truth = fetch_ground_truth(session)

    similarities_title = session.execute_write(
        lambda tx: calculate_similarity(tx, "vector_title")
    )
    similarities_cat = session.execute_write(
        lambda tx: calculate_similarity(tx, "vector_category")
    )
    similarities_desc = session.execute_write(
        lambda tx: calculate_similarity(tx, "vector_desc")
    )
    similarities_body = session.execute_write(
        lambda tx: calculate_similarity(tx, "vector_body")
    )
    similarities_combined = session.execute_write(
        lambda tx: calculate_similarity(tx, "vector_combined")
    )
    similarities_kws = session.execute_write(
        lambda tx: calculate_similarity(tx, "vector_kws")
    )

    combined_similarities = []

    for key in similarities_title.keys():
        if (
            key in similarities_cat
            and key in similarities_desc
            and key in similarities_body
            and key in similarities_combined
            and key in similarities_kws
        ):
            node_1_ground_truth = ground_truth.get(key[0])
            node_2_ground_truth = ground_truth.get(key[1])

            combined_similarity = {
                "node_1_id": key[0],
                "node_2_id": key[1],
                "node_1_ground_truth": node_1_ground_truth,
                "node_2_ground_truth": node_2_ground_truth,
                "similarities_title": similarities_title[key],
                "similarities_cat": similarities_cat[key],
                "similarities_desc": similarities_desc[key],
                "similarities_body": similarities_body[key],
                "similarities_combined": similarities_combined[key],
                "similarities_kws": similarities_kws[key],
            }
            combined_similarities.append(combined_similarity)

    df = pd.DataFrame(combined_similarities)
    columns_to_fill = [
        "similarities_title",
        "similarities_cat",
        "similarities_desc",
        "similarities_body",
        "similarities_combined",
        "similarities_kws",
    ]
    df[columns_to_fill] = df[columns_to_fill].fillna(0)
    df["weighted_similarity"] = (
        weight_title * df["similarities_title"]
        + weight_cat * df["similarities_cat"]
        + weight_desc * df["similarities_desc"]
        + weight_body * df["similarities_body"]
        + weight_combined * df["similarities_combined"]
        + weight_kws * df["similarities_kws"]
    )
    return df


def median_threshold(combined_similarities):
    print("Calculating Median")
    df = combined_similarities.dropna(
        subset=["node_1_ground_truth", "node_2_ground_truth"]
    )
    df_filtered = df[df["node_1_ground_truth"] == df["node_2_ground_truth"]]
    threshold = df_filtered["weighted_similarity"].median()
    return threshold


def create_sim_edges(tx, similarities, threshold):
    print("Creating Edges")
    logging.info("Creating edges")
    for _, record in similarities.iterrows():
        if record["weighted_similarity"] > threshold:
            tx.run(
                """
                MATCH (a:Article {id: $node_1_id}), (b:Article {id: $node_2_id})
                CREATE (a)-[:SIMILAR {similarity: $weighted_similarity}]->(b)
                """,
                node_1_id=record["node_1_id"],
                node_2_id=record["node_2_id"],
                weighted_similarity=record["weighted_similarity"],
            )


def drop_graph_projection(tx):
    result = tx.run(
        """
    CALL gds.graph.exists('articleGraph')
    YIELD exists
    RETURN exists
    """
    )
    if result.single()["exists"]:
        tx.run("CALL gds.graph.drop('articleGraph')")


def create_graph_proj(tx):
    # logging.info("Create projection")
    tx.run(
        """
           CALL gds.graph.project(
            'articleGraph',
            'Article',
            {
                SIMILAR: {
                    properties: 'similarity'
                }
            }
           )
    """
    )


def detect_community(tx):
    # logging.info("Detect community")
    tx.run(
        """
        CALL gds.louvain.write(
        'articleGraph',
        {
            writeProperty: 'community'
        }
        )
    """
    )


def return_pred_cluster(tx):
    query = """
        MATCH (a:Article)
        RETURN a.id AS id,
            a.title AS title,
            a.url AS url,
            a.content AS body_content,
            a.community AS cluster
        ORDER BY a.community
        """
    result = tx.run(query)
    df = pd.DataFrame(result.data())
    return df


def get_cluster_size(pred_cluster, column_name="cluster"):
    grouped_counts = pred_cluster.groupby(column_name).size()
    filtered_grouped_counts = grouped_counts[grouped_counts != 1]
    single_nodes = len(grouped_counts[grouped_counts == 1])
    bins = range(1, filtered_grouped_counts.max() + 5, 5)
    labels = [f"{i}-{i+4}" for i in bins[:-1]]
    labels[0] = "2-5"
    binned_counts = pd.cut(
        filtered_grouped_counts, bins=bins, labels=labels, right=False
    )
    banded_counts = binned_counts.value_counts().sort_index()
    cluster_size = (
        pd.DataFrame(banded_counts)
        .reset_index()
        .rename(columns={"index": "Cluster size", "count": "Num of clusters"})
    )
    new_row = {
        "Cluster size": "1",
        "Num of clusters": single_nodes,
    }  # Customize with your data
    cluster_size.loc[-1] = new_row
    cluster_size = cluster_size.sort_index().reset_index(drop=True)
    return cluster_size


def generate_cluster_keywords(pred_cluster):
    docs = pd.DataFrame(
        {"Document": pred_cluster.body_content, "Class": pred_cluster.new_cluster}
    )
    docs_per_class = docs.groupby(["Class"], as_index=False).agg({"Document": " ".join})
    docs_per_class["Document"] = docs_per_class["Document"].apply(
        lambda text: " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    )

    count_vectorizer = CountVectorizer(stop_words=stopwords).fit(
        docs_per_class.Document
    )
    count = count_vectorizer.transform(docs_per_class.Document)
    words = count_vectorizer.get_feature_names_out()
    ctfidf = CTFIDFVectorizer().fit_transform(count, n_samples=len(docs)).toarray()

    cluster_keywords_dict = {}
    for idx, cluster in enumerate(docs_per_class.Class):
        top_indices = ctfidf[idx].argsort()[-5:][
            ::-1
        ]  # Get top 5 indices, sorted in descending order
        top_words = [words[index] for index in top_indices]
        cluster_keywords_dict[cluster] = top_words

    return cluster_keywords_dict


def get_clustered_nodes(tx):
    query = """
        MATCH (n)-[r]->(m)
        RETURN n.id AS node_1_id,
            m.id AS node_2_id,
            n.title AS node_1_title,
            m.title AS node_2_title,
            r.similarity AS edge_weight,
            n.ground_truth AS node_1_ground_truth,
            m.ground_truth AS node_2_ground_truth,
            n.community AS node_1_pred_cluster,
            m.community AS node_2_pred_cluster
        """
    result = tx.run(query)
    df = pd.DataFrame(result.data())
    return df


def get_unclustered_nodes(tx):
    query = """
        MATCH (n)
        WHERE NOT EXISTS ((n)--())
        RETURN n.title AS node_title,
            n.ground_truth AS node_ground_truth,
            n.community AS node_community,
            n.meta_desc AS node_meta_desc
        """
    result = tx.run(query)
    df = pd.DataFrame(result.data())
    return df


def count_articles(tx):
    query = """
        MATCH (a:Article)
        RETURN a.community AS cluster, count(a) AS article_count
        ORDER BY cluster
        """
    result = tx.run(query)
    df = pd.DataFrame(result.data())
    return df


def return_by_cluster(tx):
    """Return only clusters with more than one article"""

    query = """
    MATCH (n)
    WITH n.community AS cluster, collect(n.title) AS titles, count(n) AS count
    WHERE count > 1
    RETURN cluster, titles
    ORDER BY cluster
        """
    result = tx.run(query)
    df = pd.DataFrame(result.data())
    return df


def get_embeddings(cluster_df, umap_parameters):
    embeddings = np.array(cluster_df.extracted_content_body_embeddings.to_list())
    doc_titles = cluster_df.title.to_list()
    docs = cluster_df.body_content.to_list()
    ids = cluster_df.id.to_list()
    # umap_model = UMAP(n_neighbors=15, n_components=8, min_dist=0.0, metric='cosine', random_state=42)
    umap_model = UMAP(
        n_neighbors=umap_parameters["n_neighbors"],
        n_components=umap_parameters["n_components"],
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    umap_embeddings = umap_model.fit_transform(embeddings)

    return embeddings, doc_titles, docs, ids, umap_embeddings


def hyperparameter_tuning(embeddings):
    best_score = 0

    for min_cluster_size in [2, 3, 4, 5, 6]:
        for min_samples in [1, 2, 3, 4, 5, 6, 7]:
            for cluster_selection_method in ["leaf"]:  # can use 'eom' too
                for metric in ["euclidean", "manhattan"]:
                    # for each combination of parameters of hdbscan
                    hdb = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_method=cluster_selection_method,
                        metric=metric,
                        gen_min_span_tree=True,
                    ).fit(embeddings)
                    # DBCV score
                    score = hdb.relative_validity_
                    if score > best_score:
                        best_score = score
                        best_parameters = {
                            "min_cluster_size": min_cluster_size,
                            "min_samples": min_samples,
                            "cluster_selection_method": cluster_selection_method,
                            "metric": metric,
                        }

    print(f"Best DBCV score: {best_score:.3f}")
    print(f"Best parameters: {best_parameters}")
    return best_parameters


def topic_modelling(hyperparameters):
    np.random.seed(42)
    # Step 3 - Cluster reduced embeddings
    hdbscan_model = HDBSCAN(
        min_cluster_size=hyperparameters["min_cluster_size"],
        min_samples=hyperparameters["min_samples"],
        metric=hyperparameters["metric"],
        cluster_selection_method=hyperparameters["cluster_selection_method"],
        prediction_data=True,
        gen_min_span_tree=True,
    )

    # Step 4 - Tokenize topics
    vectorizer_model = CountVectorizer(stop_words="english")

    # Step 5 - Create topic representation
    ctfidf_model = ClassTfidfTransformer()

    # Step 6 - (Optional) Fine-tune topic representations with
    representation_model = MaximalMarginalRelevance(diversity=0.3)

    # All steps together
    topic_model = BERTopic(
        # embedding_model=embedding_model,          # Step 1 - Extract embeddings
        # umap_model=umap_model,                    # Step 2 - Reduce dimensionality
        hdbscan_model=hdbscan_model,  # Step 3 - Cluster reduced embeddings
        vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
        ctfidf_model=ctfidf_model,  # Step 5 - Extract topic words
        representation_model=representation_model,  # Step 6 - (Optional) Fine-tune topic represenations
        # nr_topics="auto" #default is none, will auto reduce topics using HDBSCAN
    )
    return topic_model


def create_topic_assigner(start_counter):
    counter = start_counter

    def assign_new_topic(x):
        nonlocal counter
        if x == -1:
            new_topic = counter
            counter += 1
            return new_topic
        else:
            return x

    return assign_new_topic


def process_cluster(cluster_df, umap_parameters):
    # Step 1: Extract embeddings and umap_embeddings
    embeddings, doc_titles, docs, ids, umap_embeddings = get_embeddings(
        cluster_df, umap_parameters
    )

    # Step 2: Perform hyperparameter tuning for berttopic
    hyperparameters = hyperparameter_tuning(umap_embeddings)

    # Step 3: Create and fit topic model
    topic_model = topic_modelling(hyperparameters)
    topics, _ = topic_model.fit_transform(docs, embeddings)

    ###############
    # Visualisation
    ################

    # Uncomment and adjust as needed for visualization purposes

    # top_n = 50
    # top_topics = topic_model.get_topic_freq().head(top_n)['Topic'].tolist()

    # reduced_embeddings = topic_model.umap_model.embedding_
    # hover_data = [f"{title} - Topic {topic}" for title, topic in zip(doc_titles, topics)]
    # visualization = topic_model.visualize_documents(hover_data, reduced_embeddings=reduced_embeddings, topics=top_topics, title=f'Top {top_n} Topics')
    # visualization.show()

    # visualization_barchart = topic_model.visualize_barchart(top_n_topics=top_n)
    # visualization_barchart.show()
    #################

    # Step 4: Create a DataFrame with assigned topics, titles and ids.
    result_df = pd.DataFrame({"Assigned Topic": topics, "Title": doc_titles, "id": ids})

    # Step 5: Extract topic information and get top 5 keywords, if article is unclustered where Topic is -1, topic representation/kws will be removed
    topic_kws = topic_model.get_topic_info()[["Topic", "Representation"]]
    topic_kws["top_5_kws"] = topic_kws.apply(
        lambda row: row["Representation"][:5] if row["Topic"] != -1 else np.nan, axis=1
    )

    # Step 6: Merge results with the top keywords
    result_df_kws = pd.merge(
        result_df, topic_kws, how="left", left_on="Assigned Topic", right_on="Topic"
    )
    result_df_kws = result_df_kws.drop(["Representation", "Topic"], axis=1)
    result_df_kws = result_df_kws[["id", "Title", "Assigned Topic", "top_5_kws"]]

    # Step 7: Assign new topic numbers to topics that are -1, starting from the max assigned topic in the results_df_kws.
    max_topic = result_df_kws["Assigned Topic"].max()
    new_topic_counter = max_topic + 1
    assign_new_topic_func = create_topic_assigner(new_topic_counter)
    result_df_kws["Assigned Topic"] = result_df_kws["Assigned Topic"].apply(
        assign_new_topic_func
    )

    # Step 8: Update the 'Assigned Topic' column with cluster information to prevent repeat cluster numbers
    cluster_id = cluster_df["cluster"].unique()[0]
    result_df_kws["Assigned Topic"] = result_df_kws["Assigned Topic"].apply(
        lambda x: "Cluster_" + str(cluster_id) + "_" + str(x)
    )

    return result_df_kws


def process_all_clusters(cluster_morethan10_embeddings, umap_parameters):
    unique_clusters = cluster_morethan10_embeddings["cluster"].unique()
    all_results = []

    for cluster_id in unique_clusters:
        print(f"cluster id: {cluster_id}")
        cluster_df = cluster_morethan10_embeddings[
            cluster_morethan10_embeddings["cluster"] == cluster_id
        ]
        result_df_kws = process_cluster(cluster_df, umap_parameters)
        all_results.append(result_df_kws)

    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df


def assign_unique_numbers_to_topics(final_result_df, pred_cluster_df):
    """
    Assigns unique numbers to each unique 'Assigned Topic' in the final_result_df
    based on the maximum cluster value from the pred_cluster_df.

    Parameters:
    final_result_df (pd.DataFrame): DataFrame containing the final results with an 'Assigned Topic' column.
    pred_cluster_df (pd.DataFrame): DataFrame containing the predicted clusters with a 'cluster' column.

    Returns:
    pd.DataFrame: Updated final_result_df with an additional 'Assigned Topic Number' column.
    """
    max_cluster_value = pred_cluster_df["cluster"].max()
    unique_assigned_topics = final_result_df["Assigned Topic"].unique()
    topic_number_mapping = {
        topic: idx + max_cluster_value + 1
        for idx, topic in enumerate(unique_assigned_topics)
    }

    final_result_df["Assigned Topic Number"] = final_result_df["Assigned Topic"].map(
        topic_number_mapping
    )
    return final_result_df
