from typing import Annotated, TypedDict

import pandas as pd
from agents.enums import ROLES
from agents.models import start_llm
from config import settings
from langgraph.graph import END, START
from states.definitions import (
    ArticleInputs,
    ChecksAgents,
    ContentFlags,
    ContentJudge,
    MetaFlags,
    MetaJudge,
    TitleFlags,
    TitleJudge,
)
from utils.evaluations import calculate_readability
from utils.graphs import create_graph, draw_graph, execute_graph
from utils.paths import get_root_dir
from utils.reducers import merge_dict

# Declaring maximum new tokens
MAX_NEW_TOKENS = settings.MAX_NEW_TOKENS

# Declaring model to use
MODEL = settings.MODEL_NAME


class ChecksState(TypedDict):
    """This class contains the different keys relevant to the project. It inherits from the TypedDict class.

    Attributes:
        article_inputs:
        content_flags:
        title_flags:
        meta_flags:
        content_judge:
        title_judge:
        meta_judge:
        llm_agents:
    """

    # Article Inputs
    article_inputs: ArticleInputs

    # Flags for rule-based/statistical-based methods
    content_flags: Annotated[ContentFlags, merge_dict]
    title_flags: Annotated[TitleFlags, merge_dict]
    meta_flags: Annotated[MetaFlags, merge_dict]

    # Explanations for various criteria
    content_judge: Annotated[ContentJudge, merge_dict]
    title_judge: Annotated[TitleJudge, merge_dict]
    meta_judge: Annotated[MetaJudge, merge_dict]

    # Agents
    llm_agents: ChecksAgents


def content_evaluation_rules_node(state: ChecksState) -> dict:
    article_content = state.get("article_inputs")["article_content"]
    content_category = state.get("article_inputs")["content_category"]
    content_flags = state.get("content_flags", {})

    # Check readability -
    # Hemmingway Metric
    metric = calculate_readability(article_content, "hemmingway")
    score = metric.get("score", -1)

    # NOTE: Exceptions must be handled to prevent the program from being prematurely terminated
    if score <= 0:
        raise ValueError("The readability score must be greater than 0.")
    elif score >= 10:
        content_flags["is_unreadable"] = True
    else:
        content_flags["is_unreadable"] = False

    content_flags["readability_score"] = score

    # Check for insufficient content -
    # Less than 300 - 400 words is considered too brief
    # to adequately cover a topic unless it's a very specific and narrow subject
    # Threshold is set at the 25th percentile - Flag articles below 25th percentile for each content category
    word_count_threshold_dict = {
        "cost-and-financing": 267,
        "live-healthy-articles": 413,
        "diseases-and-conditions": 368,
        "medical-care-and-facilities": 202,
        "support-group-and-others": 213,
    }
    word_count = len(article_content.split())
    # Flag articles based on the provided threshold. Otherwise, flag articles below 300 words
    threshold = word_count_threshold_dict.get(content_category, 300)
    if word_count < threshold:
        content_flags["low_word_count"] = True
    else:
        content_flags["low_word_count"] = False

    return {"content_flags": content_flags}


def content_explanation_node(state: ChecksState) -> dict:
    article_content = state.get("article_inputs")["article_content"]
    content_flags = state.get("content_flags", {})
    content_judge = state.get("content_judge", {})

    # Explain Poor Readability based on Readability Matrix - Low Priority
    readability = content_flags.get("is_unreadable", False)

    if readability:
        explanation_agent = state.get("llm_agents")["explanation_agent"]
        readability_explanation = explanation_agent.evaluate_content(
            article_content, choice="readability"
        )
        content_judge["readability"] = readability_explanation

    return {"content_judge": content_judge}


def content_evaluation_llm_node(state: ChecksState) -> dict:
    article_content = state.get("article_inputs")["article_content"]
    content_judge = state.get("content_judge", {})
    content_evaluation_agent = state.get("llm_agents")["evaluation_agent"]

    # Check poor content structure -
    # No clear sections
    # No introduction or conclusion
    # Absence of headings or subheadings
    content_structure_eval = content_evaluation_agent.evaluate_content(
        article_content, choice="structure"
    )
    content_judge["structure"] = content_structure_eval

    return {"content_judge": content_judge}


def title_evaluation_rules_node(state: ChecksState) -> dict:
    article_title = state.get("article_inputs")["article_title"]
    title_flags = state.get("title_flags", {})

    # Not Within Page Title Char Count - Up to 70 Characters
    # NOTE: Exceptions must be handled to prevent the program from being prematurely terminated
    # TODO: Check whether an error is raised for all articles - Stress test
    char_count = len(article_title)
    if char_count <= 0:
        raise ValueError("The title character count must be greater than 0.")
    elif char_count > 70:
        title_flags["long_title"] = True
    else:
        title_flags["long_title"] = False

    return {"title_flags": title_flags}


def title_evaluation_llm_node(state: ChecksState) -> dict:
    article_title = state.get("article_inputs")["article_title"]
    article_content = state.get("article_inputs")["article_content"]
    title_judge = state.get("title_judge", {})

    # Irrelevant Page Title
    title_evaluation_agent = state.get("llm_agents")["evaluation_agent"]
    title_evaluation = title_evaluation_agent.evaluate_title(
        article_title, article_content
    )
    title_judge["title"] = title_evaluation

    return {"title_judge": title_judge}


def meta_desc_evaluation_rules_node(state: ChecksState) -> dict:
    meta_desc = state.get("article_inputs")["meta_desc"]
    meta_flags = state.get("meta_flags", {})

    # Not Within Meta Description Char Count - Between 70 and 160 characters
    # NOTE: Exceptions must be handled to prevent the program from being prematurely terminated
    char_count = len(meta_desc)
    if char_count <= 0:
        raise ValueError("The meta description character count must be greater than 0.")
    elif 70 <= char_count <= 160:
        meta_flags["not_within_char_count"] = False
    else:
        meta_flags["not_within_char_count"] = True

    return {"meta_flags": meta_flags}


def meta_desc_evaluation_llm_node(state: ChecksState) -> dict:
    meta_desc = state.get("article_inputs")["meta_desc"]
    article_content = state.get("article_inputs")["article_content"]
    meta_judge = state.get("meta_judge", {})

    # Irrelevant Meta Description
    meta_evaluation_agent = state.get("llm_agents")["evaluation_agent"]
    meta_evaluation = meta_evaluation_agent.evaluate_meta_description(
        meta_desc, article_content
    )
    meta_judge["meta_desc"] = meta_evaluation

    return {"meta_judge": meta_judge}


if __name__ == "__main__":
    ROOT_DIR = get_root_dir()

    nodes = {
        "content_evaluation_rules_node": content_evaluation_rules_node,
        "content_explanation_node": content_explanation_node,
        "content_evaluation_llm_node": content_evaluation_llm_node,
        "title_evaluation_rules_node": title_evaluation_rules_node,
        "title_evaluation_llm_node": title_evaluation_llm_node,
        "meta_desc_evaluation_rules_node": meta_desc_evaluation_rules_node,
        "meta_desc_evaluation_llm_node": meta_desc_evaluation_llm_node,
    }

    edges = {
        START: [
            "content_evaluation_rules_node",
            "title_evaluation_rules_node",
            "meta_desc_evaluation_rules_node",
        ],
        "content_evaluation_rules_node": ["content_explanation_node"],
        "content_explanation_node": ["content_evaluation_llm_node"],
        "content_evaluation_llm_node": [END],
        "title_evaluation_rules_node": ["title_evaluation_llm_node"],
        "title_evaluation_llm_node": [END],
        "meta_desc_evaluation_rules_node": ["meta_desc_evaluation_llm_node"],
        "meta_desc_evaluation_llm_node": [END],
    }

    app = create_graph(ChecksState, nodes, edges)

    # Save Graph
    draw_graph(
        app,
        f"{ROOT_DIR}/article-harmonisation/docs/images/optimisation_checks_flow.png",
    )

    # Start LLM
    evaluation_agent = start_llm(MODEL, ROLES.EVALUATOR)
    explanation_agent = start_llm(MODEL, ROLES.EXPLAINER)

    # Load data from merged_data.parquet and randomly sample 30 rows
    df = pd.read_parquet(f"{ROOT_DIR}/article-harmonisation/data/merged_data.parquet")
    df_keep = df[~df["to_remove"]]
    df_sample = df_keep.sample(n=1, replace=False, random_state=42)
    rows = df_sample.shape[0]
    print(rows)

    records = []

    for i in range(rows):
        # Set up
        article_id = df_sample["id"].iloc[i]
        article_content = df_sample["extracted_content_body"].iloc[i]
        article_title = df_sample["title"].iloc[i]
        meta_desc = df_sample["category_description"].iloc[i]  # meta_desc can be null
        meta_desc = meta_desc if meta_desc is not None else "No meta description"
        article_url = df_sample["full_url"].iloc[i]
        content_category = df_sample["content_category"].iloc[i]
        article_category_names = df_sample["article_category_names"].iloc[i]
        page_views = df_sample["page_views"].iloc[i]

        print(f"Checking {article_title} now...")
        # Set up Inputs
        inputs = {
            "article_inputs": {
                "article_id": article_id,
                "article_content": article_content,
                "article_title": article_title,
                "meta_desc": meta_desc,
                "article_url": article_url,
                "content_category": content_category,
                "article_category_names": article_category_names,
                "page_views": page_views,
            },
            "content_flags": {},
            "title_flags": {},
            "meta_flags": {},
            "content_judge": {},
            "title_judge": {},
            "meta_judge": {},
            "llm_agents": {
                "evaluation_agent": evaluation_agent,
                "explanation_agent": explanation_agent,
            },
        }

        response = execute_graph(app, inputs)
        print(response)
        del response["llm_agents"]
        records.append(response)

    df_save = pd.DataFrame.from_records(records)
    df_save.to_parquet(
        f"{ROOT_DIR}/article-harmonisation/data/optimization_checks/agentic_response.parquet"
    )
