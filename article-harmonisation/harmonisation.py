from typing import TypedDict

from agents.enums import ROLES
from agents.models import start_llm
from checks import ChecksState
from config import settings
from langgraph.graph import END, START
from states.definitions import (
    ArticleInputs,
    OptimisationAgents,
    OptimisationFlags,
    OptimisedArticle,
)
from utils.evaluations import calculate_readability
from utils.formatters import concat_headers_to_content, print_checks
from utils.graphs import create_graph, draw_graph, execute_graph
from utils.paths import get_root_dir

# Declaring maximum new tokens
MAX_NEW_TOKENS = settings.MAX_NEW_TOKENS

# Declaring model to use
MODEL = settings.MODEL_NAME

# Declaring the number of retries
REWRITING_TRIES = 3


class RewritingState(TypedDict):
    """This class contains the different keys relevant to the project. It inherits from the TypedDict class.

    Attributes:
        article_content: A required list where each element is a String containing the article content.
        article_title: An optional String which will contain the article title.
        meta_desc: An optional String that will contain article's meta description.
        keypoints: An optional list where each element is a String with the article keypoints.
        compiled_keypoints: An optional String containing keypoints compiled from the attribute keypoints.
        optimised_content: An optional String containing the final optimised content.
        article_researcher_counter: An optional integer serving as a counter for number of articles processed by the researcher node.
        previous_node: An optional String that will store the name of the most recently visited node by the StateGraph.
        flag_for_content_optimisation: A required boolean value, determining if the user has flagged the article for content optimisation.
        flag_for_title_optimisation: A required boolean value, determining if the user has flagged the article for title optimisation.
        flag_for_meta_desc_optimisation: A required boolean value, determining if the user has flagged the article for meta description optimisation.
    """

    article_rewriting_tries: int
    article_evaluation: ChecksState
    original_article_inputs: ArticleInputs
    optimised_article_output: OptimisedArticle
    user_flags: OptimisationFlags
    llm_agents: OptimisationAgents


# Functions defining the functionality of different nodes
def researcher_node(state):
    """Creates a researcher node, which will categorise the sentences in the article into keypoints.

    Args:
        state: a dictionary storing relevant attributes as keys and content as the respective items.

    Returns:
        a dictionary with keys "keypoints", "article_researcher_counter"
            - keypoints: an updated list storing keypoints from all articles output from the researcher node
            - article_researcher_counter: an integer serving as a counter for number of articles processed by the researcher node
    """
    article_list = state.get("original_article_inputs")["article_content"]
    keypoints = state.get("optimised_article_output")["researcher_keypoints"]
    researcher_agent = state.get("llm_agents")["researcher_agent"]

    # For loop iterating through each article

    for idx in range(len(article_list)):
        print("Processing keypoints for article", idx + 1)
        article = article_list[idx]
        article_keypoints = researcher_agent.generate_keypoints(article)
        keypoints.append(article_keypoints)
        print("Finished processing keypoints for article", idx + 1)

    return {
        "optimised_article_output": {
            "researcher_keypoints": keypoints,
        }
    }


def compiler_node(state):
    """Creates a compiler node, which will compile the keypoints from a list of keypoints

    Args:
        state:  a dictionary storing relevant attributes as keys and content as the respective items.

    Returns:
        a dictionary with keys "compiled_keypoints"
            - compiled_keypoints: a String containing the compiled keypoints from the compiler LLM
    """
    optimised_article_output = state.get("optimised_article_output")
    keypoints = optimised_article_output["researcher_keypoints"]

    # Runs the compiler LLM to compile the keypoints
    compiler_agent = state.get("llm_agents")["compiler_agent"]

    compiled_keypoints = compiler_agent.compile_points(keypoints)
    optimised_article_output["compiled_keypoints"] = compiled_keypoints

    return {"optimised_article_output": optimised_article_output}


def meta_description_optimisation_node(state):
    """Creates a meta description optimisation node, which will optimise the meta description of the article

    Args:
        state: a dictionary storing relevant attributes as keys and content as the respective items.

    Returns:
        a dictionary with keys "meta_desc",  "flag_for_meta_desc_optimisation"
            - meta_desc: a String containing the optimised meta description from the meta description optimisation LLM
            - flag_for_meta_desc_optimisation: a False boolean value to indicate that the meta description optimisation step has been completed
    """
    optimised_article_output = state.get("optimised_article_output")

    content = optimised_article_output["researcher_keypoints"]
    if "optimised_writing" in optimised_article_output.keys():
        content = optimised_article_output["optimised_writing"]

    meta_desc_optimisation_agent = state.get("llm_agents")[
        "meta_desc_optimisation_agent"
    ]
    optimised_meta_desc = meta_desc_optimisation_agent.optimise_meta_desc(content)

    optimised_article_output["optimised_meta_desc"] = optimised_meta_desc

    user_flags = state.get("user_flags")
    user_flags["flag_for_meta_desc_optimisation"] = False

    return {
        "optimised_article_output": optimised_article_output,
        "user_flags": user_flags,
    }


def title_optimisation_node(state):
    """Creates a title optimisation node, which will optimise the title of the article

    Args:
        state: a dictionary storing relevant attributes as keys and content as the respective items.

    Returns:
        a dictionary with keys "article_title",  "flag_for_title_optimisation"
            - article_title: a String containing the optimised title from the title optimisation LLM
            - flag_for_title_optimisation: a False boolean value to indicate that the title optimisation step has been completed
    """
    optimised_article_output = state.get("optimised_article_output")

    content = optimised_article_output["researcher_keypoints"]
    if "optimised_writing" in optimised_article_output.keys():
        content = optimised_article_output["optimised_writing"]

    title_optimisation_agent = state.get("llm_agents")["title_optimisation_agent"]
    optimised_article_title = title_optimisation_agent.optimise_title(content)

    optimised_article_output["optimised_article_title"] = optimised_article_title

    user_flags = state.get("user_flags")
    user_flags["flag_for_title_optimisation"] = False

    return {
        "optimised_article_output": optimised_article_output,
        "user_flags": user_flags,
    }


def content_guidelines_optimisation_node(state):
    """Creates a content guidelines node, which will rewrite missing content of the article based on the article topic's requirements state in the content playbook.

        Requirements for each topic can be found in the content playbook under:
            1. Chapter 3, Step 1, (iii) writing guide
    Args:
        state: a dictionary storing relevant attributes as keys and content as the respective items.

    Returns:
        a dictionary with keys "optimised_content"
            - optimised_content: a String containing the rewritten article from the content guidelines LLM
    """
    optimised_article_output = state.get("optimised_article_output")

    keypoints = optimised_article_output["researcher_keypoints"]
    if "compiled_keypoints" in optimised_article_output.keys():
        keypoints = optimised_article_output["compiled_keypoints"]

    content_evaluation = (
        state.get("article_evaluation").get("content_judge", {}).get("structure", "")
    )

    # Runs the compiler LLM to compile the keypoints
    content_optimisation_agent = state.get("llm_agents")["content_optimisation_agent"]

    optimised_content = content_optimisation_agent.optimise_content(
        keypoints, content_evaluation
    )

    optimised_article_output["optimised_content"] = optimised_content

    user_flags = state.get("user_flags")
    user_flags["flag_for_content_optimisation"] = False

    return {
        "optimised_article_output": optimised_article_output,
        "user_flags": user_flags,
    }


def writing_guidelines_optimisation_node(state):
    """Creates a writing optimisation node, which will optimise the article content based on the writing guidelines from the content playbook

        Content covered from the playbook include:
            1. Chapter 1, Section A, Guidelines + Examples

    Args:
        state: a dictionary storing relevant attributes as keys and content as the respective items.

    Returns:
        a dictionary with keys "optimised_content",  "flag_for_content_optimisation"
            - optimised_content: a String containing the optimised content from the writing optimisation LLM
            - flag_for_content_optimisation: a False boolean value to indicate that the writing optimisation step has been completed
    """
    optimised_article_output = state.get("optimised_article_output")

    content = optimised_article_output.get(
        "optimised_writing",  # article optimisation, subsequent rewrites
        optimised_article_output.get(
            "optimised_content",  # article optimisation, after content optimisation and first rewriting
            optimised_article_output.get(
                "compiled_keypoints",  # article harmonisation, without any content optimisation
                optimised_article_output.get(
                    "researcher_keypoints"
                ),  # article optimisation, without any content optimisation
            ),
        ),
    )

    writing_optimisation_agent = state.get("llm_agents")["writing_optimisation_agent"]
    optimised_writing = writing_optimisation_agent.optimise_writing(content)

    user_flags = state.get("user_flags")
    user_flags["flag_for_writing_optimisation"] = False

    # Updating the new optimised writing in the state
    optimised_article_output["optimised_writing"] = optimised_writing

    # Updating the readability score in the state
    new_readability_score = calculate_readability(optimised_writing)["score"]
    optimised_article_output["readability_score"] = new_readability_score

    return {
        "optimised_article_output": optimised_article_output,
        "user_flags": user_flags,
    }


def readability_evaluation_node(state):
    """Creates a readability evaluation node, used to provide feedback on an already optimised article to determine if the optimisation is sufficient good. This serves as a process of feedback loop for the writing optimisation process.

    Args:
        state: a dictionary storing relevant attributes as keys and content as the respective items.

    Returns:
    """
    # Getting the optimised writing from writing_guidelines_optimisation_node
    optimised_article_output = state.get("optimised_article_output")
    optimised_writing = optimised_article_output.get("optimised_writing")

    readability_evaluation_agent = state.get("llm_agents")[
        "readability_evaluation_agent"
    ]
    print("Evaluating article writing")
    readability_explanation = readability_evaluation_agent.evaluate_content(
        content=optimised_writing, choice="readability"
    )
    print("Article writing evaluated")
    article_evaluation = state.get("article_evaluation")
    content_judge = article_evaluation.get("content_judge", {})

    # Updating content_judge with the new readability evaluation
    content_judge["readability"] = readability_explanation

    # Updating state with the most recent content_judge updated with the new readability evaluation
    article_evaluation["content_judge"] = content_judge

    rewriting_tries = state.get("article_rewriting_tries", 0)

    return {
        "article_rewriting_tries": rewriting_tries + 1,
        "article_evaluation": article_evaluation,
        "optimised_article_output": optimised_article_output,
    }


# TODO: finish this
def readability_optimisation_node(state):
    """Creates a readability optimisation node that optimises the writing based on the feedback given."""
    # Obtaining the optimised writing
    optimised_article_output = state.get("optimised_article_output")
    optimised_writing = optimised_article_output.get("optimised_writing")

    readability_evaluation = (
        state.get("article_evaluation").get("content_judge", {}).get("readability", "")
    )

    readability_optimisation_agent = state.get("llm_agents")[
        "readability_optimisation_agent"
    ]

    print("Optimising article readability")
    optimised_readability_article = readability_optimisation_agent.optimise_readability(
        content=optimised_writing, readability_evaluation=readability_evaluation
    )
    print("Article readability optimised")

    # Updating state with the newly readability optimised article
    optimised_article_output["optimised_writing"] = optimised_readability_article

    return {
        "optimised_article_output": optimised_article_output,
    }


def personality_guidelines_evaluation_node(state):
    """Creates a personality guidelines evaluation node that evaluates the given content based on the voice and personality guidelines from HH content playbook

    Args:
        state: a dictionary storing relevant attributes as keys and content as the respective items.

    Returns:
        - "True"
        - "False"
    """
    # Obtaining the recently readability optimised writing
    optimised_article_output = state.get("optimised_article_output")
    optimised_writing = optimised_article_output.get("optimised_writing")

    personality_evaluation_agent = state.get("llm_agents")[
        "personality_evaluation_agent"
    ]

    print("Evaluating personality of article")
    personality_evaluation = personality_evaluation_agent.evaluate_personality(
        content=optimised_writing
    )
    print("Article personality evaluated")

    # Obtaining article_flags dict
    article_evaluation = state.get("article_evaluation")
    content_flags = article_evaluation.get("content_flags", {})

    # Updating article_evaluation with the updated personality_evaluation flags
    content_flags["has_personality"] = personality_evaluation
    article_evaluation["content_flags"] = content_flags

    return {
        "article_evaluation": article_evaluation,
    }


# functions to determine next node
def check_for_compiler(state):
    """Determines if the state should move on to the compiler node for harmonisation or content_guidelines for optimisation.

    Args:
        state: a dictionary storing relevant attributes as keys and content as the respective items.

    Returns:
        "researcher_node": returned if counter < number of articles to be harmonised
        "compiler_node": returned if counter >= number of articles to be harmonised
    """

    article_content = state.get("original_article_inputs")["article_content"]
    content_optimisation_flag = state.get("user_flags")["flag_for_content_optimisation"]
    writing_optimisation_flag = state.get("user_flags")["flag_for_writing_optimisation"]
    title_optimisation_flag = state.get("user_flags")["flag_for_title_optimisation"]
    meta_desc_optimisation_flag = state.get("user_flags")[
        "flag_for_meta_desc_optimisation"
    ]

    if len(article_content) < 2:
        if content_optimisation_flag:
            return "content_guidelines_optimisation_node"
        elif writing_optimisation_flag:
            return "writing_guidelines_optimisation_node"
        elif title_optimisation_flag:
            return "title_optimisation_node"
        elif meta_desc_optimisation_flag:
            return "meta_description_optimisation_node"
        else:
            return END
    else:
        return "compiler_node"


def decide_next_optimisation_node(state):
    """Checks user flags for content optimisation, title optimisation and meta description in state, and returning the appropriate node depending on the stored boolean values of each flag.

    Args:
        state: a dictionary storing relevant attributes as keys and content as the respective items.

    Returns:
        "content_guidelines_node": returned if flag_for_content_optimisation is True
        "title_optimisation_node": returned if previous flag is False and flag_for_title_optimisation is True
        "meta_description_node": returned if previous flags are False and flag_for_meta_desc_optimisation is True
        END: returned if all flags are False, as no further optimisation is required for the article
    """
    content_optimisation_flag = state.get("user_flags")["flag_for_content_optimisation"]
    writing_optimisation_flag = state.get("user_flags")["flag_for_writing_optimisation"]
    title_optimisation_flag = state.get("user_flags")["flag_for_title_optimisation"]
    meta_desc_optimisation_flag = state.get("user_flags")[
        "flag_for_meta_desc_optimisation"
    ]

    if content_optimisation_flag:
        return "content_guidelines_optimisation_node"
    elif writing_optimisation_flag:
        return "writing_guidelines_optimisation_node"
    elif title_optimisation_flag:
        return "title_optimisation_node"
    elif meta_desc_optimisation_flag:
        return "meta_description_optimisation_node"
    else:
        return END


def check_personality_after_readability_optimisation(state):
    """Checks for the personality evaluation from the personality_evaluation_node and determine if a subsequent round of writing optimisation is required."""

    # Extracting has_personality from state
    content_flags = state.get("article_evaluation").get("content_flags")
    has_personality = content_flags["has_personality"]

    if has_personality:
        return "readability_evaluation_node"
    else:
        return "writing_guidelines_optimisation_node"


def check_readability_after_writing_optimisation(state):
    """Checks for the readability score of the writing optimised article and determines if a subsequent round of rewriting is required to improve the article readability score."""

    # Extracting the new readability score from state
    optimised_article_output = state.get("optimised_article_output")
    new_readability_score = optimised_article_output["readability_score"]

    # Extractung the number of article rewriting tries
    rewriting_tries = state.get("article_rewriting_tries", 0)

    print(
        f"Number of retries: {rewriting_tries}, Readability score: {new_readability_score}"
    )

    if rewriting_tries > REWRITING_TRIES or new_readability_score < 10:
        # Respective print statements for exiting loop due to maximum number of rewritting tries or readability score
        if rewriting_tries >= REWRITING_TRIES:
            print("Number of writing retries exceeded limit hit")
        elif new_readability_score < 10:
            print(
                f"Readability score is now {new_readability_score} and considered readable"
            )
        # Checks if article is flagged for title and meta_desc optimisation
        title_optimisation_flags = state.get("user_flags")[
            "flag_for_title_optimisation"
        ]
        meta_desc_optimisation_flags = state.get("user_flags")[
            "flag_for_meta_desc_optimisation"
        ]

        if title_optimisation_flags:
            return "title_optimisation_node"
        elif meta_desc_optimisation_flags:
            return "meta_description_optimisation_node"
        else:
            return END

    else:

        # Checks if the current readability score is >= 10
        if new_readability_score >= 10:
            return "readability_evaluation_node"

        # checks if the current readability score is worse than original readability score after writing optimisation
        content_flags = state.get("article_evaluation").get("content_flags", {})
        original_readability_score = content_flags.get("readability_score", None)

        if (
            original_readability_score
        ):  # this means that this is an individual article input and it's for optimisation
            if new_readability_score >= original_readability_score:
                return "readability_evaluation_node"


if __name__ == "__main__":
    # Gets root directory
    ROOT_DIR = get_root_dir()

    # Declaring dictionary with all nodes
    nodes = {
        "researcher_node": researcher_node,
        "compiler_node": compiler_node,
        "content_guidelines_optimisation_node": content_guidelines_optimisation_node,
        "writing_guidelines_optimisation_node": writing_guidelines_optimisation_node,
        "readability_evaluation_node": readability_evaluation_node,
        "title_optimisation_node": title_optimisation_node,
        "meta_description_optimisation_node": meta_description_optimisation_node,
        "personality_guidelines_evaluation_node": personality_guidelines_evaluation_node,
        "readability_optimisation_node": readability_optimisation_node,
    }

    # Declaring dictionary with all edges
    edges = {
        START: ["researcher_node"],
        "readability_evaluation_node": ["readability_optimisation_node"],
        "readability_optimisation_node": ["personality_guidelines_evaluation_node"],
        "meta_description_optimisation_node": [END],
    }

    # Declaring dictionary with all conditional edges
    # Example element in conditional edge dictionary: {"name of node": (conditional edge function, path map)}
    conditional_edges = {
        "researcher_node": (
            check_for_compiler,
            {
                "compiler_node": "compiler_node",
                "content_guidelines_optimisation_node": "content_guidelines_optimisation_node",
            },
        ),
        "compiler_node": (
            decide_next_optimisation_node,
            {
                "content_guidelines_optimisation_node": "content_guidelines_optimisation_node",
                "writing_guidelines_optimisation_node": "writing_guidelines_optimisation_node",
                "title_optimisation_node": "title_optimisation_node",
                "meta_description_optimisation_node": "meta_description_optimisation_node",
                END: END,
            },
        ),
        "content_guidelines_optimisation_node": (
            decide_next_optimisation_node,
            {
                "writing_guidelines_optimisation_node": "writing_guidelines_optimisation_node",
                "title_optimisation_node": "title_optimisation_node",
                "meta_description_optimisation_node": "meta_description_optimisation_node",
                END: END,
            },
        ),
        "writing_guidelines_optimisation_node": (
            check_readability_after_writing_optimisation,
            {
                "readability_evaluation_node": "readability_evaluation_node",
                "title_optimisation_node": "title_optimisation_node",
                "meta_description_optimisation_node": "meta_description_optimisation_node",
                END: END,
            },
        ),
        "personality_guidelines_evaluation_node": (
            check_personality_after_readability_optimisation,
            {
                "readability_evaluation_node": "readability_evaluation_node",
                "writing_guidelines_optimisation_node": "writing_guidelines_optimisation_node",
            },
        ),
        "title_optimisation_node": (
            decide_next_optimisation_node,
            {
                "meta_description_optimisation_node": "meta_description_optimisation_node",
                END: END,
            },
        ),
    }

    app = create_graph(RewritingState, nodes, edges, conditional_edges)

    # Save Graph
    draw_graph(
        app, f"{ROOT_DIR}/article-harmonisation/docs/images/article_rewriting_flow.png"
    )

    # starting up the respective llm agents
    researcher_agent = start_llm(MODEL, ROLES.RESEARCHER)
    compiler_agent = start_llm(MODEL, ROLES.COMPILER)
    meta_desc_optimisation_agent = start_llm(MODEL, ROLES.META_DESC)
    title_optimisation_agent = start_llm(MODEL, ROLES.TITLE)
    content_optimisation_agent = start_llm(MODEL, ROLES.CONTENT_OPTIMISATION)
    writing_optimisation_agent = start_llm(MODEL, ROLES.WRITING_OPTIMISATION)
    readability_evaluation_agent = start_llm(MODEL, ROLES.READABILITY_OPTIMISATION)
    personality_evaluation_agent = start_llm(MODEL, ROLES.PERSONALITY_EVALUATION)
    readability_optimisation_agent = start_llm(MODEL, ROLES.READABILITY_OPTIMISATION)

    # List with the articles to harmonise
    article_list = [
        # "Rubella",
        "How Dangerous Is Rubella?"
        # "Weight, BMI and Health Problems"
    ]

    processed_input_articles = concat_headers_to_content(article_list)

    # Dictionary with the various input keys and items
    inputs = {
        "article_rewriting_tries": 0,
        "original_article_inputs": {"article_content": processed_input_articles},
        "article_evaluation": {"article_title": article_list},
        "optimised_article_output": {
            "researcher_keypoints": [],
            "article_researcher_counter": 0,
        },
        "user_flags": {
            "flag_for_content_optimisation": True,
            "flag_for_writing_optimisation": True,
            "flag_for_title_optimisation": True,
            "flag_for_meta_desc_optimisation": True,
        },
        "llm_agents": {
            "researcher_agent": researcher_agent,
            "compiler_agent": compiler_agent,
            "content_optimisation_agent": content_optimisation_agent,
            "writing_optimisation_agent": writing_optimisation_agent,
            "title_optimisation_agent": title_optimisation_agent,
            "meta_desc_optimisation_agent": meta_desc_optimisation_agent,
            "readability_optimisation_agent": readability_optimisation_agent,
            "readability_evaluation_agent": readability_evaluation_agent,
            "personality_evaluation_agent": personality_evaluation_agent,
        },
    }

    result = execute_graph(app, inputs)

    # Prints the various checks
    print_checks(result, MODEL)

    print(result)
