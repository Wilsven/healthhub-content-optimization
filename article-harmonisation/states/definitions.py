from typing import Optional, TypedDict, Union

from agents.models import LLMInterface


class ArticleInputs(TypedDict):
    article_id: Union[int, Optional[list[int]]]
    article_content: Union[str, Optional[list[str]]]
    article_title: Union[str, Optional[list[str]]]
    meta_desc: Union[str, Optional[list[str]]]
    article_url: Union[str, Optional[list[str]]]
    content_category: Union[str, Optional[list[str]]]
    article_category_names: Union[str, Optional[list[str]]]
    page_views: Union[int, Optional[list[int]]]


class ContentFlags(TypedDict):
    is_unreadable: bool
    low_word_count: bool
    has_personality: bool  # from the personality evaluation node to determine if the optimised writing still meets the guideliens of the HH voice and personality


class ContentJudge(TypedDict):
    readability: str
    structure: str


class TitleFlags(TypedDict):
    long_title: bool


class TitleJudge(TypedDict):
    title: str


class MetaFlags(TypedDict):
    not_within_char_count: bool


class MetaJudge(TypedDict):
    meta_desc: str


class ChecksAgents(TypedDict):
    evaluation_agent: LLMInterface
    explanation_agent: LLMInterface


class OptimisedArticle(TypedDict):
    researcher_keypoints: Optional[list[str]]
    compiled_keypoints: Optional[str]
    optimised_content: Optional[str]
    optimised_writing: Optional[str]
    optimised_article_title: Optional[str]
    optimised_meta_desc: Optional[str]


class OptimisationFlags(TypedDict):
    flag_for_content_optimisation: bool
    flag_for_title_optimisation: bool
    flag_for_meta_desc_optimisation: bool
    flag_for_writing_optimisation: bool


class OptimisationAgents(TypedDict):
    researcher_agent: LLMInterface
    compiler_agent: LLMInterface
    content_optimisation_agent: LLMInterface
    writing_optimisation_agent: LLMInterface
    title_optimisation_agent: LLMInterface
    meta_desc_optimisation_agent: LLMInterface
    readability_optimisation_agent: LLMInterface
    personality_evaluation_agent: LLMInterface
    writing_evaluation_agent: LLMInterface
