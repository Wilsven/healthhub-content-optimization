import numpy as np
from transformers.models.bert import BertTokenizerFast


def split_into_chunks(
    sentences: list[str], max_length: int, tokenizer: BertTokenizerFast
) -> list[str]:
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Tokenize the sentence
        encoded_sentence = tokenizer(sentence, return_tensors="pt")
        num_tokens = encoded_sentence["input_ids"].shape[1]

        # If adding the current sentence would exceed max_length, save the current chunk and start a new one
        if current_length + num_tokens > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += num_tokens

    # Add the last chunk if any
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def pool_embeddings(embeddings: np.ndarray, strategy: str = "mean") -> np.ndarray:
    if not embeddings:
        raise ValueError("The embeddings are empty.")

    if strategy == "mean":
        return np.mean(embeddings, axis=0)
    elif strategy == "max":
        return np.max(embeddings, axis=0)
    else:
        raise ValueError(
            "Pooling strategy not recognized. The strategy must be either 'mean' or 'max'."
        )
