import logging
import os
from collections.abc import Iterable
from typing import Optional, cast

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils._bunch import Bunch

logger = logging.getLogger(__name__)
hander = logging.StreamHandler()
if os.environ.get("DEBUG"):
    logger.setLevel(logging.DEBUG)
    hander.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARNING)
    hander.setLevel(logging.WARNING)
logger.addHandler(hander)


def fetch_train_corpus(categories: Optional[Iterable[str]] = None) -> Bunch:
    if categories is not None:
        categories = list(categories)
    news = cast(
        Bunch,
        fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=0),
    )
    # news.data: list[str] of length <=11314
    # news.target: np.array of dtype int of length <=11314
    # news.target_names: list[str] .. names of category
    assert len(news.data) == len(news.target)
    logger.debug(f"len of new data: {len(news.data)}")
    logger.debug(f"target_names: {news.target_names}")
    return news


def calclulate_tfidf(corpus: Bunch) -> pd.DataFrame:
    count_vectorizer = CountVectorizer(stop_words="english")
    bow = count_vectorizer.fit_transform(corpus.data)  # csr_matrix

    logger.debug(f"bow's type: {type(bow)}")
    logger.debug(f"bow's shape: {bow.shape}")

    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(bow)

    inverse_vocab = {v: k for k, v in count_vectorizer.vocabulary_.items()}
    tfidf = pd.DataFrame(
        sorted(
            (
                {
                    "news_id": news_id,
                    "word": inverse_vocab[word_id],
                    "tfidf": tfidf_matrix[news_id, word_id],
                }
                for news_id in range(0, tfidf_matrix.shape[0])
                for word_id in tfidf_matrix[news_id, :].indices
            ),
            key=lambda d: (d["news_id"], d["tfidf"]),
        )
    )
    return tfidf


if __name__ == "__main__":
    from ml_hands_on import PROJECT_ROOT

    OUTDIR = PROJECT_ROOT / "output"
    OUTDIR.mkdir(exist_ok=True, parents=True)

    categories = ["rec.motorcycles", "sci.med"]
    tfidf = calclulate_tfidf(fetch_train_corpus(categories))
    tfidf.to_csv(OUTDIR / "tfidf.csv", index=False, header=True)
