"""
Model loading utilities for placeness scoring.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

import streamlit as st
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_SBERT_MODEL = "jhgan/ko-sroberta-multitask"
SBERT_FALLBACK_MODELS = (
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
)

DEFAULT_SENTIMENT_MODEL = "cringepnh/koelectra-korean-sentiment"
SENTIMENT_FALLBACK_MODELS = (
    "monologg/koelectra-base-finetuned-nsmc",
    "matthewburke/korean_sentiment",
    "nlptown/bert-base-multilingual-uncased-sentiment",
    "beomi/KcELECTRA-base",
)


@dataclass
class HuggingFaceSentimentRunner:
    """A small pipeline-like wrapper for HF sequence classification models."""

    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, texts, truncation: bool = True, max_length: int = 512):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        id2label = getattr(self.model.config, "id2label", {}) or {}
        results = []
        for row in probabilities.cpu():
            score, label_idx = torch.max(row, dim=0)
            label_idx = int(label_idx.item())
            label = id2label.get(label_idx) or id2label.get(str(label_idx)) or f"LABEL_{label_idx}"
            results.append({"label": str(label), "score": float(score.item())})
        return results


class HashingSentenceEncoder:
    """Fallback encoder when Sentence-BERT is unavailable."""

    def __init__(self, n_features: int = 4096):
        self.vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
        )
        self.model_name = "hashing-char-ngram-fallback"
        self.model_name_or_path = self.model_name

    def encode(
        self,
        texts,
        convert_to_tensor: bool = True,
        show_progress_bar: bool = False,
        batch_size: Optional[int] = None,
    ):
        del show_progress_bar, batch_size
        if isinstance(texts, str):
            texts = [texts]

        matrix = self.vectorizer.transform(list(texts))
        tensor = torch.tensor(matrix.toarray(), dtype=torch.float32)
        if convert_to_tensor:
            return tensor
        return tensor.numpy()


class HeuristicSentimentRunner:
    """Rule-based fallback sentiment runner for Korean cafe reviews."""

    POSITIVE_PATTERNS = [
        "좋",
        "만족",
        "추천",
        "깔끔",
        "예쁘",
        "편안",
        "친절",
        "쾌적",
        "조용",
        "맛있",
        "고급",
        "세련",
        "감성",
        "분위기 좋",
        "뷰가 좋",
        "채광",
        "매력",
        "최고",
    ]
    NEGATIVE_PATTERNS = [
        "별로",
        "아쉽",
        "불편",
        "시끄",
        "답답",
        "부족",
        "더럽",
        "지저분",
        "비싸",
        "좁",
        "최악",
        "불친절",
        "불만",
        "오래 기다",
        "애매",
        "우울",
        "정신없",
    ]

    def __call__(self, texts, truncation: bool = True, max_length: int = 512):
        del truncation, max_length
        if isinstance(texts, str):
            texts = [texts]

        results = []
        for text in texts:
            normalized = re.sub(r"\s+", " ", str(text)).strip().lower()
            pos_hits = sum(pattern in normalized for pattern in self.POSITIVE_PATTERNS)
            neg_hits = sum(pattern in normalized for pattern in self.NEGATIVE_PATTERNS)
            raw_score = 0.5 + (pos_hits * 0.12) - (neg_hits * 0.12)
            score = max(0.05, min(0.95, raw_score))

            if score >= 0.6:
                label = "POSITIVE"
                confidence = score
            elif score <= 0.4:
                label = "NEGATIVE"
                confidence = 1.0 - score
            else:
                label = "NEUTRAL"
                confidence = 0.5 + abs(score - 0.5)

            results.append({"label": label, "score": float(confidence)})

        return results


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


def get_embedding_model_name(model) -> str:
    direct_name = getattr(model, "model_name_or_path", None) or getattr(model, "model_name", None)
    if direct_name:
        return str(direct_name)

    try:
        first_module = model[0]
        auto_model = getattr(first_module, "auto_model", None)
        nested_name = getattr(auto_model, "name_or_path", None)
        if nested_name:
            return str(nested_name)
    except Exception:
        pass

    return type(model).__name__


def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def _load_sbert_model(preferred_model_name: Optional[str] = None):
    """Load Sentence-BERT and return a fallback encoder on failure."""
    candidate_names = _dedupe_preserve_order(
        [
            preferred_model_name,
            os.getenv("PLACENESS_SBERT_MODEL"),
            DEFAULT_SBERT_MODEL,
            *SBERT_FALLBACK_MODELS,
        ]
    )

    try:
        for model_name in candidate_names:
            try:
                return _load_sentence_transformer(model_name)
            except Exception:
                continue
    except Exception:
        pass

    st.warning(
        "Sentence-BERT 로드에 실패해 fallback 텍스트 인코더를 사용합니다. "
        "이 환경에서는 장소성 유사도 품질이 다소 낮아질 수 있습니다."
    )
    return HashingSentenceEncoder()


def _build_sentiment_runner(model_name: str) -> HuggingFaceSentimentRunner:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return HuggingFaceSentimentRunner(tokenizer=tokenizer, model=model, device="cpu")


@st.cache_resource
def load_models(
    preferred_sbert_model: Optional[str] = None,
    preferred_sentiment_model: Optional[str] = None,
):
    """
    Load Sentence-BERT and sentiment models.

    Returns:
        tuple: (sbert_model, sentiment_runner, sentiment_model_name)
    """
    sbert_model = _load_sbert_model(preferred_sbert_model)

    sentiment_pipeline = None
    sentiment_model_name = ""
    candidate_names = _dedupe_preserve_order(
        [
            preferred_sentiment_model,
            os.getenv("PLACENESS_SENTIMENT_MODEL"),
            DEFAULT_SENTIMENT_MODEL,
            *SENTIMENT_FALLBACK_MODELS,
        ]
    )

    for model_name in candidate_names:
        try:
            sentiment_pipeline = _build_sentiment_runner(model_name)
            sentiment_model_name = model_name
            break
        except Exception:
            continue

    if sentiment_pipeline is None:
        st.warning(
            "감성 분석 모델 로드에 실패해 규칙 기반 fallback 감성 분석기를 사용합니다. "
            "이 환경에서는 리뷰 감성 품질이 다소 낮아질 수 있습니다."
        )
        sentiment_pipeline = HeuristicSentimentRunner()
        sentiment_model_name = "heuristic-rule-based-fallback"

    return sbert_model, sentiment_pipeline, sentiment_model_name
