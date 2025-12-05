"""
모델 로딩 모듈
"""
import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_models():
    """
    Sentence-BERT와 감성 분석 모델을 로드합니다.
    
    Returns:
        tuple: (sbert_model, sentiment_pipeline, sentiment_model_name)
    """
    # 1. Sentence-BERT 모델 로드 (임베딩 및 유사도 계산용)
    try:
        sbert_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    except Exception as e:
        # 기본 모델 로드 실패 시 대체 모델 사용
        try:
            sbert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
        except Exception as e2:
            raise Exception(f"Sentence-BERT 모델 로드 실패: {e}, {e2}")
    
    # 2. 감성 분석 모델 로드 (한국어 리뷰 감성 분석 특화 모델)
    sentiment_pipeline = None
    model_loaded = False
    last_error = None
    
    # 우선순위: 더 가벼운 모델부터 시도 (Streamlit Cloud 메모리 제한 고려)
    model_candidates = [
        {
            "name": "nlptown/bert-base-multilingual-uncased-sentiment",
            "description": "다국어 감성 분석 모델 (한국어 포함, 5단계 감성)",
            "is_nlptown": True
        },
        {
            "name": "matthewburke/korean_sentiment",
            "description": "한국어 감성 분석 전용 모델",
            "is_nlptown": False
        },
        {
            "name": "beomi/KcELECTRA-base",
            "description": "KoELECTRA base",
            "is_nlptown": False
        },
        {
            "name": "monologg/kobert-base-v1",
            "description": "KoBERT (fallback)",
            "is_nlptown": False
        }
    ]
    
    for model_info in model_candidates:
        try:
            sentiment_model_name = model_info["name"]
            
            # 특별 처리: nlptown 모델은 이미 fine-tuned되어 있음
            if model_info.get("is_nlptown", False):
                sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=sentiment_model_name,
                    device=-1,  # Streamlit Cloud는 CPU만 사용
                    truncation=True,
                    max_length=512
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
                num_labels = 2  # 대부분의 한국어 모델은 2-class
                
                model = AutoModelForSequenceClassification.from_pretrained(
                    sentiment_model_name, 
                    num_labels=num_labels
                )
                sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1,  # Streamlit Cloud는 CPU만 사용
                    truncation=True,
                    max_length=512
                )
            
            sentiment_model_name = model_info["name"]
            model_loaded = True
            break
            
        except Exception as e:
            last_error = e
            continue
    
    if not model_loaded or sentiment_pipeline is None:
        error_msg = f"모든 감성 분석 모델 로드 실패. 마지막 오류: {last_error}"
        raise Exception(error_msg)

    return sbert_model, sentiment_pipeline, sentiment_model_name

