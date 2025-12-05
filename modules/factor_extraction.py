"""
요인 추출 모듈 (SBERT 리뷰 → 요인 매핑)
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def extract_factor_mentions(reviews, sbert_model, factor_embeddings, factor_names, threshold):
    """
    리뷰에서 각 요인에 대한 언급을 추출합니다.
    
    Args:
        reviews: 리뷰 텍스트 리스트
        sbert_model: Sentence-BERT 모델
        factor_embeddings: 요인 정의 임베딩 (텐서)
        factor_names: 요인 이름 리스트
        threshold: 유사도 임계값
    
    Returns:
        dict: {review_idx: [(factor_name, similarity_score), ...]}
    """
    # 리뷰 임베딩 생성
    review_embeddings = sbert_model.encode(reviews, convert_to_tensor=True, show_progress_bar=False)
    
    # 코사인 유사도 계산
    similarity_matrix = cosine_similarity(
        review_embeddings.cpu().numpy(),
        factor_embeddings.cpu().numpy()
    )
    
    # 각 리뷰별로 임계값 이상인 요인 추출
    mentions = {}
    for review_idx in range(len(reviews)):
        relevant_factors = []
        for factor_idx, factor_name in enumerate(factor_names):
            similarity = similarity_matrix[review_idx, factor_idx]
            if similarity >= threshold:
                relevant_factors.append((factor_name, similarity))
        mentions[review_idx] = relevant_factors
    
    return mentions, similarity_matrix

