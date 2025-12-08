"""
장소성 점수 계산 모듈 (fsi, wi, mu, sigma, df)

[알고리즘 개요]

1) SBERT 기반 요인 매핑
   - 각 리뷰가 12개 장소성 요인 중 어떤 것들을 '언급'하는지 판정
   - similarity >= threshold 인 경우에만 해당 요인에 대한 언급으로 간주

2) 리뷰 감성 분석
   - 각 리뷰별로 감성 점수 sentiment_score \in [0, 1] 계산
   - SBERT similarity는 "언급 여부" 결정에만 사용하고,
     감성 점수는 순수하게 sentiment 모델 출력에 기반

3) fsi 계산 (Factor Sentiment Index)
   - 요인 i 에 대해:
       fsi_i = mean( sentiment_score(r) | 리뷰 r 이 요인 i 를 언급 )

   - 언급이 하나도 없으면 fsi_i = 0.5 (중립)

4) 가중치 Wi 계산 (요인 중요도)
   - n_i = 요인 i 를 언급한 리뷰 수
   - Sum_n = Σ_i n_i
   - Wi_i = n_i / Sum_n  (Sum_n == 0 이면 Wi_i = 0)

5) 종합 장소성 점수 Mu
   - Mu = Σ_i (Wi_i * fsi_i)

6) 요인 점수 표준편차 Sigma
   - Sigma = std( {fsi_i} )

7) 강점/약점 요인 df+, df-
   - 기준: 평균(Mu) 대비 DEVIATION_THRESHOLD 이상 차이 나는 요인만 선별
   - score >= Mu + δ 이면 강점(df+)
   - score <= Mu - δ 이면 약점(df-)
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from modules.config import DEVIATION_THRESHOLD
from modules.factor_extraction import extract_factor_mentions
from modules.sentiment import process_sentiment_result
from modules.preprocess import truncate_text_for_bert, is_numeric_only, is_metadata_only
from sklearn.metrics.pairwise import cosine_similarity
import re


# -----------------------------------------------------------------------------
# 1. fsi, 가중치, Mu, Sigma, df 계산용 순수 함수
# -----------------------------------------------------------------------------


def calculate_fsi(
    factor_mentions: Dict[int, List[Tuple[str, float]]],
    sentiment_scores: Dict[int, float],
    factor_names: List[str],
) -> Dict[str, float]:
    """
    Factor Sentiment Index (fsi)를 계산합니다.

    Args:
        factor_mentions:
            리뷰 인덱스 -> [(factor_name, similarity), ...]
            (similarity는 언급 여부 판단용이며, fsi 계산에는 사용하지 않음)
        sentiment_scores:
            리뷰 인덱스 -> sentiment_score (0~1, 긍정 확률)
        factor_names:
            전체 요인 이름 리스트 (e.g. 12개)

    Returns:
        fsi: {factor_name: fsi_score}
    """
    # 요인별로 관련 리뷰들의 감성 점수 모으기
    factor_to_scores: Dict[str, List[float]] = {f: [] for f in factor_names}

    for review_idx, factors in factor_mentions.items():
        if review_idx not in sentiment_scores:
            continue
        s_score = sentiment_scores[review_idx]
        for factor_name, _sim in factors:
            if factor_name in factor_to_scores:
                factor_to_scores[factor_name].append(s_score)

    # fsi = 요인별 감성 점수 평균 (언급 없는 경우 0.5)
    fsi: Dict[str, float] = {}
    for factor_name in factor_names:
        scores = factor_to_scores.get(factor_name, [])
        if scores:
            fsi[factor_name] = float(np.mean(scores))
        else:
            fsi[factor_name] = 0.5  # 언급이 없는 요인은 중립 점수로 처리

    return fsi


def calculate_weights(n_i: Dict[str, int]) -> Dict[str, float]:
    """
    요인별 언급 수 n_i 로부터 가중치 Wi 를 계산합니다.

    Wi_i = n_i / Σ_j n_j

    Args:
        n_i: {factor_name: mention_count}

    Returns:
        Wi: {factor_name: weight}
    """
    total_mentions = sum(n_i.values())
    if total_mentions == 0:
        # 언급 자체가 없는 경우: 모든 가중치 0
        return {factor: 0.0 for factor in n_i.keys()}

    Wi = {factor: count / total_mentions for factor, count in n_i.items()}
    return Wi


def calculate_mu_sigma(
    fsi: Dict[str, float],
    Wi: Dict[str, float],
) -> Tuple[float, float]:
    """
    종합 장소성 점수 Mu 와 요인 점수의 표준편차 Sigma 를 계산합니다.

    Args:
        fsi: {factor_name: fsi_score}
        Wi:  {factor_name: weight}

    Returns:
        (Mu, Sigma)
    """
    # Mu: 가중 평균
    mu = 0.0
    for factor_name, fsi_score in fsi.items():
        w = Wi.get(factor_name, 0.0)
        mu += fsi_score * w

    # Sigma: 요인 점수들의 표준편차 (가중치와 무관, 순수 분산)
    scores = list(fsi.values()) if fsi else [0.5]
    sigma = float(np.std(scores)) if len(scores) > 0 else 0.0

    return mu, sigma


def extract_deviant_features(
    fsi: Dict[str, float],
    mu: float,
    n_i: Dict[str, int],
) -> Tuple[List[str], List[str]]:
    """
    특이 특징 (df+, df-)를 추출합니다.

    Args:
        fsi: {factor_name: fsi_score}
        mu:  종합 장소성 점수
        n_i: {factor_name: mention_count}

    Returns:
        (strong_factors, weak_factors)
        - strong_factors: 평균 대비 충분히 높은 요인 목록 (df+)
        - weak_factors:   평균 대비 충분히 낮은 요인 목록 (df-)
    """
    strong_factors: List[str] = []
    weak_factors: List[str] = []

    for factor_name, score in fsi.items():
        # 언급이 아예 없는 요인은 df 판단에서 제외
        if n_i.get(factor_name, 0) <= 0:
            continue

        if score >= mu + DEVIATION_THRESHOLD:
            strong_factors.append(factor_name)
        elif score <= mu - DEVIATION_THRESHOLD:
            weak_factors.append(factor_name)

    return strong_factors, weak_factors


# -----------------------------------------------------------------------------
# 2. 리뷰 단위 감성 분석 헬퍼
# -----------------------------------------------------------------------------


def _analyze_sentiment_for_reviews(
    review_texts: List[str],
    sentiment_pipeline,
    sentiment_model_name: str = "",
) -> Dict[int, float]:
    """
    리뷰 리스트에 대해 감성 점수를 계산합니다.

    - SBERT similarity와는 독립적으로, 순수 텍스트 기반 감성 분석
    - 숫자-only / 메타데이터-only 리뷰는 간단한 규칙 기반 처리

    Returns:
        sentiment_scores: {local_review_idx: sentiment_score(0~1)}
    """
    sentiment_scores: Dict[int, float] = {}

    if not review_texts:
        return sentiment_scores

    batch_size = 8
    total = len(review_texts)
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = review_texts[start:end]

        status_text.text(f"리뷰 감성 분석 중... ({end}/{total})")
        progress_bar.progress(end / total)

        # 숫자-only / 메타데이터-only와 일반 텍스트 분리
        model_inputs: List[Tuple[int, str]] = []
        for idx_local, text in enumerate(batch):
            global_idx = start + idx_local
            t = (text or "").strip()

            # 숫자-only: 예를 들어 "5", "4.5" 등
            if is_numeric_only(t):
                try:
                    rating_val = float(t)
                    if rating_val >= 4.0:
                        sentiment_scores[global_idx] = 0.9
                    elif rating_val >= 3.0:
                        sentiment_scores[global_idx] = 0.5
                    else:
                        sentiment_scores[global_idx] = 0.1
                except ValueError:
                    sentiment_scores[global_idx] = 0.5
                continue

            # 메타데이터-only: 식사 유형 텍스트 등 → 중립 처리
            if is_metadata_only(t):
                sentiment_scores[global_idx] = 0.5
                continue

            # 실제 텍스트: BERT 모델에 전달
            truncated_text = truncate_text_for_bert(t)
            model_inputs.append((global_idx, truncated_text))

        # 모델에 배치로 전달
        if model_inputs:
            texts_for_model = [t for _, t in model_inputs]
            try:
                results = sentiment_pipeline(
                    texts_for_model,
                    truncation=True,
                    max_length=512,
                )
                for (global_idx, _), res in zip(model_inputs, results):
                    _label, score = process_sentiment_result(res, sentiment_model_name)
                    sentiment_scores[global_idx] = float(score)
            except Exception as e:
                st.warning(f"감성 분석 배치 처리 중 오류 발생, 중립 점수로 대체합니다: {e}")
                for global_idx, _ in model_inputs:
                    sentiment_scores[global_idx] = 0.5

    progress_bar.empty()
    status_text.empty()
    return sentiment_scores


# -----------------------------------------------------------------------------
# 2-1. 문장 단위 분리 헬퍼
# -----------------------------------------------------------------------------


def split_into_sentences(text: str) -> List[str]:
    """
    텍스트를 문장 단위로 분리합니다.
    
    Args:
        text: 분리할 텍스트
    
    Returns:
        문장 리스트
    """
    if not text or not isinstance(text, str):
        return []
    
    text = text.strip()
    if not text:
        return []
    
    # 한국어 문장 종결 부호로 분리 (. ! ?)
    # 연속된 공백 정리
    sentences = re.split(r'[.!?]\s+', text)
    
    # 빈 문장 제거 및 정리
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 문장이 너무 짧으면 제거 (1글자 이하)
    sentences = [s for s in sentences if len(s) > 1]
    
    return sentences if sentences else [text]  # 분리 실패 시 원본 반환


# -----------------------------------------------------------------------------
# 3. 메인: 장소성 요인 점수 계산 (카페별 fsi, 리뷰수 등)
# -----------------------------------------------------------------------------


def calculate_place_scores(
    df_reviews: pd.DataFrame,
    sbert_model,
    sentiment_pipeline,
    factor_defs: Dict[str, str],
    similarity_threshold: float = 0.4,
    sentiment_model_name: str = "",
):
    """
    Sentence-BERT와 감성 분석을 사용하여
    - 카페별 요인 점수 (fsi)
    - 카페별 요인 언급 수 (n_i)
    - 리뷰별 요인 점수/유사도
    를 계산합니다.
    
    **새로운 방식**: 각 요인별로 유사도 임계값 이상인 리뷰만 선별하여
    해당 요인에 대한 감성 분석을 독립적으로 실행합니다.
    이를 통해 요인별로 다른 감성 점수를 얻을 수 있습니다.

    Args:
        df_reviews: 컬럼에 ['cafe_name', 'review_text'] 포함된 DataFrame
        sbert_model: SentenceTransformer 모델
        sentiment_pipeline: HuggingFace transformers pipeline (sentiment-analysis)
        factor_defs: {factor_name: SBERT 정의 문장}
        similarity_threshold: 요인 언급으로 인정할 SBERT 코사인 유사도 임계값
        sentiment_model_name: process_sentiment_result 에 전달할 모델 이름

    Returns:
        df_cafe_scores: 카페별 요인 점수/언급수 테이블
        df_review_scores: 리뷰별 요인 점수/유사도 테이블
    """
    st.subheader("1. SBERT 기반 요인 매핑 및 정밀 감성 분석")

    # 1) SBERT용 요인 정의 문장 임베딩 미리 계산
    factor_names: List[str] = list(factor_defs.keys())
    factor_sentences: List[str] = list(factor_defs.values())

    with st.spinner("장소성 요인 정의 임베딩 생성 중..."):
        factor_embeddings = sbert_model.encode(
            factor_sentences,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

    results_list: List[Dict] = []
    review_scores_list: List[Dict] = []

    cafe_groups = df_reviews.groupby("cafe_name")
    total_cafes = len(cafe_groups)

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    for cafe_idx, (cafe_name, group) in enumerate(cafe_groups):
        try:
            # 진행 상황 업데이트 (Streamlit Cloud 타임아웃 방지)
            status_text.text(f"카페 처리 중: {cafe_name} ({cafe_idx + 1}/{total_cafes})")
            progress_bar.progress((cafe_idx + 1) / total_cafes)
            
            # 주기적으로 Streamlit이 응답할 수 있도록 (매 5개 카페마다)
            if cafe_idx % 5 == 0 and cafe_idx > 0:
                import time
                time.sleep(0.2)  # 짧은 대기로 Streamlit이 응답 처리할 시간 제공

            # 해당 카페의 리뷰 텍스트 및 인덱스
            review_texts: List[str] = group["review_text"].astype(str).tolist()
            review_indices: List[int] = group.index.tolist()
            n_reviews_cafe = len(review_texts)

            if n_reviews_cafe == 0:
                # 리뷰가 아예 없는 카페
                cafe_scores = {"cafe_name": cafe_name}
                for factor in factor_names:
                    cafe_scores[f"점수_{factor}"] = 0.5
                    cafe_scores[f"리뷰수_{factor}"] = 0
                results_list.append(cafe_scores)
                continue

            # 2) 리뷰를 문장 단위로 분리 및 요인 매핑
            with st.spinner(f"{cafe_name} - 문장 단위 분리 및 요인 매핑 중..."):
                # 2-1. 각 리뷰를 문장 단위로 분리
                all_sentences: List[str] = []
                sentence_to_review: List[int] = []  # 각 문장이 어느 리뷰에서 왔는지
                
                for local_idx, review_text in enumerate(review_texts):
                    sentences = split_into_sentences(review_text)
                    all_sentences.extend(sentences)
                    sentence_to_review.extend([local_idx] * len(sentences))
                
                if len(all_sentences) == 0:
                    # 문장이 없는 경우 처리
                    cafe_scores = {"cafe_name": cafe_name}
                    for factor in factor_names:
                        cafe_scores[f"점수_{factor}"] = 0.5
                        cafe_scores[f"리뷰수_{factor}"] = 0
                    results_list.append(cafe_scores)
                    continue
                
                # 2-2. 각 문장에 대해 요인 매핑 (SBERT)
                sentence_embeddings = sbert_model.encode(
                    all_sentences,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                
                # 문장-요인 유사도 행렬 계산
                from sklearn.metrics.pairwise import cosine_similarity
                sentence_factor_similarity = cosine_similarity(
                    sentence_embeddings.cpu().numpy(),
                    factor_embeddings.cpu().numpy()
                )
                # shape: (n_sentences, n_factors)
            
                # 3) 리뷰별 요인 점수 저장을 위한 딕셔너리 초기화
            # {local_review_idx: {factor_name: [sentiment_scores]}}
            review_factor_sentiment_scores: Dict[int, Dict[str, List[float]]] = {
                idx: {factor: [] for factor in factor_names} for idx in range(n_reviews_cafe)
            }
            
            # 리뷰별 요인 점수 (최종 평균용)
            review_factor_avg_scores: Dict[int, Dict[str, float]] = {
                idx: {} for idx in range(n_reviews_cafe)
            }

            # 4) 각 요인별로 유효 문장 수집 및 감성 분석 재실행
            cafe_scores = {"cafe_name": cafe_name}
            
            for f_idx, factor_name in enumerate(factor_names):
                # 4-1. 유사도 임계값 이상인 문장(유효 언급) 선별
                relevant_sentence_indices = np.where(
                    sentence_factor_similarity[:, f_idx] >= similarity_threshold
                )[0]
                
                if len(relevant_sentence_indices) > 0:
                    # 유효 문장 추출
                    relevant_sentences = [all_sentences[idx] for idx in relevant_sentence_indices]
                    
                    # 4-2. 유효 문장에 대해서만 감성 분석 재실행
                    try:
                        # 숫자-only / 메타데이터-only와 일반 텍스트 분리
                        model_inputs: List[Tuple[int, str]] = []
                        rule_based_scores: Dict[int, float] = {}
                        
                        for sent_idx in relevant_sentence_indices:
                            sentence = all_sentences[sent_idx]
                            t = (sentence or "").strip()
                            
                            # 숫자-only 처리
                            if is_numeric_only(t):
                                try:
                                    rating_val = float(t)
                                    if rating_val >= 4.0:
                                        rule_based_scores[sent_idx] = 0.9
                                    elif rating_val >= 3.0:
                                        rule_based_scores[sent_idx] = 0.5
                                    else:
                                        rule_based_scores[sent_idx] = 0.1
                                except ValueError:
                                    rule_based_scores[sent_idx] = 0.5
                                continue
                            
                            # 메타데이터-only 처리
                            if is_metadata_only(t):
                                rule_based_scores[sent_idx] = 0.5
                                continue
                            
                            # 일반 텍스트: 모델에 전달
                            truncated_text = truncate_text_for_bert(t)
                            model_inputs.append((sent_idx, truncated_text))
                    
                        # 배치로 감성 분석 수행
                        if model_inputs:
                            texts_for_model = [t for _, t in model_inputs]
                            try:
                                results = sentiment_pipeline(
                                    texts_for_model,
                                    truncation=True,
                                    max_length=512,
                                )
                                for (sent_idx, _), res in zip(model_inputs, results):
                                    _label, score = process_sentiment_result(res, sentiment_model_name)
                                    # 해당 문장이 속한 리뷰 찾기
                                    review_idx = sentence_to_review[sent_idx]
                                    review_factor_sentiment_scores[review_idx][factor_name].append(float(score))
                            except Exception as e:
                                st.warning(f"{cafe_name} - {factor_name} 감성 분석 배치 처리 중 오류: {e}")
                                for sent_idx, _ in model_inputs:
                                    review_idx = sentence_to_review[sent_idx]
                                    review_factor_sentiment_scores[review_idx][factor_name].append(0.5)
                        
                        # 규칙 기반 점수 추가
                        for sent_idx, score in rule_based_scores.items():
                            review_idx = sentence_to_review[sent_idx]
                            review_factor_sentiment_scores[review_idx][factor_name].append(score)
                        
                        # 4-3. 리뷰별 요인 점수 계산 (각 리뷰의 해당 요인 문장들의 평균)
                        factor_sentiment_scores_for_fsi = []
                        for review_idx in range(n_reviews_cafe):
                            scores = review_factor_sentiment_scores[review_idx][factor_name]
                            if scores:
                                avg_score = float(np.mean(scores))
                                review_factor_avg_scores[review_idx][factor_name] = avg_score
                                factor_sentiment_scores_for_fsi.append(avg_score)
                        
                        # 4-4. FSI 산출: 유효 언급된 리뷰들의 감성 점수 평균
                        if len(factor_sentiment_scores_for_fsi) > 0:
                            avg_score = float(np.mean(factor_sentiment_scores_for_fsi))
                            cafe_scores[f"점수_{factor_name}"] = avg_score
                            # 해당 요인을 언급한 리뷰 수 (최소 1개 문장이라도 해당 요인 언급)
                            cafe_scores[f"리뷰수_{factor_name}"] = len(factor_sentiment_scores_for_fsi)
                        else:
                            cafe_scores[f"점수_{factor_name}"] = 0.5
                            cafe_scores[f"리뷰수_{factor_name}"] = 0
                        
                    except Exception as e:
                        st.warning(f"{cafe_name} - {factor_name} 감성 분석 오류: {e}")
                        cafe_scores[f"점수_{factor_name}"] = 0.5
                        cafe_scores[f"리뷰수_{factor_name}"] = 0
                else:
                    # 언급이 없는 경우
                    cafe_scores[f"점수_{factor_name}"] = 0.5
                    cafe_scores[f"리뷰수_{factor_name}"] = 0
        
            # 5) 리뷰별 요인 점수/유사도 기록 (df_review_scores용)
            # UI 호환을 위해 기존 구조 유지 (wide format)
            # 리뷰 전체에 대한 유사도는 원래 방식대로 계산 (리뷰 전체 기준)
            review_embeddings = sbert_model.encode(
                review_texts,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            from sklearn.metrics.pairwise import cosine_similarity
            review_similarity_matrix = cosine_similarity(
                review_embeddings.cpu().numpy(),
                factor_embeddings.cpu().numpy()
            )
            
            for local_idx, (text, global_idx) in enumerate(zip(review_texts, review_indices)):
                row_dict = {
                    "review_index": global_idx,
                    "cafe_name": cafe_name,
                    "review_text": text,
                }
                
                for f_idx, factor_name in enumerate(factor_names):
                    # 리뷰 전체에 대한 유사도
                    sim = float(review_similarity_matrix[local_idx, f_idx])
                    row_dict[f"{factor_name}_유사도"] = sim
                    
                    # 해당 요인에 대한 감성 점수가 있으면 사용 (문장 단위 분석 결과)
                    if factor_name in review_factor_avg_scores[local_idx]:
                        row_dict[f"{factor_name}_점수"] = float(review_factor_avg_scores[local_idx][factor_name])
                    else:
                        row_dict[f"{factor_name}_점수"] = np.nan
                
                review_scores_list.append(row_dict)
            
            results_list.append(cafe_scores)
            
        except Exception as e:
            # 개별 카페 처리 중 오류 발생 시 로그 남기고 계속 진행
            st.warning(f"카페 '{cafe_name}' 처리 중 오류 발생: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            # 오류 발생한 카페는 기본값으로 추가
            cafe_scores = {"cafe_name": cafe_name}
            for factor in factor_names:
                cafe_scores[f"점수_{factor}"] = 0.5
                cafe_scores[f"리뷰수_{factor}"] = 0
            results_list.append(cafe_scores)
            continue

    progress_bar.empty()
    status_text.empty()

    df_cafe_scores = pd.DataFrame(results_list)
    df_review_scores = pd.DataFrame(review_scores_list)

    return df_cafe_scores, df_review_scores


# -----------------------------------------------------------------------------
# 4. 최종 연구 지표: Wi, Mu, Sigma, df+, df- 계산
# -----------------------------------------------------------------------------


def calculate_final_research_metrics(
    df_scores: pd.DataFrame,
    factor_names: List[str],
    total_reviews: int,  # ⚠️ 인터페이스 유지용 인자 (실제 계산에는 사용하지 않음)
) -> pd.DataFrame:
    """
    최종 연구 알고리즘 지표 (Wi, Mu, Sigma, df+, df-)를 계산합니다.

    Args:
        df_scores:
            calculate_place_scores 에서 반환된 카페별 점수 DataFrame
            (컬럼에 'cafe_name', '점수_{factor}', '리뷰수_{factor}' 포함)
        factor_names:
            장소성 요인 이름 리스트
        total_reviews:
            전체 리뷰 수 (이전 버전과의 호환을 위해 인자로 받지만,
            현재 구현에서는 per-place 정규화를 사용하므로 직접 사용하지는 않습니다.)

    Returns:
        df_final: Wi, Mu, Sigma, df+, df-가 포함된 최종 DataFrame
    """
    df = df_scores.copy()

    # NaN 값 처리: 점수는 0.5, 리뷰수는 0으로
    for factor in factor_names:
        score_col = f"점수_{factor}"
        count_col = f"리뷰수_{factor}"
        if score_col in df.columns:
            df[score_col] = df[score_col].fillna(0.5)
        else:
            df[score_col] = 0.5

        if count_col in df.columns:
            df[count_col] = df[count_col].fillna(0).astype(int)
        else:
            df[count_col] = 0

    # 각 카페(row)마다 Wi, Mu, Sigma, df+, df- 계산
    final_rows: List[Dict] = []

    for _, row in df.iterrows():
        cafe_name = row["cafe_name"]

        # 1) fsi, n_i 구성
        fsi: Dict[str, float] = {}
        n_i: Dict[str, int] = {}

        for factor in factor_names:
            fsi[factor] = float(row.get(f"점수_{factor}", 0.5))
            n_i[factor] = int(row.get(f"리뷰수_{factor}", 0))

        # 2) Wi 계산 (per-place 정규화)
        Wi = calculate_weights(n_i)

        # 3) Mu, Sigma 계산
        mu, sigma = calculate_mu_sigma(fsi, Wi)

        # 4) df+, df- 추출
        strong_factors, weak_factors = extract_deviant_features(fsi, mu, n_i)

        # 5) 결과 row 구성
        result_row: Dict = {"cafe_name": cafe_name}

        # fsi, 리뷰수
        for factor in factor_names:
            result_row[f"점수_{factor}"] = fsi[factor]
            result_row[f"리뷰수_{factor}"] = n_i[factor]

        # Wi
        for factor in factor_names:
            result_row[f"Wi_{factor}"] = Wi.get(factor, 0.0)

        # Mu, Sigma, df+, df-
        result_row["종합_장소성_점수_Mu"] = mu
        result_row["요인_점수_표준편차_Sigma"] = sigma
        result_row["강점_요인(+df+)"] = ", ".join(strong_factors) if strong_factors else "N/A"
        result_row["약점_요인(-df-)"] = ", ".join(weak_factors) if weak_factors else "N/A"

        # 논문에서 쓰는 Summary 형식: [mu ± sigma, +count, -count]
        result_row["Final_PlaceScore_Summary"] = (
            f"[{mu:.3f} ± {sigma:.3f}, +{len(strong_factors)}, -{len(weak_factors)}]"
        )

        final_rows.append(result_row)

    df_final = pd.DataFrame(final_rows)

    # 원래 df_scores에 있던 다른 컬럼이 필요하면 merge로 합쳐도 됨
    # 여기서는 'cafe_name' 기준 left join으로 보존
    df_final = pd.merge(df, df_final, on="cafe_name", how="left", suffixes=("", "_calc"))

    return df_final
