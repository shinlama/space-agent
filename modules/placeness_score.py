"""
장소성 점수 계산 모듈 (fsi, wi, mu, sigma, df)
"""
import numpy as np
import pandas as pd
import streamlit as st
from modules.config import DEVIATION_THRESHOLD
from modules.factor_extraction import extract_factor_mentions
from modules.sentiment import analyze_sentiment, process_sentiment_result
from modules.preprocess import truncate_text_for_bert, is_numeric_only, is_metadata_only


def calculate_fsi(factor_mentions, sentiment_scores):
    """
    Factor Sentiment Index (fsi)를 계산합니다.
    각 요인에 대한 관련 리뷰들의 감성 점수의 산술 평균입니다.
    
    Args:
        factor_mentions: {review_idx: [(factor_name, similarity), ...]}
        sentiment_scores: {review_idx: (label, score)}
    
    Returns:
        dict: {factor_name: fsi_score}
    """
    factor_scores = {}
    
    # 요인별로 그룹화
    factor_reviews = {}
    for review_idx, factors in factor_mentions.items():
        for factor_name, similarity in factors:
            if factor_name not in factor_reviews:
                factor_reviews[factor_name] = []
            if review_idx in sentiment_scores:
                factor_reviews[factor_name].append(sentiment_scores[review_idx][1])
    
    # 각 요인의 fsi 계산 (산술 평균)
    for factor_name, scores in factor_reviews.items():
        if len(scores) > 0:
            factor_scores[factor_name] = np.mean(scores)
        else:
            factor_scores[factor_name] = 0.5  # 언급이 없으면 0.5
    
    return factor_scores


def calculate_weight(n_i, total_reviews):
    """
    가중치 (Wi)를 계산합니다.
    Wi = Ri / Sum_R, where Ri = n_i / total_reviews
    
    Args:
        n_i: 요인별 언급 수
        total_reviews: 전체 리뷰 수
    
    Returns:
        dict: {factor_name: weight}
    """
    # Ri 계산
    Ri = {factor: count / total_reviews for factor, count in n_i.items()}
    
    # Sum_R 계산
    Sum_R = sum(Ri.values())
    
    # Wi 계산
    if Sum_R == 0:
        return {factor: 0.0 for factor in n_i.keys()}
    
    Wi = {factor: ri / Sum_R for factor, ri in Ri.items()}
    
    return Wi, Ri


def calculate_mu_sigma(fsi, weight):
    """
    종합 장소성 점수 (Mu)와 표준편차 (Sigma)를 계산합니다.
    
    Args:
        fsi: {factor_name: fsi_score}
        weight: {factor_name: weight}
    
    Returns:
        tuple: (mu, sigma)
    """
    # Mu 계산: 가중 평균
    mu = sum(fsi.get(factor, 0.5) * weight.get(factor, 0.0) for factor in fsi.keys())
    
    # Sigma 계산: 표준편차
    scores = [fsi.get(factor, 0.5) for factor in fsi.keys()]
    sigma = np.std(scores) if len(scores) > 0 else 0.0
    
    return mu, sigma


def extract_deviant_features(fsi, mu, n_i):
    """
    특이 특징 (df+, df-)를 추출합니다.
    
    Args:
        fsi: {factor_name: fsi_score}
        mu: 종합 장소성 점수
        n_i: {factor_name: mention_count}
    
    Returns:
        tuple: (strong_factors, weak_factors)
    """
    strong_factors = []
    weak_factors = []
    
    for factor_name, score in fsi.items():
        # 언급이 0인 요인은 제외
        if n_i.get(factor_name, 0) > 0:
            if score >= mu + DEVIATION_THRESHOLD:
                strong_factors.append(factor_name)
            elif score <= mu - DEVIATION_THRESHOLD:
                weak_factors.append(factor_name)
    
    return strong_factors, weak_factors


def calculate_place_scores(df_reviews, sbert_model, sentiment_pipeline, factor_defs, similarity_threshold=0.4, sentiment_model_name=""):
    """
    Sentence-BERT와 감성 분석을 사용하여 장소성 요인별 점수를 계산합니다.
    리뷰별 점수도 함께 반환합니다.
    """
    st.subheader("1. Sentence-BERT 임베딩 생성")
    
    # 1. 장소성 정의 문장 임베딩 (고정 벡터)
    factor_sentences = list(factor_defs.values())
    factor_names = list(factor_defs.keys())
    
    with st.spinner("장소성 요인 정의 임베딩 생성 중..."):
        factor_embeddings = sbert_model.encode(factor_sentences, convert_to_tensor=True, show_progress_bar=False)
    
    # 결과를 저장할 빈 리스트
    results_list = []
    review_scores_list = []  # 리뷰별 점수 저장
    
    # 카페별로 그룹화하여 처리
    cafe_groups = df_reviews.groupby('cafe_name')
    total_cafes = len(cafe_groups)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (cafe_name, group) in enumerate(cafe_groups):
        status_text.text(f"처리 중: {cafe_name} ({idx+1}/{total_cafes})")
        progress_bar.progress((idx + 1) / total_cafes)
        
        # 2. 개별 리뷰 임베딩
        review_texts = group['review_text'].astype(str).tolist()
        review_indices = group.index.tolist()
        
        with st.spinner(f"{cafe_name} 리뷰 임베딩 생성 중..."):
            # 요인 언급 추출
            mentions, similarity_matrix = extract_factor_mentions(
                review_texts, sbert_model, factor_embeddings, factor_names, similarity_threshold
            )
        
        # 4. 요인별 점수 집계
        cafe_scores = {'cafe_name': cafe_name}
        
        # 각 리뷰별로 요인 점수 계산
        for review_idx, (review_text, review_original_idx) in enumerate(zip(review_texts, review_indices)):
            review_factor_scores = {
                'review_index': review_original_idx,
                'cafe_name': cafe_name,
                'review_text': review_text
            }
            
            # 각 요인별로 반복
            for i, factor_name in enumerate(factor_names):
                similarity_score = similarity_matrix[review_idx, i]
                review_factor_scores[f'{factor_name}_유사도'] = similarity_score
                
                # 유사도 임계값 이상인 경우에만 점수 계산
                if similarity_score >= similarity_threshold:
                    # 해당 리뷰에 대한 감성 분석
                    try:
                        truncated_text = truncate_text_for_bert(review_text)
                        sentiment_result = sentiment_pipeline([truncated_text], truncation=True, max_length=512)[0]
                        label, positive_prob = process_sentiment_result(sentiment_result, sentiment_model_name)
                        
                        # 유사도와 감성 점수를 결합 (가중 평균)
                        combined_score = 0.6 * similarity_score + 0.4 * positive_prob
                        review_factor_scores[f'{factor_name}_점수'] = combined_score
                    except Exception as e:
                        review_factor_scores[f'{factor_name}_점수'] = np.nan
                else:
                    review_factor_scores[f'{factor_name}_점수'] = np.nan
            
            review_scores_list.append(review_factor_scores)
        
        # 각 요인별로 반복 (카페별 평균 점수 계산)
        for i, factor_name in enumerate(factor_names):
            # 4-1. 유사도 임계값 이상인 문장 선별
            relevant_review_indices = np.where(similarity_matrix[:, i] >= similarity_threshold)[0]
            
            if len(relevant_review_indices) > 0:
                relevant_texts = [review_texts[idx] for idx in relevant_review_indices]
                relevant_original_indices = [review_indices[idx] for idx in relevant_review_indices]
                
                # 4-2. 감성 분석 적용 (0~1 긍정 점수)
                try:
                    sentiment_scores = []
                    
                    # 숫자/메타데이터 리뷰와 일반 텍스트 리뷰 분리
                    text_batch = []
                    text_batch_indices = []
                    rating_scores = {}
                    
                    for batch_idx, text_idx in enumerate(relevant_review_indices):
                        text = review_texts[text_idx]
                        original_idx = review_indices[text_idx]
                        
                        # 평점 정보 추출
                        rating = None
                        if 'rating' in group.columns:
                            rating = group.loc[group.index == original_idx, 'rating'].values
                            rating = rating[0] if len(rating) > 0 else None
                        elif '평점' in df_reviews.columns:
                            rating = df_reviews.loc[original_idx, '평점'] if original_idx in df_reviews.index else None
                        
                        # 숫자-only 또는 메타데이터-only 리뷰는 평점 기반으로 처리
                        if is_numeric_only(text) or (is_metadata_only(text) and rating is not None and pd.notna(rating)):
                            try:
                                rating_val = float(text) if is_numeric_only(text) else float(rating)
                                if rating_val >= 4.0:
                                    score = 0.9
                                elif rating_val >= 3.0:
                                    score = 0.5
                                else:
                                    score = 0.1
                                rating_scores[batch_idx] = score
                            except (ValueError, TypeError):
                                rating_scores[batch_idx] = 0.5
                        else:
                            # 일반 텍스트는 배치로 모델에 전달
                            text_batch.append(text)
                            text_batch_indices.append(batch_idx)
                    
                    # 배치로 감성 분석 수행
                    if text_batch:
                        try:
                            truncated_batch = [truncate_text_for_bert(text) for text in text_batch]
                            sentiment_results = sentiment_pipeline(truncated_batch, truncation=True, max_length=512)
                            for batch_idx, result in zip(text_batch_indices, sentiment_results):
                                _, score = process_sentiment_result(result, sentiment_model_name)
                                rating_scores[batch_idx] = score
                        except Exception as e:
                            if idx == 0 and i == 0:
                                st.warning(f"감성 분석 배치 처리 예외, 개별 처리로 전환: {e}")
                            for batch_idx, text in zip(text_batch_indices, text_batch):
                                try:
                                    truncated_text = truncate_text_for_bert(text)
                                    result = sentiment_pipeline([truncated_text], truncation=True, max_length=512)[0]
                                    _, score = process_sentiment_result(result, sentiment_model_name)
                                    rating_scores[batch_idx] = score
                                except:
                                    rating_scores[batch_idx] = 0.5
                    
                    # 원래 순서대로 점수 수집
                    for batch_idx in range(len(relevant_review_indices)):
                        sentiment_scores.append(rating_scores.get(batch_idx, 0.5))
                    
                    # 4-3. 세부 항목 최종 점수 산출 (산술 평균)
                    if len(sentiment_scores) > 0:
                        avg_score = np.mean(sentiment_scores)
                        cafe_scores[f'점수_{factor_name}'] = avg_score
                        cafe_scores[f'리뷰수_{factor_name}'] = len(relevant_texts)
                    else:
                        cafe_scores[f'점수_{factor_name}'] = 0.5
                        cafe_scores[f'리뷰수_{factor_name}'] = len(relevant_texts)
                    
                except Exception as e:
                    st.warning(f"{cafe_name} - {factor_name} 감성 분석 오류: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    cafe_scores[f'점수_{factor_name}'] = np.nan
                    cafe_scores[f'리뷰수_{factor_name}'] = 0
            else:
                # 언급이 전혀 없는 경우: fsi=0.5, Wi=0 처리
                cafe_scores[f'점수_{factor_name}'] = 0.5
                cafe_scores[f'리뷰수_{factor_name}'] = 0
        
        results_list.append(cafe_scores)
    
    progress_bar.empty()
    status_text.empty()
    
    df_cafe_scores = pd.DataFrame(results_list)
    df_review_scores = pd.DataFrame(review_scores_list)
    
    return df_cafe_scores, df_review_scores


def calculate_final_research_metrics(df_scores: pd.DataFrame, factor_names: list, total_reviews: int):
    """
    연구 알고리즘의 핵심인 Wi, Mu, Sigma, Deviant Features를 계산합니다.
    
    Args:
        df_scores: calculate_place_scores에서 반환된 카페별 점수 데이터프레임
        factor_names: 장소성 요인 이름 리스트
        total_reviews: 전체 리뷰 수 (가중치 계산용)
    
    Returns:
        pd.DataFrame: 가중치, 종합 점수, 표준편차, 특이 특징이 포함된 최종 데이터프레임
    """
    df = df_scores.copy()
    
    # NaN 값을 0.5로 대체 (언급이 없는 요인은 0.5로 처리)
    for factor in factor_names:
        df[f'점수_{factor}'] = df[f'점수_{factor}'].fillna(0.5)
        df[f'리뷰수_{factor}'] = df[f'리뷰수_{factor}'].fillna(0)
    
    # --- 1. 가중치 (Wi) 계산 ---
    
    # 1-1. 요인별 언급 비율 (Ri) 계산: Ri = 유효 언급 문장 수 / 전체 리뷰 수
    Ri_cols = []
    for factor in factor_names:
        Ri_col = f'Ri_{factor}'
        df[Ri_col] = df[f'리뷰수_{factor}'].fillna(0) / total_reviews
        Ri_cols.append(Ri_col)
    
    # 1-2. 정규화 상수 (Sum_R) 계산: 모든 Ri의 합
    df['Sum_R'] = df[Ri_cols].sum(axis=1)
    
    # 1-3. 가중치 (Wi) 계산: Wi = Ri / Sum_R
    for factor in factor_names:
        # Sum_R이 0인 경우 (리뷰가 아예 없는 경우) 분모를 1로 처리하여 Wi = 0 처리
        df[f'Wi_{factor}'] = df[f'Ri_{factor}'] / df['Sum_R'].replace(0, 1)
    
    # --- 2. 종합 장소성 점수 (Mu) 계산 ---
    
    mu_scores = []
    for index, row in df.iterrows():
        weighted_sum = 0
        for factor in factor_names:
            fsi = row.get(f'점수_{factor}', 0.5)  # fsi (0.5로 대체됨)
            wi = row.get(f'Wi_{factor}', 0)
            weighted_sum += fsi * wi
        
        mu_scores.append(weighted_sum)
    
    df['종합_장소성_점수_Mu'] = mu_scores
    
    # --- 3. 특이 특징 (df+, df-) 추출 ---
    
    score_cols = [f'점수_{factor}' for factor in factor_names]
    
    # 3-1. 요인 점수의 표준편차 (Sigma) 계산
    df['요인_점수_표준편차_Sigma'] = df[score_cols].std(axis=1, skipna=False)
    
    # 3-2. 강점/약점 추출
    deviant_results = []
    for index, row in df.iterrows():
        mu = row['종합_장소성_점수_Mu']
        sigma = row['요인_점수_표준편차_Sigma']
        
        strong_factors = []
        weak_factors = []
        
        for factor in factor_names:
            score = row.get(f'점수_{factor}')
            
            # 언급이 0인 요인은 특이 특징 추출에서 제외
            if row[f'리뷰수_{factor}'] > 0:
                # 기준: 평균(mu) 대비 DEVIATION_THRESHOLD (0.05) 이상 차이
                if score >= mu + DEVIATION_THRESHOLD:
                    strong_factors.append(factor)
                elif score <= mu - DEVIATION_THRESHOLD:
                    weak_factors.append(factor)
        
        deviant_results.append({
            'cafe_name': row['cafe_name'],
            '강점_요인(+df+)': ', '.join(strong_factors) if strong_factors else 'N/A',
            '약점_요인(-df-)': ', '.join(weak_factors) if weak_factors else 'N/A',
            # 최종 연구 수식 형태: [mu ± sigma, +count, -count]
            'Final_PlaceScore_Summary': f"[{mu:.3f} ± {sigma:.3f}, +{len(strong_factors)}, -{len(weak_factors)}]"
        })
    
    df_deviant = pd.DataFrame(deviant_results)
    
    # 최종 결과 병합
    df_final = pd.merge(df, df_deviant, on='cafe_name', how='left')
    
    return df_final

