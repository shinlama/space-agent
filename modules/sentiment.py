"""
감성 분석 모듈
"""
import pandas as pd
import streamlit as st
from modules.preprocess import truncate_text_for_bert, is_numeric_only, is_metadata_only


def process_sentiment_result(result, model_name=""):
    """
    다양한 감성 분석 모델의 결과를 통일된 형식(긍정/부정, 점수)으로 변환합니다.
    
    Args:
        result: sentiment_pipeline의 결과 (dict 또는 list)
        model_name: 사용된 모델 이름 (선택적)
    
    Returns:
        tuple: (label: str, score: float) - '긍정'/'부정'/'중립', 0.0~1.0 점수
    """
    if isinstance(result, list):
        # 배치 결과인 경우 첫 번째 결과 사용
        result = result[0] if len(result) > 0 else {}
    
    label = str(result.get('label', '')).upper()
    score = float(result.get('score', 0.5))
    
    # nlptown 모델 처리 (5단계: 1-5점)
    if 'nlptown' in model_name.lower() or 'multilingual' in model_name.lower():
        # label 형식: "1 star", "2 stars", "3 stars", "4 stars", "5 stars"
        if '5' in label or 'FIVE' in label:
            return ('긍정', 0.9)
        elif '4' in label or 'FOUR' in label:
            return ('긍정', 0.7)
        elif '3' in label or 'THREE' in label:
            return ('중립', 0.5)
        elif '2' in label or 'TWO' in label:
            return ('부정', 0.3)
        elif '1' in label or 'ONE' in label:
            return ('부정', 0.1)
        else:
            # 점수 기반으로 판단
            if score >= 0.6:
                return ('긍정', score)
            elif score <= 0.4:
                return ('부정', 1 - score)
            else:
                return ('중립', 0.5)
    
    # 일반적인 2단계 모델 처리 (긍정/부정)
    if any(pos in label for pos in ['POSITIVE', '긍정', 'LABEL_1', '1', 'POS']):
        return ('긍정', score)
    elif any(neg in label for neg in ['NEGATIVE', '부정', 'LABEL_0', '0', 'NEG']):
        return ('부정', 1 - score)
    else:
        # 레이블을 알 수 없는 경우 점수로 판단
        if score >= 0.6:
            return ('긍정', score)
        elif score <= 0.4:
            return ('부정', 1 - score)
        else:
            return ('중립', 0.5)


def analyze_sentiment(reviews, sentiment_pipeline, model_name="", ratings=None):
    """
    리뷰 단위 감성 분석을 수행합니다.
    
    Args:
        reviews: 리뷰 텍스트 리스트
        sentiment_pipeline: 감성 분석 파이프라인
        model_name: 모델 이름
        ratings: 평점 리스트 (선택적, 메타데이터-only 리뷰 처리용)
    
    Returns:
        dict: {review_id: (label, score)}
    """
    results = {}
    
    for idx, text in enumerate(reviews):
        rating = ratings[idx] if ratings and idx < len(ratings) and ratings[idx] is not None else None
        
        # 숫자-only 리뷰는 별점 기반으로 처리
        if is_numeric_only(text):
            try:
                rating_value = float(text)
                if rating_value >= 4.0:
                    results[idx] = ("긍정", 0.9)
                elif rating_value >= 3.0:
                    results[idx] = ("중립", 0.5)
                else:
                    results[idx] = ("부정", 0.1)
            except ValueError:
                results[idx] = ("중립", 0.5)
        # 메타데이터-only 리뷰도 별점 기반으로 처리
        elif is_metadata_only(text) and rating is not None:
            try:
                rating_value = float(rating)
                if rating_value >= 4.0:
                    results[idx] = ("긍정", 0.9)
                elif rating_value >= 3.0:
                    results[idx] = ("중립", 0.5)
                else:
                    results[idx] = ("부정", 0.1)
            except (ValueError, TypeError):
                results[idx] = ("중립", 0.5)
        else:
            # 일반 텍스트 리뷰는 모델 사용
            try:
                truncated_text = truncate_text_for_bert(text)
                sentiment_result = sentiment_pipeline([truncated_text], truncation=True, max_length=512)[0]
                label, score = process_sentiment_result(sentiment_result, model_name)
                results[idx] = (label, score)
            except Exception as e:
                results[idx] = ("중립", 0.5)
    
    return results


def run_sentiment_analysis(df_reviews, sentiment_pipeline, model_name="", ratings=None):
    """
    개별 리뷰 텍스트에 대해 한국어 감성 분석 모델을 사용하여 감성 분석을 수행합니다.
    
    Args:
        df_reviews: 리뷰 데이터프레임
        sentiment_pipeline: 감성 분석 파이프라인
        model_name: 모델 이름
        ratings: 평점 리스트 (선택적, 메타데이터-only 리뷰 처리용)
    
    Returns:
        tuple: (df_reviews_with_sentiment, df_avg_sentiment)
    """
    st.subheader("2. 개별 리뷰 감성 분석 (한국어 감성 분석 모델)")
    
    review_texts = df_reviews['review_text'].astype(str).tolist()
    
    # 평점 정보 추출 (있으면 사용)
    if ratings is None:
        if '평점' in df_reviews.columns:
            ratings = df_reviews['평점'].astype(float).tolist()
        elif 'rating' in df_reviews.columns:
            ratings = df_reviews['rating'].astype(float).tolist()
        else:
            ratings = [None] * len(review_texts)
    
    # 진행 상황 표시
    progress_bar = st.progress(0)
    batch_size = 32
    total_batches = (len(review_texts) + batch_size - 1) // batch_size
    
    sentiment_scores = []
    sentiment_labels = []
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(review_texts))
        batch_texts = review_texts[start_idx:end_idx]
        
        progress_bar.progress((batch_idx + 1) / total_batches)
        
        try:
            # 숫자-only 리뷰와 일반 텍스트 리뷰 분리
            text_batch = []
            batch_results_map = {}  # 인덱스 -> 결과 매핑
            
            for idx, text in enumerate(batch_texts):
                global_idx = start_idx + idx
                rating = ratings[global_idx] if global_idx < len(ratings) and ratings[global_idx] is not None else None
                
                # 숫자-only 리뷰는 별점 기반으로 처리
                if is_numeric_only(text):
                    try:
                        rating_value = float(text)
                        if rating_value >= 4.0:
                            batch_results_map[idx] = ("긍정", 0.9)
                        elif rating_value >= 3.0:
                            batch_results_map[idx] = ("중립", 0.5)
                        else:
                            batch_results_map[idx] = ("부정", 0.1)
                    except ValueError:
                        # 숫자 변환 실패 시 중립 처리
                        batch_results_map[idx] = ("중립", 0.5)
                # 메타데이터-only 리뷰도 별점 기반으로 처리
                elif is_metadata_only(text) and rating is not None:
                    try:
                        rating_value = float(rating)
                        if rating_value >= 4.0:
                            batch_results_map[idx] = ("긍정", 0.9)
                        elif rating_value >= 3.0:
                            batch_results_map[idx] = ("중립", 0.5)
                        else:
                            batch_results_map[idx] = ("부정", 0.1)
                    except (ValueError, TypeError):
                        # 평점 변환 실패 시 중립 처리
                        batch_results_map[idx] = ("중립", 0.5)
                else:
                    # 일반 텍스트 리뷰는 모델 사용을 위해 수집
                    text_batch.append((idx, text))
            
            # 일반 텍스트 리뷰는 모델 사용
            if text_batch:
                text_only = [text for _, text in text_batch]
                # 텍스트 길이 제한 (BERT 512 토큰 제한 대응)
                truncated_texts = [truncate_text_for_bert(text) for text in text_only]
                model_results = sentiment_pipeline(truncated_texts, truncation=True, max_length=512)
                
                # 모델 결과를 인덱스에 매핑
                for (idx, _), res in zip(text_batch, model_results):
                    label, score = process_sentiment_result(res, model_name)
                    batch_results_map[idx] = (label, score)
            
            # 원래 순서대로 결과 추가
            for idx in range(len(batch_texts)):
                label, score = batch_results_map[idx]
                sentiment_labels.append(label)
                sentiment_scores.append(score)
            
        except Exception as e:
            st.warning(f"배치 {batch_idx+1} 처리 중 오류: {e}")
            # 오류 발생 시 중립 점수 할당
            sentiment_labels.extend(['중립'] * len(batch_texts))
            sentiment_scores.extend([0.5] * len(batch_texts))
    
    progress_bar.empty()
    
    # 리뷰 데이터프레임에 추가
    df_reviews = df_reviews.copy()
    df_reviews['sentiment_score'] = sentiment_scores
    df_reviews['sentiment_label'] = sentiment_labels
    
    # 카페별 평균 감성 점수 산출
    avg_sentiment = df_reviews.groupby('cafe_name')['sentiment_score'].mean().reset_index()
    avg_sentiment.rename(columns={'sentiment_score': '평균_리뷰_감성점수'}, inplace=True)
    
    st.success("개별 리뷰 감성 분석 및 카페별 평균 산출 완료.")
    return df_reviews, avg_sentiment

