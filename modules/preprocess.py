"""
데이터 전처리 모듈
"""
import streamlit as st
import pandas as pd
import re
from pathlib import Path


def truncate_text_for_bert(text: str, max_chars: int = 2000) -> str:
    """
    BERT 모델의 토큰 제한(512)을 고려하여 텍스트를 자릅니다.
    대략적으로 2000자 = 512 토큰 정도로 가정합니다.
    
    Args:
        text: 원본 텍스트
        max_chars: 최대 문자 수 (기본값 2000)
    
    Returns:
        str: 잘린 텍스트
    """
    if text is None:
        return ""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    # 공백이나 문장 경계에서 자르기
    truncated = text[:max_chars]
    # 마지막 공백이나 문장 부호에서 자르기
    last_space = truncated.rfind(' ')
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    cut_point = max(last_space, last_period, last_newline)
    if cut_point > max_chars * 0.8:  # 너무 앞에서 자르지 않도록
        return truncated[:cut_point + 1]
    return truncated


def is_numeric_only(text: str) -> bool:
    """
    텍스트가 숫자만 포함되어 있는지 확인합니다.
    
    Args:
        text: 확인할 텍스트
    
    Returns:
        bool: 숫자만 포함되어 있으면 True
    """
    if text is None:
        return False
    text = str(text).strip()
    return bool(re.fullmatch(r"[0-9]+(\.[0-9]+)?", text))


def is_metadata_only(text: str) -> bool:
    """
    텍스트가 메타데이터만 포함되어 있는지 확인합니다.
    (예: "서비스매장 내 식사식사 유형아침 식사", "식사 유형브런치" 등)
    
    Args:
        text: 확인할 텍스트
    
    Returns:
        bool: 메타데이터만 포함되어 있으면 True
    """
    if text is None:
        return False
    text = str(text).strip()
    
    # 메타데이터 패턴들
    metadata_patterns = [
        r'^서비스.*식사.*유형',
        r'^식사.*유형',
        r'^서비스.*매장.*내.*식사',
        r'^음식:\s*\d+.*서비스:\s*\d+.*분위기:\s*\d+$',  # "음식: 5서비스: 5분위기: 5" 같은 패턴
        r'^음식:\s*\d+$',  # "음식: 5" 같은 패턴
        r'^서비스:\s*\d+$',
        r'^분위기:\s*\d+$',
    ]
    
    for pattern in metadata_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    
    return False


@st.cache_data
def load_csv_raw(file_path: Path):
    """
    CSV 파일을 원본 형태로 로드합니다 (컬럼명 정규화 없음).
    미리보기용으로 사용됩니다.
    """
    try:
        df = pd.read_csv(
            file_path, 
            encoding="utf-8-sig",
            on_bad_lines='skip',
            quoting=1,
            escapechar='\\'
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            file_path, 
            encoding="cp949",
            on_bad_lines='skip',
            quoting=1,
            escapechar='\\'
        )
    except Exception as e:
        try:
            df = pd.read_csv(
                file_path, 
                encoding="utf-8-sig",
                on_bad_lines='skip',
                engine='python'
            )
        except:
            df = pd.read_csv(
                file_path, 
                encoding="utf-8-sig",
                on_bad_lines='skip',
                sep=',',
                quotechar='"',
                escapechar='\\',
                engine='python'
            )
    return df


@st.cache_data
def load_data(file_path: Path):
    """리뷰 데이터를 로드하고 전처리합니다."""
    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    # CSV 파일 로드 (원본 형태)
    df = load_csv_raw(file_path)
    
    # 컬럼명 정규화 (한국어 컬럼명 처리)
    column_mapping = {
        "상호명": "cafe_name",
        "리뷰": "review_text",
        "cafe_name": "cafe_name",
        "review_text": "review_text"
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # 필요한 컬럼 확인
    if "cafe_name" not in df.columns or "review_text" not in df.columns:
        st.error(f"필수 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")
        st.stop()
    
    # 결측치 제거 전 상태 확인
    initial_count = len(df)
    initial_cafe_count = df['cafe_name'].nunique() if 'cafe_name' in df.columns else 0
    
    # 결측치 제거 (카페명과 리뷰 모두 있는 행만 유지)
    df = df[['cafe_name', 'review_text']].dropna()
    after_dropna_count = len(df)
    after_dropna_cafe_count = df['cafe_name'].nunique() if 'cafe_name' in df.columns else 0
    
    # 빈 리뷰 제거 (하지만 카페는 유지 - 빈 리뷰만 있는 카페도 포함)
    # 빈 리뷰 행만 제거하고, 카페별로 최소 1개 리뷰가 있도록 보장
    df_valid_reviews = df[df['review_text'].astype(str).str.strip() != '']
    final_count = len(df_valid_reviews)
    final_cafe_count = df_valid_reviews['cafe_name'].nunique() if 'cafe_name' in df_valid_reviews.columns else 0
    
    # 빈 리뷰만 있는 카페 확인
    cafes_with_valid = set(df_valid_reviews['cafe_name'].unique())
    cafes_with_empty_only = set(df[df['review_text'].astype(str).str.strip() == '']['cafe_name'].unique()) - cafes_with_valid
    cafes_with_empty_only_count = len(cafes_with_empty_only)
    
    st.success(f"리뷰 데이터 로드 완료: 총 {final_count}건")
    
    if initial_cafe_count > final_cafe_count:
        excluded = initial_cafe_count - final_cafe_count
        st.warning(f"⚠️ {excluded}개 카페가 빈 리뷰만 있어서 제외되었습니다. (유효 리뷰가 있는 카페만 분석에 포함됩니다)")
    
    if cafes_with_empty_only_count > 0:
        st.info(f"ℹ️ {cafes_with_empty_only_count}개 카페는 빈 리뷰만 있어서 해당 카페의 리뷰는 분석에서 제외되지만, 카페 자체는 카운트에 포함됩니다.")
    
    return df_valid_reviews

