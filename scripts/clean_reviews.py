"""
구글 리뷰 CSV에서 UI 메타데이터를 제거하는 스크립트
- "지역 가이드", "리뷰 N개", "사진 N장" 제거
- "음식: 5서비스: 5분위기: 5" 같은 상세 평점은 유지
"""

import re
import pandas as pd
from pathlib import Path


def clean_review_text(text: str, author: str = "") -> str:
    """
    리뷰 텍스트에서 UI 메타데이터를 제거합니다.
    
    Args:
        text: 원본 리뷰 텍스트
        author: 작성자 이름 (중복 제거용)
    
    Returns:
        정제된 리뷰 텍스트
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    cleaned = text.strip()
    
    # 1. 작성자 이름이 리뷰 시작 부분에 중복되는 경우 제거
    if author and author.strip():
        author_escaped = re.escape(author.strip())
        # 작성자 이름으로 시작하는 경우 제거 (공백 없이 붙어있는 경우 포함)
        cleaned = re.sub(rf'^{author_escaped}(?=\s|지역|리뷰|사진|·)', '', cleaned, flags=re.IGNORECASE)
    
    # 2. "지역 가이드" 제거 (앞뒤 공백/구분자 포함)
    cleaned = re.sub(r'지역\s*가이드\s*·?\s*', '', cleaned, flags=re.IGNORECASE)
    
    # 3. "리뷰 N개" 제거 (숫자 포함, 콤마 포함, 앞뒤 구분자 포함)
    # 패턴: "리뷰 1개", "리뷰 1,370개", "리뷰1개" 등
    # 모든 위치에서 제거 (반복 적용)
    for _ in range(3):  # 여러 번 반복하여 모든 패턴 제거
        cleaned = re.sub(r'·?\s*리뷰\s*\d+[,\.]?\d*\s*개\s*·?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'리뷰\s*\d+[,\.]?\d*\s*개\s*·?\s*', '', cleaned, flags=re.IGNORECASE)
        # 특수 문자나 다른 텍스트 앞에 오는 경우도 제거
        cleaned = re.sub(r'리뷰\s*\d+[,\.]?\d*\s*개(?=[^\d\s]|$)', '', cleaned, flags=re.IGNORECASE)
    
    # 4. "사진 N장" 제거 (숫자 포함, 콤마 포함, 앞뒤 구분자 포함)
    # 패턴: "사진 34장", "사진 7,920장", "사진34장" 등
    # 모든 위치에서 제거 (반복 적용)
    for _ in range(3):  # 여러 번 반복하여 모든 패턴 제거
        cleaned = re.sub(r'·?\s*사진\s*\d+[,\.]?\d*\s*장\s*·?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'사진\s*\d+[,\.]?\d*\s*장\s*·?\s*', '', cleaned, flags=re.IGNORECASE)
        # 특수 문자나 다른 텍스트 앞에 오는 경우도 제거
        cleaned = re.sub(r'사진\s*\d+[,\.]?\d*\s*장(?=[^\d\s]|$)', '', cleaned, flags=re.IGNORECASE)
    
    # 5. 메타데이터 구분자 "·" 제거 (단독으로 있는 경우 또는 앞뒤 공백과 함께)
    cleaned = re.sub(r'\s*·\s*', ' ', cleaned)
    
    # 6. 특수 문자 패턴 제거 (리뷰, 사진 등 UI 관련 단독 패턴)
    # 단, "음식:", "서비스:", "분위기:" 같은 상세 평점은 유지
    cleaned = re.sub(r'^리뷰\s*\d+[,\.]?\d*\s*개\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^사진\s*\d+[,\.]?\d*\s*장\s*', '', cleaned, flags=re.IGNORECASE)
    
    # 7. 특수 문자 패턴 제거 (리뷰 같은 반복되는 특수 문자)
    # 예: "리뷰리뷰리뷰리뷰리뷰" 같은 패턴
    # 특수 문자 범위: 리뷰, 사진 등 UI 관련 특수 문자
    cleaned = re.sub(r'[리뷰]{2,}', '', cleaned)
    cleaned = re.sub(r'[사진]{2,}', '', cleaned)
    
    # 8. 특수 유니코드 문자 제거 (리뷰 같은 UI 아이콘 문자)
    # 범위: U+E000-U+F8FF (Private Use Area) 및 기타 특수 문자
    cleaned = re.sub(r'[\ue000-\uf8ff]', '', cleaned)
    
    # 9. 날짜 패턴 제거
    # "N년 전", "N개월 전", "N주 전", "N일 전" 등
    # "수정일: N년 전" 같은 패턴도 제거
    cleaned = re.sub(r'수정일:\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\d+\s*년\s*전', '', cleaned)
    cleaned = re.sub(r'\d+\s*개월\s*전', '', cleaned)
    cleaned = re.sub(r'\d+\s*주\s*전', '', cleaned)
    cleaned = re.sub(r'\d+\s*일\s*전', '', cleaned)
    cleaned = re.sub(r'\d+\s*시간\s*전', '', cleaned)
    cleaned = re.sub(r'\d+\s*분\s*전', '', cleaned)
    
    # 10. 작성자 이름 뒤에 붙어있는 날짜 제거 (예: "박경섭2년 전")
    if author and author.strip():
        author_escaped = re.escape(author.strip())
        # 작성자 이름 뒤에 날짜가 바로 붙어있는 경우
        cleaned = re.sub(rf'{author_escaped}\s*\d+\s*(년|개월|주|일|시간|분)\s*전', '', cleaned, flags=re.IGNORECASE)
    
    # 11. 연속된 공백 정리
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # 8. 앞뒤 공백 제거
    cleaned = cleaned.strip()
    
    return cleaned


def main():
    input_file = Path("google_reviews_scraped.csv")
    output_file = Path("google_reviews_scraped_cleaned.csv")
    
    if not input_file.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
        return
    
    import sys
    import io
    
    # Windows 콘솔 인코딩 설정
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print(f"CSV 파일 로드 중: {input_file}")
    df = pd.read_csv(input_file, encoding="utf-8-sig")
    print(f"총 {len(df)}개 리뷰 로드 완료")
    
    # 리뷰 컬럼이 있는지 확인
    if "리뷰" not in df.columns:
        print("'리뷰' 컬럼을 찾을 수 없습니다.")
        return
    
    # 작성자 컬럼 확인 (있으면 사용)
    author_col = "작성자" if "작성자" in df.columns else None
    
    print("리뷰 텍스트 정제 중...")
    cleaned_reviews = []
    
    for idx, row in df.iterrows():
        review_text = row["리뷰"]
        author = str(row[author_col]) if author_col and pd.notna(row[author_col]) else ""
        
        cleaned = clean_review_text(review_text, author)
        cleaned_reviews.append(cleaned)
        
        if (idx + 1) % 1000 == 0:
            print(f"  진행 중: {idx + 1}/{len(df)}")
    
    # 정제된 리뷰로 업데이트
    df["리뷰"] = cleaned_reviews
    
    # 빈 리뷰 확인 및 평점으로 채우기
    empty_mask = df["리뷰"].str.strip() == ""
    empty_count = empty_mask.sum()
    
    # 작성자 이름만 남은 리뷰 확인
    author_only_mask = pd.Series([False] * len(df))
    if "평점" in df.columns:
        for idx, row in df.iterrows():
            review_text = str(row["리뷰"]).strip()
            author_name = ""
            if author_col and pd.notna(row[author_col]):
                author_name = str(row[author_col]).strip()
            
            # 리뷰가 작성자 이름과 동일하거나 작성자 이름으로만 구성된 경우
            if review_text and author_name and review_text == author_name:
                author_only_mask[idx] = True
            # 작성자 컬럼이 비어있고 리뷰가 매우 짧은 경우 (10자 이하, 평점처럼 보이지 않는 경우)
            elif review_text and not author_name and len(review_text) <= 10 and not review_text.replace(".", "").isdigit():
                # 평점 형식(숫자)이 아닌 짧은 텍스트는 작성자 이름일 가능성이 높음
                author_only_mask[idx] = True
    
    author_only_count = author_only_mask.sum()
    
    # 빈 리뷰 + 작성자 이름만 있는 리뷰를 평점으로 채우기
    fill_mask = empty_mask | author_only_mask
    fill_count = fill_mask.sum()
    
    if fill_count > 0:
        if empty_count > 0:
            print(f"빈 리뷰 {empty_count}개 발견 (메타데이터만 있었던 경우)")
        if author_only_count > 0:
            print(f"작성자 이름만 있는 리뷰 {author_only_count}개 발견")
        
        # 평점 컬럼이 있으면 빈 리뷰와 작성자 이름만 있는 리뷰에 평점 복사
        if "평점" in df.columns:
            # 빈 리뷰와 작성자 이름만 있는 리뷰에 평점 값 복사
            df.loc[fill_mask, "리뷰"] = df.loc[fill_mask, "평점"].astype(str)
            print(f"총 {fill_count}개 리뷰에 평점 값 복사 완료")
        else:
            print("'평점' 컬럼을 찾을 수 없어 리뷰를 채울 수 없습니다.")
    
    print(f"정제된 CSV 저장 중: {output_file}")
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {output_file}")
    
    # 최종 통계 (평점 복사 후)
    final_empty_count = (df["리뷰"].str.strip() == "").sum()
    print(f"통계:")
    print(f"   - 총 리뷰: {len(df)}개")
    print(f"   - 빈 리뷰 (정제 전): {empty_count}개")
    print(f"   - 빈 리뷰 (평점 복사 후): {final_empty_count}개")
    print(f"   - 유효 리뷰: {len(df) - final_empty_count}개")


if __name__ == "__main__":
    main()

