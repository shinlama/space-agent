from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


TARGET_PER_DISTRICT = 100
RANDOM_SEED = 42
DISTRICT_COLUMN = "시군구명"


def load_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV file using UTF-8 encoding with BOM first, then cp949 as a fallback.
    """
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def sample_by_district(df: pd.DataFrame, target: int, seed: int) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Sample up to `target` rows per district. Returns the concatenated dataframe
    and a dict containing districts that had fewer rows than requested.
    """
    insufficient: Dict[str, int] = {}
    sampled_frames = []

    for district, group in df.groupby(DISTRICT_COLUMN):
        if len(group) >= target:
            sampled = group.sample(n=target, random_state=seed)
        else:
            insufficient[district] = len(group)
            sampled = group.copy()
        sampled_frames.append(sampled)

    sampled_df = pd.concat(sampled_frames, ignore_index=True)
    sampled_df = sampled_df.sort_values([DISTRICT_COLUMN, "상호명"], na_position="last").reset_index(drop=True)
    return sampled_df, insufficient


def main(input_path: Path, output_path: Path, seed: int) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    df = load_csv(input_path)

    if DISTRICT_COLUMN not in df.columns:
        raise ValueError(f"'{DISTRICT_COLUMN}' 컬럼이 존재하지 않습니다. CSV 구조를 확인해주세요.")

    sampled_df, insufficient = sample_by_district(df, TARGET_PER_DISTRICT, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    total_districts = sampled_df[DISTRICT_COLUMN].nunique()
    print(f"✅ 샘플링 완료: {total_districts}개 행정구, 총 {len(sampled_df)}개 행.")
    if insufficient:
        print("⚠️ 일부 행정구는 요청한 수보다 적은 데이터가 존재합니다:")
        for district, count in insufficient.items():
            print(f"   - {district}: {count}개 (원본 전체 행 사용)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="행정구별로 100개씩 랜덤 추출하여 표본 CSV를 생성합니다.",
    )
    parser.add_argument(
        "--input",
        default="서울시_상권_카페빵.csv",
        type=Path,
        help="전체 카페 데이터 CSV 경로",
    )
    parser.add_argument(
        "--output",
        default="서울시_상권_카페빵_표본.csv",
        type=Path,
        help="생성할 표본 CSV 경로",
    )
    parser.add_argument(
        "--seed",
        default=RANDOM_SEED,
        type=int,
        help="무작위 시드 (재현성 확보용)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.output, args.seed)

