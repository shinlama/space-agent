from __future__ import annotations

import importlib
import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.research_scoring import (
    DEFAULT_FACTOR_SCORES_PARQUET,
    DEFAULT_MAPPING_CSV,
    DEFAULT_PLACE_SCORES_PARQUET,
    DEFAULT_SCORED_EVIDENCE_PARQUET,
    FACTOR_ORDER,
    complete_factor_table,
    compute_factor_scores,
    score_mapped_evidence,
)


class ResearchDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = importlib.import_module("app")

    def assert_no_exceptions(self, app: AppTest) -> None:
        self.assertFalse(app.exception, [item.message for item in app.exception])

    def test_korean_mapping_and_scoring_views(self) -> None:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
        self.assert_no_exceptions(app)
        self.assertEqual(app.selectbox(key="sidebar_district").value, "종로구")
        self.assertEqual(app.radio[0].options, ["평가 체계", "리뷰 매핑", "점수 계산"])
        self.assertIn(
            "매핑 모델: gpt-5.4-nano (v6)", [item.value for item in app.caption]
        )

        app.radio[0].set_value("리뷰 매핑").run()
        self.assert_no_exceptions(app)
        self.assertEqual(list(app.dataframe[0].value.columns), ["근거 구절", "매핑 요인"])

        app.radio[0].set_value("점수 계산").run()
        self.assert_no_exceptions(app)
        self.assertEqual(app.selectbox(key="sidebar_district").value, "종로구")
        scores = app.dataframe[0].value
        self.assertEqual(len(scores), len(FACTOR_ORDER))
        self.assertIn("장소성 요인 점수", scores)
        self.assertNotIn("언급 비중", scores)
        self.assertNotIn("가중 점수", scores)
        self.assertEqual(
            [item.label for item in app.metric],
            ["매핑된 리뷰 수", "매핑된 구절 수", "매핑된 요인 수"],
        )
        self.assertEqual(
            list(app.dataframe[1].value.columns), ["근거 구절", "평가 방향", "구절별 점수"]
        )
        rendered = "\n".join(item.value for item in app.markdown)
        self.assertNotIn("언급비중", rendered)
        self.assertNotIn("언급 비중", rendered)
        self.assertNotIn("점수 범위는 -1에서 +1입니다.", rendered)

        app.selectbox(key="sidebar_district").set_value("강남구").run()
        self.assert_no_exceptions(app)

    def test_confirmed_gpt54nano_v6_dataset(self) -> None:
        self.assertEqual(DEFAULT_MAPPING_CSV.parent.name, "openai_batch_full_gpt54nano_v6")
        for path in [
            DEFAULT_SCORED_EVIDENCE_PARQUET,
            DEFAULT_FACTOR_SCORES_PARQUET,
            DEFAULT_PLACE_SCORES_PARQUET,
        ]:
            self.assertEqual(path.parent.name, "gpt54nano_v6")

        summary = json.loads(
            (DEFAULT_SCORED_EVIDENCE_PARQUET.parent / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["model"], "gpt-5.4-nano")
        evidence = pd.read_parquet(
            DEFAULT_SCORED_EVIDENCE_PARQUET,
            columns=["review_index", "place_id", "factor"],
        )
        self.assertEqual(len(evidence), 648463)
        self.assertEqual(evidence["review_index"].nunique(), 263888)
        self.assertEqual(evidence["place_id"].nunique(), 10978)
        self.assertEqual(summary["mapping_rows"], len(evidence))
        self.assertEqual(summary["mapped_reviews"], evidence["review_index"].nunique())
        self.assertEqual(summary["mapped_places"], evidence["place_id"].nunique())
        self.assertEqual(summary["factor_counts"], evidence["factor"].value_counts().to_dict())

        sample = pd.read_parquet(
            DEFAULT_SCORED_EVIDENCE_PARQUET,
            columns=["evidence", "factor", "sentiment_key"],
            filters=[("review_index", "=", 288824)],
        )
        expected = [
            ("주차공간 꽤 넓습니다.", "접근성", "positive"),
            ("신기하게 주문하면 가져다 주시고 치워주시네요.", "상호작용성", "positive"),
            ("화장실도 깔끔하고 관리 잘 되어 있습니다.", "쾌적성", "positive"),
            ("전체적인 인테리어는 노출콘트리트~?", "심미성", "neutral"),
            ("넓고 괜찮긴 합니다.", "개방성", "positive"),
            ("의자가 나무로 되어 있어 조금 불편하구요.", "쾌적성", "negative"),
            ("좌식으로 앉는 곳도 생각보다 많이 불편하더라구요.", "쾌적성", "negative"),
            ("조명이 어두워서 2층 올라가는 계단 조심하셔야 합니다.", "쾌적성", "negative"),
            ("재방문 의사는 있습니다.", "기억 및 선호", "positive"),
        ]
        self.assertCountEqual(list(sample.itertuples(index=False, name=None)), expected)

    def test_factor_scores_are_unweighted_phrase_means(self) -> None:
        evidence = pd.read_parquet(
            DEFAULT_SCORED_EVIDENCE_PARQUET,
            columns=["place_id", "factor", "sentiment_value"],
        )
        expected = evidence.groupby(["place_id", "factor"])["sentiment_value"].mean()
        actual = pd.read_parquet(
            DEFAULT_FACTOR_SCORES_PARQUET, columns=["place_id", "factor", "factor_score"]
        ).set_index(["place_id", "factor"])["factor_score"]
        pd.testing.assert_series_equal(
            actual.sort_index(), expected.sort_index(), check_names=False, check_dtype=False
        )

    def test_neutral_mixed_and_missing_factors(self) -> None:
        mapping = pd.DataFrame([
            {
                "place_id": "sample", "cafe_name": "sample", "source_cafe_name": "sample",
                "district": "sample", "neighborhood": "sample", "address": "sample",
                "factor": "개방성", "sentiment_hint": label, "confidence": 1.0,
                "evidence": label, "sentence": label, "reason": "", "review_text": label,
            }
            for label in ["positive", "negative", "neutral", "mixed"]
        ])
        factors = compute_factor_scores(score_mapped_evidence(mapping))
        table = complete_factor_table(factors, "sample").set_index("factor")
        self.assertEqual(table.loc["개방성", "factor_score"], 0.0)
        self.assertEqual(table.loc["개방성", "mention_count"], 4)
        self.assertTrue(pd.isna(table.loc["지역 정체성", "factor_score"]))
        self.assertEqual(table.loc["지역 정체성", "mention_count"], 0)

        source = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import app
import pandas as pd
factors = pd.DataFrame({factors.to_dict(orient='records')!r})
phrases = pd.DataFrame([
    {{'cafe_name': 'sample', 'review_index': 1, 'factor': '개방성', 'evidence': 'open', 'sentiment_label': '중립', 'sentiment_value': 0.0}},
    {{'cafe_name': 'other', 'review_index': 2, 'factor': '개방성', 'evidence': 'other', 'sentiment_label': '긍정', 'sentiment_value': 1.0}},
])
places = pd.DataFrame([{{'cafe_name': 'sample', 'mapped_evidence_count': 4, 'mentioned_factor_count': 1}}])
app.render_score_results(phrases, factors, places, 'sample')
"""
        ascii_source = source.encode("ascii", "backslashreplace").decode("ascii")
        app = AppTest.from_string(ascii_source, default_timeout=30).run()
        self.assert_no_exceptions(app)
        rendered_table = app.dataframe[0].value.set_index("요인")
        self.assertEqual(rendered_table.loc["개방성", "장소성 요인 점수"], "+0.000")
        self.assertEqual(rendered_table.loc["지역 정체성", "장소성 요인 점수"], "-")
        self.assertEqual(app.metric[0].value, "1")
        plot = json.loads(app.get("plotly_chart")[0].proto.spec)
        physical_scores = plot["data"][0]["y"]
        if isinstance(physical_scores, dict):
            physical_scores = np.frombuffer(
                base64.b64decode(physical_scores["bdata"]), dtype=physical_scores["dtype"]
            )
        self.assertTrue(pd.isna(physical_scores[0]))
        self.assertEqual(physical_scores[1], 0.0)
        app.selectbox(key="scoring_factor").set_value("지역 정체성").run()
        self.assert_no_exceptions(app)
        self.assertTrue(any("점수를 산출하지 않았습니다" in item.value for item in app.info))

    def test_loader_discards_legacy_weighted_columns(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            factor_path, place_path = Path(folder)/"factors.parquet", Path(folder)/"places.parquet"
            pd.DataFrame([{
                "factor_score": 0.75, "mention_share": 0.01, "weighted_score": 0.0075,
            }]).to_parquet(factor_path)
            pd.DataFrame([{
                "mapped_evidence_count": 4, "placeness_score": 0.0075, "mean_factor_score": 0.75,
            }]).to_parquet(place_path)
            factors, places = self.app.load_demo_data.__wrapped__(str(factor_path), str(place_path))
            self.assertEqual(list(factors.columns), ["factor_score"])
            self.assertEqual(list(places.columns), ["mapped_evidence_count"])


if __name__ == "__main__":
    unittest.main()
