"""
실제 VLM(OpenAI vision) 기반 공간 이미지 분석 모듈
"""
from __future__ import annotations

import base64
import io
import json
import os
from hashlib import sha1
from pathlib import Path
from typing import Iterable, List, Optional

from openai import OpenAI

from modules.config import FACTOR_DEFINITIONS

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


VLM_FACTOR_ORDER = [
    "심미성",
    "형태성",
    "감각적 경험",
    "접근성",
    "쾌적성",
    "활동성",
    "사회성",
    "참여성",
    "고유성",
    "기억/경험",
    "지역 정체성",
]


def get_openai_client(streamlit_secrets=None) -> Optional[OpenAI]:
    """OPENAI_API_KEY를 찾아 OpenAI 클라이언트를 반환합니다."""
    api_key = None

    try:
        from dotenv import load_dotenv

        base_dir = Path(__file__).resolve().parent.parent
        env_path = base_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
            api_key = os.getenv("OPENAI_API_KEY")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key and streamlit_secrets is not None:
        try:
            api_key = streamlit_secrets.get("OPENAI_API_KEY", None)
        except Exception:
            api_key = None

    if not api_key:
        return None

    return OpenAI(api_key=api_key)


def _resize_for_vlm(image: Image.Image, max_side: int = 1600) -> Image.Image:
    """업로드 이미지를 VLM 입력용으로 적절히 축소합니다."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image

    scale = max_side / float(longest)
    resized = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)
    return resized


def image_bytes_to_data_url(raw_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """이미지 바이트를 data URL로 변환합니다."""
    if HAS_PIL:
        image = Image.open(io.BytesIO(raw_bytes))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")
        image = _resize_for_vlm(image)

        output = io.BytesIO()
        image.save(output, format="JPEG", quality=88, optimize=True)
        raw_bytes = output.getvalue()
        mime_type = "image/jpeg"

    encoded = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_vlm_fingerprint(cafe_name: str, uploaded_files, shot_types: List[str], analyst_note: str, model_name: str) -> str:
    """현재 분석 입력을 식별하는 fingerprint를 만듭니다."""
    hasher = sha1()
    hasher.update(cafe_name.encode("utf-8"))
    hasher.update(model_name.encode("utf-8"))
    hasher.update((analyst_note or "").encode("utf-8"))

    for idx, uploaded_file in enumerate(uploaded_files):
        hasher.update(uploaded_file.name.encode("utf-8"))
        hasher.update(uploaded_file.getvalue())
        if idx < len(shot_types):
            hasher.update((shot_types[idx] or "").encode("utf-8"))

    return hasher.hexdigest()


def _factor_definitions_text() -> str:
    lines = []
    for category, factor_map in FACTOR_DEFINITIONS.items():
        lines.append(f"[{category}]")
        for factor_name, definition in factor_map.items():
            lines.append(f"- {factor_name}: {definition}")
    return "\n".join(lines)


def _build_prompt(cafe_name: str, shot_types: List[str], analyst_note: str) -> str:
    factor_text = _factor_definitions_text()
    shot_lines = []
    for idx, shot_type in enumerate(shot_types, start=1):
        shot_lines.append(f"- 이미지 {idx}: {shot_type or '미분류'}")

    note_text = analyst_note.strip() if analyst_note and analyst_note.strip() else "없음"

    return f"""
당신은 카페 공간을 '장소성(placeness)' 관점에서 분석하는 멀티모달 연구 보조 모델입니다.
업로드된 이미지만 보고 판단하세요. 이미지에서 직접 보이지 않는 정보는 추정하지 마세요.

분석 대상 카페: {cafe_name}

이미지 메타데이터:
{chr(10).join(shot_lines) if shot_lines else "- 별도 메타데이터 없음"}

분석 메모:
{note_text}

장소성 요인 정의:
{factor_text}

평가 원칙:
1. 반드시 이미지에 직접 보이는 단서만 근거로 사용하세요.
2. 냄새, 소리, 실제 서비스 태도, 실제 접근 편의성처럼 이미지로 확인 불가능한 것은 단정하지 마세요.
3. 시각적으로 판단이 어려운 요인은 score를 0.50 전후로 두고 insufficient_visual_evidence를 true로 표시하세요.
4. score는 0.0~1.0 실수로 주세요.
5. confidence는 0.0~1.0 실수로 주세요.
6. evidence는 짧은 한국어 문장 1~3개로 주세요.
7. visible_cues는 시각적으로 보이는 명사구 위주로 2~6개 주세요.
8. 반드시 아래 11개 요인을 모두 반환하세요:
   심미성, 형태성, 감각적 경험, 접근성, 쾌적성, 활동성, 사회성, 참여성, 고유성, 기억/경험, 지역 정체성

반환 형식:
JSON만 반환하세요. Markdown 금지.

JSON 스키마:
{{
  "overall_summary": "2~4문장 요약",
  "limitations": ["한계 1", "한계 2"],
  "factors": [
    {{
      "name": "심미성",
      "score": 0.0,
      "confidence": 0.0,
      "insufficient_visual_evidence": false,
      "visible_cues": ["단서1", "단서2"],
      "evidence": ["근거1", "근거2"]
    }}
  ],
  "cross_image_observations": ["여러 이미지 종합 관찰 1", "여러 이미지 종합 관찰 2"],
  "recommended_3dgs_capture": ["3DGS를 위해 추가로 찍으면 좋은 장면 1", "장면 2"]
}}
""".strip()


def _vlm_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "placeness_vision_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "overall_summary": {"type": "string"},
                    "limitations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "factors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "score": {"type": "number"},
                                "confidence": {"type": "number"},
                                "insufficient_visual_evidence": {"type": "boolean"},
                                "visible_cues": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "evidence": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "name",
                                "score",
                                "confidence",
                                "insufficient_visual_evidence",
                                "visible_cues",
                                "evidence",
                            ],
                        },
                    },
                    "cross_image_observations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "recommended_3dgs_capture": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "overall_summary",
                    "limitations",
                    "factors",
                    "cross_image_observations",
                    "recommended_3dgs_capture",
                ],
            },
        },
    }


def _normalize_vlm_result(parsed: dict) -> dict:
    """모델 출력 JSON을 앱에서 쓰기 쉬운 형태로 정규화합니다."""
    factors = parsed.get("factors", [])
    factor_by_name = {item.get("name"): item for item in factors if isinstance(item, dict)}

    normalized_factors = []
    for factor_name in VLM_FACTOR_ORDER:
        item = factor_by_name.get(factor_name, {})
        score = item.get("score", 0.5)
        confidence = item.get("confidence", 0.3)
        insufficient = item.get("insufficient_visual_evidence", True)
        visible_cues = item.get("visible_cues", [])
        evidence = item.get("evidence", [])

        try:
            score = float(score)
        except Exception:
            score = 0.5
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.3

        normalized_factors.append({
            "name": factor_name,
            "score": max(0.0, min(1.0, score)),
            "confidence": max(0.0, min(1.0, confidence)),
            "insufficient_visual_evidence": bool(insufficient),
            "visible_cues": visible_cues if isinstance(visible_cues, list) else [],
            "evidence": evidence if isinstance(evidence, list) else [],
        })

    return {
        "overall_summary": parsed.get("overall_summary", ""),
        "limitations": parsed.get("limitations", []) if isinstance(parsed.get("limitations", []), list) else [],
        "cross_image_observations": parsed.get("cross_image_observations", []) if isinstance(parsed.get("cross_image_observations", []), list) else [],
        "recommended_3dgs_capture": parsed.get("recommended_3dgs_capture", []) if isinstance(parsed.get("recommended_3dgs_capture", []), list) else [],
        "factors": normalized_factors,
    }


def analyze_cafe_images_with_openai(
    cafe_name: str,
    uploaded_files,
    shot_types: List[str],
    analyst_note: str = "",
    model_name: str = "gpt-4o-mini",
    streamlit_secrets=None,
) -> dict:
    """업로드 이미지를 실제 OpenAI 비전 모델로 분석합니다."""
    client = get_openai_client(streamlit_secrets=streamlit_secrets)
    if client is None:
        raise RuntimeError("OPENAI_API_KEY를 찾지 못했습니다. .env 또는 Streamlit secrets를 확인해주세요.")

    if not uploaded_files:
        raise ValueError("분석할 이미지가 없습니다.")

    content_blocks = [{"type": "text", "text": _build_prompt(cafe_name, shot_types, analyst_note)}]

    for uploaded_file in uploaded_files[:10]:
        mime_type = uploaded_file.type or "image/jpeg"
        data_url = image_bytes_to_data_url(uploaded_file.getvalue(), mime_type=mime_type)
        content_blocks.append({
            "type": "image_url",
            "image_url": {
                "url": data_url,
                "detail": "high",
            },
        })

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "당신은 카페 공간의 장소성을 이미지로 분석하는 엄격한 시각 연구 보조 모델입니다. JSON만 반환하세요.",
            },
            {
                "role": "user",
                "content": content_blocks,
            },
        ],
        response_format=_vlm_response_format(),
        temperature=0.2,
        max_completion_tokens=3000,
    )

    raw_content = response.choices[0].message.content
    parsed = json.loads(raw_content)
    normalized = _normalize_vlm_result(parsed)
    normalized["model_name"] = model_name
    normalized["raw_json"] = parsed
    return normalized
