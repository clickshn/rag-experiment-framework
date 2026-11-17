"""구성 읽기·검증 시험: YAML 함정, 스키마, level·부품, ${VAR}·주소·주석, 값 누출."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import CONFIGS, ENV, SECRET_VALUES

from ragexp.config import (
    ConfigError,
    is_connection_field,
    load_config,
    parse_yaml,
    scan_raw_text,
)
from ragexp.registry import Registry

BASE_TEXT = (CONFIGS / "base.yaml").read_text(encoding="utf-8")
PROMPT = (CONFIGS / "prompt.md").read_text(encoding="utf-8")
QUESTIONS = (CONFIGS / "questions.jsonl").read_text(encoding="utf-8")


def base_data() -> dict[str, Any]:
    data = yaml.safe_load(BASE_TEXT)
    assert isinstance(data, dict)
    return data


def write(tmp_path: Path, config: dict[str, Any] | str) -> Path:
    text = config if isinstance(config, str) else yaml.safe_dump(config, allow_unicode=True)
    (tmp_path / "prompt.md").write_text(PROMPT, encoding="utf-8")
    (tmp_path / "questions.jsonl").write_text(QUESTIONS, encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_bytes(text.encode("utf-8"))
    return path


def errors_of(tmp_path: Path, reg: Registry, config: dict[str, Any] | str, **kw: Any) -> str:
    with pytest.raises(ConfigError) as info:
        load_config(write(tmp_path, config), reg, env=kw.pop("env", ENV), **kw)
    msg = str(info.value)
    for secret in SECRET_VALUES:
        assert secret not in msg  # 기준 9: 오류 메시지에 치환·리터럴 값이 없다
    return msg


def mutate(fn: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    data = base_data()
    fn(data)
    return data


# --- 정상 ---------------------------------------------------------------------------


def test_base_loads(tmp_path: Path, fake_registry: Registry) -> None:
    loaded = load_config(write(tmp_path, BASE_TEXT), fake_registry, env=ENV)
    assert loaded.level == "generation_only" and list(loaded.arms) == ["A", "B", "C"]
    assert [(j.judge_id, j.role) for j in loaded.judges] == [
        ("fake_rules", "rules"),
        ("judge_main", "primary"),
        ("judge_side", "secondary"),
    ]
    a = loaded.arms["A"]
    assert a.run.concurrency == 8 and a.run.timeout_s == 180 and a.run.max_retries == 2
    assert loaded.arms["B"].run.concurrency == 4  # 기본값
    assert a.meta is not None and a.meta.price_per_1m_input == 1.5
    assert loaded.run.invalid_ratio_max == 0.05
    assert a.generator.cfg.prompt_template == PROMPT
    assert a.generator.cfg.base_url.get_secret_value() == ENV["GEN_A_URL"]
    assert loaded.warnings == []


def test_raw_text_is_preserved_byte_for_byte(tmp_path: Path, fake_registry: Registry) -> None:
    text = "# 주석은 그대로 남는다\n" + BASE_TEXT.replace("\n", "\r\n")
    path = write(tmp_path, text)
    loaded = load_config(path, fake_registry, env=ENV)
    assert loaded.raw_text == text
    assert loaded.raw["generators"]["A"]["base_url"] == "${GEN_A_URL}"  # 치환 전 원문


def test_design_example_passes(fake_registry: Registry) -> None:
    loaded = load_config(CONFIGS / "design_example.yaml", fake_registry, env=ENV)
    assert set(loaded.arms) == {"A", "B"}
    assert loaded.arms["B"].meta is not None and loaded.arms["B"].meta.license == "Apache-2.0"


def test_reprs_do_not_leak(tmp_path: Path, fake_registry: Registry) -> None:
    loaded = load_config(write(tmp_path, BASE_TEXT), fake_registry, env=ENV)
    texts = [repr(loaded), repr(loaded.arms["A"]), repr(loaded.arms["A"].generator.cfg)]
    texts += [repr(j) for j in loaded.judges] + [repr(j.component.cfg) for j in loaded.judges[1:]]
    for text in texts:
        for secret in SECRET_VALUES:
            assert secret not in text


# --- level·부품(기준 5) -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("level", "msg"), [("full", "R2 이후"), ("retrieval_only", "R2 이후"), ("x", "모르는 값")]
)
def test_levels(level: str, msg: str, tmp_path: Path, fake_registry: Registry) -> None:
    assert msg in errors_of(tmp_path, fake_registry, mutate(lambda d: d.update(level=level)))


def test_unused_part_warns_and_is_not_built(tmp_path: Path, fake_registry: Registry) -> None:
    # 안 쓰는 부품의 ${VAR}는 정의 여부를 확인하지 않는다(NOT_SET은 환경에 없음).
    data = mutate(lambda d: d.update(embedder={"type": "fake_embedder", "base_url": "${NOT_SET}"}))
    loaded = load_config(write(tmp_path, data), fake_registry, env=ENV)  # 만들면 AssertionError
    assert any("embedder" in w for w in loaded.warnings)
    assert "embedder" not in loaded.parts


def test_unused_part_unknown_type_is_error(tmp_path: Path, fake_registry: Registry) -> None:
    data = mutate(lambda d: d.update(reranker={"type": "no_such"}))
    assert "reranker.type" in errors_of(tmp_path, fake_registry, data)


def test_missing_used_part(tmp_path: Path, fake_registry: Registry) -> None:
    assert "chunker" in errors_of(tmp_path, fake_registry, mutate(lambda d: d.pop("chunker")))


def test_retriever_must_need_gold(tmp_path: Path, fake_registry: Registry) -> None:
    data = mutate(lambda d: d.update(retriever={"type": "fake_vector"}))
    assert "needs_gold" in errors_of(tmp_path, fake_registry, data)


def test_unknown_component_type(tmp_path: Path, fake_registry: Registry) -> None:
    data = mutate(lambda d: d["generators"]["A"].update(type="fake_chatt"))
    msg = errors_of(tmp_path, fake_registry, data)
    assert "generators.A.type" in msg and "fake_chat" in msg  # 비슷한 이름 제안


# --- ${VAR}·주소·주석(기준 6) ------------------------------------------------------------

BAD_VALUES: list[tuple[str, Callable[[dict[str, Any]], Any], str]] = [
    (
        "접속 칸 아닌 곳의 ${VAR}",
        lambda d: d["generators"]["A"].update(model="${MODEL}"),
        "generators.A.model",
    ),
    (
        "params 속 ${VAR}",
        lambda d: d["generators"]["A"]["params"].update(seed="${SEED}"),
        "generators.A.params.seed",
    ),
    (
        "접속 칸 리터럴 키(원문 검사가 줄 번호로 잡음)",
        lambda d: d["generators"]["A"].update(api_key="sk-test-not-real"),
        "접속 칸 값이 직접 적혀 있습니다",
    ),
    (
        "접속 칸 리터럴 짧은 값(값 검사가 칸 경로로 잡음)",
        lambda d: d["generators"]["A"].update(api_key="abc"),
        "generators.A.api_key",
    ),
    (
        "접속 칸 리터럴 비문자열",
        lambda d: d["generators"]["A"].update(api_key=123),
        "generators.A.api_key",
    ),
    (
        "접속 칸 기본값 문법",
        lambda d: d["generators"]["A"].update(base_url="${X:-y}"),
        "generators.A.base_url",
    ),
    ("빈 문자열", lambda d: d["generators"]["A"].update(model=""), "generators.A.model"),
]


@pytest.mark.parametrize("case", BAD_VALUES, ids=[c[0] for c in BAD_VALUES])
def test_bad_values(
    case: tuple[str, Callable[[dict[str, Any]], Any], str], tmp_path: Path, fake_registry: Registry
) -> None:
    _, fn, where = case
    assert where in errors_of(tmp_path, fake_registry, mutate(fn))


ADDRESS_TEXTS: list[tuple[str, str]] = [
    ("접속 칸에 URL", BASE_TEXT.replace("${GEN_A_URL}", "http://localhost:8000/v1")),
    ("접속 칸에 IP:포트", BASE_TEXT.replace("${LOCAL_URL}", "10.0.0.5:8000", 1)),
    ("meta에 링크", BASE_TEXT.replace("license: proprietary", "license: https://example.com/l")),
    (
        "매핑 키에 주소",
        BASE_TEXT.replace("max_tokens: 2048}", "max_tokens: 2048, gpu-01:8000: 1}", 1),
    ),
    (
        "주석에 주소",
        BASE_TEXT.replace("repeats: 3", "repeats: 3  # 예전 서버 http://10.0.0.5:8000"),
    ),
    ("주석에 호스트:포트", "# 서버 gpu-01:8000 에서 돌림\n" + BASE_TEXT),
    ("주석에 키 값", "# api_key: sk-live-abc\n" + BASE_TEXT),
    ("주석에 base_url 리터럴", BASE_TEXT.replace("repeats: 3", "repeats: 3 # base_url=gpu-box-01")),
    ("name에 주소", BASE_TEXT.replace("name: test-base", "name: see 192.168.0.10")),
    # 구현 검토 지적 1: 문장 속·조사·문장 부호가 붙은 주소
    ("조사가 붙은 호스트:포트", "# gpu-01:8000에서 돌림\n" + BASE_TEXT),
    ("마침표가 붙은 호스트:포트", "# 서버 gpu-01:8000.\n" + BASE_TEXT),
    ("세미콜론", "# localhost:8000; 예전 값\n" + BASE_TEXT),
    ("이름=값 꼴", "# --host=gpu-01:8000 로 띄움\n" + BASE_TEXT),
    ("콜론으로 붙음", "# 서버:gpu-01:8000\n" + BASE_TEXT),
    ("문장 끝 IPv4", "# 예전 서버는 10.0.0.5.\n" + BASE_TEXT),
    # 구현 검토 지적 5·15: 비밀값 꼴, 아포스트로피 뒤 주석
    ("Bearer 토큰", "# Authorization: Bearer abcdefgh12345678\n" + BASE_TEXT),
    ("sk- 키", "# sk-abcdefghijklmnop1234 예전 키\n" + BASE_TEXT),
    ("아포스트로피 뒤 주석", "description: don't  # api_key: sk-live-abc\n" + BASE_TEXT),
    # 승인 때 더한 비밀값 꼴(gitleaks에 걸리지 않게 실행할 때 이어 붙임)
    ("GitHub 토큰", "# 예전 토큰 " + "gh" + "p_" + "a" * 36 + "\n" + BASE_TEXT),
    ("GitHub 세분 토큰", "# " + "github" + "_pat_" + "a" * 30 + "\n" + BASE_TEXT),
    ("AWS 키", "# " + "AK" + "IA" + "Q" * 16 + "\n" + BASE_TEXT),
    ("Google API 키", "# " + "AI" + "za" + "a" * 35 + "\n" + BASE_TEXT),
]


@pytest.mark.parametrize("case", ADDRESS_TEXTS, ids=[c[0] for c in ADDRESS_TEXTS])
def test_addresses_rejected_anywhere(
    case: tuple[str, str], tmp_path: Path, fake_registry: Registry
) -> None:
    msg = errors_of(tmp_path, fake_registry, case[1])
    assert "행" in msg  # 원문 텍스트(주석 포함)를 줄 단위로 검사한 결과
    assert "sk-live-abc" not in msg and "gpu-01" not in msg and "192.168" not in msg


def test_comment_with_var_is_fine(tmp_path: Path, fake_registry: Registry) -> None:
    text = "# api_key: ${GEN_A_KEY} 처럼 쓴다\n" + BASE_TEXT
    load_config(write(tmp_path, text), fake_registry, env=ENV)


def test_scan_raw_text_on_text_only() -> None:
    """주석 속 주소는 파싱 전 원문 텍스트로 검사한다(파서는 주석을 버리므로)."""
    assert scan_raw_text("a: 1  # http://x.y/z\n")
    assert scan_raw_text("# token: abc12345xyz\n")
    assert not scan_raw_text("# key: 정렬 기준, token: 토큰 수\n")  # 키처럼 생기지 않은 값은 통과
    assert not scan_raw_text("model: qwen3:32b  # 모델 태그는 주소 아님\n")
    assert not scan_raw_text("a: 1  # 시각 12:30, 비율 3:2, law:000729:제27조\n")
    assert not scan_raw_text('a: "#not-a-comment base_url: x"\n')  # 따옴표 안 #은 주석 아님


@pytest.mark.parametrize(
    ("name", "allow", "expected"),
    [
        ("base_url", (), True),
        ("API-KEY", (), True),
        ("oc", (), True),
        ("judge_api_key", (), True),
        ("stop_token", (), True),
        ("stop_token", ("stop_token",), False),  # 부품이 fingerprint_allow로 연 이름
        ("license", (), False),  # meta 칸은 접속 칸 아님
        ("deploy", (), False),
        ("price_per_1m_input", (), False),
        ("max_tokens", (), False),
        ("model", (), False),
    ],
)
def test_connection_fields(name: str, allow: tuple[str, ...], expected: bool) -> None:
    assert is_connection_field(name, frozenset(allow)) is expected


def test_missing_env_names_only(tmp_path: Path, fake_registry: Registry) -> None:
    env = {k: v for k, v in ENV.items() if k not in ("GEN_A_KEY", "LOCAL_URL")}
    env["JUDGE2_URL"] = ""
    msg = errors_of(tmp_path, fake_registry, BASE_TEXT, env=env)
    assert "GEN_A_KEY" in msg and "LOCAL_URL" in msg and "JUDGE2_URL" in msg


def test_allow_missing_env(tmp_path: Path, fake_registry: Registry) -> None:
    loaded = load_config(write(tmp_path, BASE_TEXT), fake_registry, env={}, allow_missing_env=True)
    assert any("자리표시" in w for w in loaded.warnings)


# --- YAML 함정·스키마(기준 8) -----------------------------------------------------------

BAD_YAML: list[tuple[str, str, str]] = [
    ("중복 키", BASE_TEXT.replace("repeats: 3", "repeats: 3\nrepeats: 4"), "중복"),
    ("사용자 태그", BASE_TEXT.replace("name: test-base", "name: !foo bar"), "YAML 문법 오류"),
    ("binary 태그", BASE_TEXT.replace("name: test-base", "name: !!binary aGk="), "bytes"),
    ("set 태그", BASE_TEXT.replace("name: test-base", "name: !!set {a: null}"), "set"),
    (
        "문법 오류에 비밀값 줄",
        BASE_TEXT.replace("repeats: 3", 'repeats: "dummy-key'),
        "YAML 문법 오류(",
    ),
    ("최상위가 목록", "- a\n- b\n", "매핑"),
]


@pytest.mark.parametrize("case", BAD_YAML, ids=[c[0] for c in BAD_YAML])
def test_yaml_traps(case: tuple[str, str, str], tmp_path: Path, fake_registry: Registry) -> None:
    assert case[2] in errors_of(tmp_path, fake_registry, case[1])


def test_dates_stay_strings() -> None:
    assert parse_yaml("d: 2026-11-17\n") == {"d": "2026-11-17"}


def test_date_in_meta_accepted(tmp_path: Path, fake_registry: Registry) -> None:
    text = BASE_TEXT.replace(
        "license: proprietary", "license: proprietary, end_of_life: 2027-06-30"
    )
    loaded = load_config(write(tmp_path, text), fake_registry, env=ENV)
    assert loaded.arms["A"].meta is not None and loaded.arms["A"].meta.end_of_life == "2027-06-30"


SCHEMA: list[tuple[str, Callable[[dict[str, Any]], Any], str]] = [
    ("모르는 최상위 칸", lambda d: d.update(colour="x"), "colour"),
    ("빈 name", lambda d: d.update(name=""), "name"),
    ("null 값", lambda d: d.update(questions=None), "questions"),
    ("repeats 0", lambda d: d.update(repeats=0), "repeats"),
    ("arm 이름", lambda d: d["generators"].update({"A B": d["generators"]["C"]}), "generators.A B"),
    ("generators 비어 있음", lambda d: d.update(generators={}), "generators"),
    ("주 판정 0개", lambda d: d["judges"][1].update(role="secondary"), "주 판정"),
    ("주 판정 2개", lambda d: d["judges"][2].update(role="primary"), "주 판정"),
    (
        "같은 type id 없음",
        lambda d: d["judges"].append({"type": "fake_rules", "role": "rules"}),
        "fake_rules",
    ),
    ("모르는 role", lambda d: d["judges"][0].update(role="main"), "judges[0].role"),
    (
        "모르는 meta 키",
        lambda d: d["generators"]["A"]["meta"].update(color="red"),
        "generators.A.meta.color",
    ),
    (
        "meta 값 범위",
        lambda d: d["generators"]["B"]["meta"].update(vram_gb=0),
        "generators.B.meta.vram_gb",
    ),
    (
        "run 값 범위",
        lambda d: d["generators"]["A"]["run"].update(concurrency=0),
        "generators.A.run.concurrency",
    ),
    (
        "prompt_template 직접",
        lambda d: d["generators"]["A"].update(prompt_template="x"),
        "generators.A.prompt_template",
    ),
    ("arm별 prompt(R4)", lambda d: d["generators"]["A"].update(prompt="p.md"), "R4"),
    ("corpus에 run", lambda d: d["corpus"].update(run={"concurrency": 1}), "corpus.run"),
    ("절대 경로 prompt", lambda d: d.update(prompt="C:/x/prompt.md"), "상대 경로"),
    ("없는 문항 파일", lambda d: d.update(questions="nope.jsonl"), "파일이 없습니다"),
    ("generator 설정 검증(모델 없음)", lambda d: d["generators"]["A"].pop("model"), "generators.A"),
    ("judge 설정 검증(모르는 칸)", lambda d: d["judges"][1].update(temperature=0), "judges[1]"),
]


@pytest.mark.parametrize("case", SCHEMA, ids=[c[0] for c in SCHEMA])
def test_schema_errors(
    case: tuple[str, Callable[[dict[str, Any]], Any], str], tmp_path: Path, fake_registry: Registry
) -> None:
    _, fn, where = case
    data = mutate(fn)
    text = yaml.safe_dump(copy.deepcopy(data), allow_unicode=True)
    assert where in errors_of(tmp_path, fake_registry, text)


def test_errors_are_collected(tmp_path: Path, fake_registry: Registry) -> None:
    data = mutate(
        lambda d: (
            d["generators"]["A"].update(model="${M}"),
            d["generators"]["B"].update(api_key="k"),
        )
    )
    msg = errors_of(tmp_path, fake_registry, data)
    assert "generators.A.model" in msg and "generators.B.api_key" in msg


# --- 구현 검토 지적 반영 ------------------------------------------------------------


def test_unused_part_values_are_still_checked(tmp_path: Path, fake_registry: Registry) -> None:
    """지적 2: 안 쓰는 부품도 리터럴 키·주소는 검사한다(원문에 남으므로)."""
    data = mutate(lambda d: d.update(embedder={"type": "fake_embedder", "api_key": "abc"}))
    assert "embedder.api_key" in errors_of(tmp_path, fake_registry, data)


def test_nested_connection_field_rejected(tmp_path: Path, fake_registry: Registry) -> None:
    """지적 4: 접속 칸은 부품 설정 맨 위에만(중첩된 치환 값은 일반 문자열로 남음)."""
    data = mutate(
        lambda d: d["generators"]["A"]["params"].update(extra_headers={"x_api_key": "${GEN_A_KEY}"})
    )
    assert "맨 위에만" in errors_of(tmp_path, fake_registry, data)


def test_connection_field_needs_secret_config_model(
    tmp_path: Path, fake_registry: Registry
) -> None:
    """지적 4: config_model이 없는(SecretStr로 받지 않는) 부품은 접속 칸을 쓸 수 없다."""
    data = mutate(lambda d: d["generators"]["A"].update(type="fake_plain_gen"))
    assert "SecretStr" in errors_of(tmp_path, fake_registry, data)


def test_auth_prefix_is_connection(tmp_path: Path, fake_registry: Registry) -> None:
    """지적 5: authorization 같은 auth* 이름은 접속 칸이다."""
    assert is_connection_field("Authorization") and is_connection_field("auth_header")
    data = mutate(lambda d: d["generators"]["A"].update(authorization="Basic abc"))
    assert "generators.A.authorization" in errors_of(tmp_path, fake_registry, data)


@pytest.mark.parametrize(
    ("fn", "msg"),
    [
        (lambda d: d["judges"][2].pop("role"), "judges[2].role: 필수"),
        (lambda d: d["judges"][2].update(role="rules"), "모델 판정은 role: rules가 될 수 없습니다"),
        (
            lambda d: (
                d["judges"][0].update(role="primary"),
                d["judges"][1].update(role="secondary"),
            ),
            "코드 규칙 판정은 role: rules",
        ),
    ],
    ids=["role 없음", "모델 판정을 rules로", "코드 판정을 primary로"],
)
def test_judge_roles(
    fn: Callable[[dict[str, Any]], Any], msg: str, tmp_path: Path, fake_registry: Registry
) -> None:
    """지적 6: role은 필수이고, 판정 성격(judge_kind)과 맞아야 한다."""
    assert msg in errors_of(tmp_path, fake_registry, mutate(fn))


def test_path_like_model_name_is_config_error(tmp_path: Path, fake_registry: Registry) -> None:
    """지적 7: 경로형 모델 이름은 실행 전 검사에서 칸 경로와 해결 방법을 담아 알린다."""
    data = mutate(lambda d: d["generators"]["B"].update(model="/models/Qwen3.5-27B-FP8"))
    msg = errors_of(tmp_path, fake_registry, data)
    assert "generators.B" in msg and "--served-model-name" in msg


def test_constructor_error_with_config_model_hides_values(
    tmp_path: Path, fake_registry: Registry
) -> None:
    """지적 8: config_model 경로(cls(validated))의 생성자 예외에도 값이 새지 않는다."""
    data = mutate(lambda d: d["generators"]["A"].update(type="fake_chat_boom"))
    msg = errors_of(tmp_path, fake_registry, data)  # errors_of가 SECRET_VALUES 누출을 검사
    assert "generators.A" in msg and "ValueError" in msg


def test_config_model_validated_once_after_substitution(
    tmp_path: Path, fake_registry: Registry
) -> None:
    """지적 8: config_model 검증은 치환 뒤 한 번만 한다."""
    from ragexp_test_fake_components import CountingSettings  # type: ignore[import-not-found]

    CountingSettings.seen.clear()
    data = mutate(lambda d: d["generators"]["A"].update(type="fake_chat_counting"))
    load_config(write(tmp_path, data), fake_registry, env=ENV)
    assert CountingSettings.seen == [ENV["GEN_A_URL"]]


def test_fingerprint_allow_is_respected_by_loader(tmp_path: Path, fake_registry: Registry) -> None:
    """지적 8: 부품이 fingerprint_allow로 연 이름(stop_token)은 접속 칸으로 보지 않는다."""
    ok = mutate(lambda d: d["generators"]["A"].update(type="fake_chat_stop", stop_token="</s>"))
    loaded = load_config(write(tmp_path, ok), fake_registry, env=ENV)
    assert fake_registry.component_fingerprint(loaded.arms["A"].generator)["stop_token"] == "</s>"
    (tmp_path / "b").mkdir()
    bad = mutate(lambda d: d["generators"]["A"].update(stop_token="</s>"))  # fake_chat은 안 엶
    assert "generators.A.stop_token" in errors_of(tmp_path / "b", fake_registry, bad)


def test_raw_bytes_preserved_with_bom(tmp_path: Path, fake_registry: Registry) -> None:
    """지적 13: BOM·CRLF까지 바이트 그대로 보관한다(raw_bytes). raw_text는 BOM 제외."""
    text = BASE_TEXT.replace("\n", "\r\n")
    path = write(tmp_path, text)
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    loaded = load_config(path, fake_registry, env=ENV)
    assert loaded.raw_bytes == path.read_bytes()
    assert loaded.raw_text == text


def test_non_utf8_config(tmp_path: Path, fake_registry: Registry) -> None:
    path = write(tmp_path, BASE_TEXT)
    path.write_bytes(BASE_TEXT.replace("test-base", "한글").encode("cp949"))
    with pytest.raises(ConfigError, match="UTF-8"):
        load_config(path, fake_registry, env=ENV)


def test_exponent_without_dot_is_float() -> None:
    """지적 17: YAML 1.1은 1e-3을 문자열로 읽는다. 1.2처럼 float로 읽는다."""
    assert parse_yaml("t: 1e-3\nu: -2E+2\nv: 1.0e-3\n") == {"t": 0.001, "u": -200.0, "v": 0.001}


# --- U3: base_url 형식·run 주입·params 예약 키 -------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "localhost:8000/v1",
        "ftp://gpu-box/v1",
        "http://user:pw@gpu-box/v1",
        "http://gpu-box/v1?debug=1",
        "http://gpu-box/v1#frag",
        "http:///v1",
        "http://gpu-box:port/v1",
        "http://gpu box/v1",
        "http://10.0.0.256:8000/v1",
    ],
    ids=[
        "스킴 없음",
        "ftp",
        "계정",
        "쿼리",
        "조각",
        "호스트 없음",
        "포트 글자",
        "공백",
        "IPv4 범위",
    ],
)
def test_base_url_format_rejected(bad: str, tmp_path: Path, fake_registry: Registry) -> None:
    msg = errors_of(tmp_path, fake_registry, BASE_TEXT, env={**ENV, "GEN_A_URL": bad})
    assert "generators.A" in msg and "base_url_format" in msg
    assert bad not in msg and "gpu-box" not in msg


def test_base_url_trailing_slash_normalized(tmp_path: Path, fake_registry: Registry) -> None:
    env = {**ENV, "GEN_A_URL": "http://localhost:1/v1/"}
    loaded = load_config(write(tmp_path, BASE_TEXT), fake_registry, env=env)
    assert loaded.arms["A"].generator.cfg.base_url.get_secret_value() == "http://localhost:1/v1"


def test_model_settings_error_hides_input() -> None:
    """레지스트리를 거치지 않고 직접 만들어도 검증 오류 문구에 값이 없다."""
    from pydantic import ValidationError

    from ragexp.config import ModelSettings

    with pytest.raises(ValidationError) as info:
        ModelSettings.model_validate({"model": "m", "base_url": "ftp://gpu-box:8000/v1"})
    assert "base_url_format" in str(info.value) and "gpu-box" not in str(info.value)


def test_no_env_builds_components_but_client_refuses(
    tmp_path: Path, fake_registry: Registry
) -> None:
    """--no-env: 부품을 만들고 해시를 내지만, 모델 클라이언트는 자리표시 값을 거부한다."""
    from ragexp.client import ChatClient, ClientConfigError
    from ragexp.config import UNSET_PLACEHOLDER

    loaded = load_config(write(tmp_path, BASE_TEXT), fake_registry, env={}, allow_missing_env=True)
    cfg = loaded.arms["A"].generator.cfg
    assert cfg.base_url.get_secret_value() == UNSET_PLACEHOLDER
    with pytest.raises(ClientConfigError, match="base_url"):
        ChatClient(cfg)


def test_run_is_injected_into_model_settings(tmp_path: Path, fake_registry: Registry) -> None:
    data = mutate(lambda d: d["judges"][1].update(run={"timeout_s": 30, "max_retries": 0}))
    loaded = load_config(write(tmp_path, data), fake_registry, env=ENV)
    assert loaded.arms["A"].generator.cfg.run.concurrency == 8  # YAML run
    assert loaded.arms["A"].generator.cfg.run == loaded.arms["A"].run
    assert loaded.arms["B"].generator.cfg.run.timeout_s == 180  # 기본값
    judge = loaded.judges[1].component.cfg
    assert judge.run.timeout_s == 30 and judge.run.max_retries == 0


def test_run_is_not_in_fingerprint(tmp_path: Path, fake_registry: Registry) -> None:
    loaded = load_config(write(tmp_path, BASE_TEXT), fake_registry, env=ENV)
    fp = fake_registry.component_fingerprint(loaded.arms["A"].generator)
    assert "run" not in fp and "concurrency" not in str(fp)


@pytest.mark.parametrize("key", ["model", "messages", "stream", "response_format"])
def test_params_reserved_keys(key: str, tmp_path: Path, fake_registry: Registry) -> None:
    data = mutate(lambda d: d["generators"]["A"]["params"].update({key: True}))
    msg = errors_of(tmp_path, fake_registry, data)
    assert "generators.A" in msg and "params_reserved_key" in msg


def test_run_not_injected_without_run_field(tmp_path: Path, fake_registry: Registry) -> None:
    """run 칸이 없는 config_model(extra 금지)과 config_model이 없는 부품에는 넣지 않는다."""
    data = mutate(lambda d: d["judges"][2].update(type="fake_llm_norun", run={"timeout_s": 5}))
    data["judges"][2].pop("base_url")
    loaded = load_config(write(tmp_path, data), fake_registry, env=ENV)
    judge = loaded.judges[2].component
    assert judge.cfg.model_dump() == {"model": "judge-y"}  # run이 넘어오지 않았다
    assert loaded.judges[2].run.timeout_s == 5  # 실행 설정은 JudgeConfig.run으로만 간다
    assert loaded.judges[0].component.received == {}  # config_model 없는 코드 규칙 판정
