"""U2 시험용 가짜 부품. fingerprint에서 실행 설정은 빼고, 출력에 영향 주는 값만 낸다."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from ragexp.config import GeneratorSettings, ModelSettings
from ragexp.registry import register
from ragexp.types import (
    Answer,
    Chunk,
    Document,
    GenerationInput,
    Hit,
    JudgeInput,
    JudgeQuestion,
    RetrievalQuery,
    Usage,
    Verdict,
)


@register("corpus", "fake_corpus")
class FakeCorpus:
    version: ClassVar[str] = "1"

    def __init__(self, snapshot_sha256: str) -> None:
        self.snapshot_sha256 = snapshot_sha256

    def documents(self) -> Iterator[Document]:
        yield Document(doc_id="d1")

    def fingerprint(self) -> dict[str, Any]:
        return {"snapshot_sha256": self.snapshot_sha256}


@register("chunker", "fake_chunker")
class FakeChunker:
    version: ClassVar[str] = "1"

    def __init__(self, max_chars: int = 1000) -> None:
        self.max_chars = max_chars

    def chunk(self, doc: Document) -> list[Chunk]:
        return []

    def fingerprint(self) -> dict[str, Any]:
        return {"max_chars": self.max_chars}


@register("retriever", "fake_oracle")
class FakeOracle:
    version: ClassVar[str] = "1"
    needs_gold: ClassVar[bool] = True

    def __init__(self, distractors: int = 0, seed: int = 0) -> None:
        self.distractors, self.seed = distractors, seed

    def retrieve(self, query: RetrievalQuery) -> list[Hit]:
        return []

    def fingerprint(self) -> dict[str, Any]:
        return {"distractors": self.distractors, "seed": self.seed}


@register("retriever", "fake_vector")
class FakeVector(FakeOracle):
    needs_gold: ClassVar[bool] = False


@register("embedder", "fake_embedder")
class FakeEmbedder:
    version: ClassVar[str] = "1"

    def __init__(self, **config: Any) -> None:
        raise AssertionError("안 쓰는 부품은 만들지 않아야 한다")

    def embed(
        self, texts: Sequence[str], *, kind: Literal["query", "document"]
    ) -> list[list[float]]:
        return []

    def dimension(self) -> int:
        return 1

    def fingerprint(self) -> dict[str, Any]:
        return {}


@register("generator", "fake_chat")
class FakeChat:
    version: ClassVar[str] = "1"
    config_model: ClassVar[type] = GeneratorSettings

    def __init__(self, cfg: GeneratorSettings) -> None:
        self.cfg = cfg

    def generate(self, inp: GenerationInput, context: Sequence[Chunk]) -> Answer:
        return Answer(text="", finish_reason="stop", usage=Usage(latency_ms=0))

    def fingerprint(self) -> dict[str, Any]:
        return self.cfg.fingerprint_items()


@register("judge", "fake_llm")
class FakeLlmJudge:
    version: ClassVar[str] = "1"
    judge_kind: ClassVar[str] = "model"
    config_model: ClassVar[type] = ModelSettings

    def __init__(self, cfg: ModelSettings) -> None:
        self.cfg = cfg

    def judge(self, question: JudgeQuestion, response: JudgeInput) -> Verdict:
        return Verdict(judge_id="x", judge_type="fake_llm", status="ok")

    def fingerprint(self) -> dict[str, Any]:
        return self.cfg.fingerprint_items()


@register("judge", "fake_rules")
class FakeRulesJudge:
    version: ClassVar[str] = "1"
    judge_kind: ClassVar[str] = "code"

    def __init__(self, **config: Any) -> None:
        self.received = config  # run 주입 여부 시험용(아무것도 받지 않아야 함)

    def judge(self, question: JudgeQuestion, response: JudgeInput) -> Verdict:
        return Verdict(judge_id="x", judge_type="fake_rules", status="ok")

    def fingerprint(self) -> dict[str, Any]:
        return {"citation": "bracket_number"}


# --- 구현 검토 지적 반영용 가짜 부품 ---------------------------------------------------


@register("generator", "fake_chat2")
class FakeChat2(FakeChat):
    """다른 type의 생성기(한 arm만 version을 올리는 시험용)."""


@register("generator", "fake_chat_boom")
class FakeChatBoom(FakeChat):
    """생성자가 비밀값을 담은 예외를 낸다(오류 메시지 마스킹 시험용)."""

    def __init__(self, cfg: GeneratorSettings) -> None:
        raise ValueError(f"cannot reach {cfg.base_url.get_secret_value()}")


class CountingSettings(GeneratorSettings):
    """검증 횟수와, 검증 때 받은 값이 치환 뒤인지 기록한다."""

    seen: ClassVar[list[str]] = []

    @model_validator(mode="after")
    def _record(self) -> CountingSettings:
        CountingSettings.seen.append(self.base_url.get_secret_value())
        return self


@register("generator", "fake_chat_counting")
class FakeChatCounting(FakeChat):
    config_model: ClassVar[type] = CountingSettings


class StopSettings(GeneratorSettings):
    stop_token: str | None = None


@register("generator", "fake_chat_stop")
class FakeChatStop(FakeChat):
    """`stop_token`을 fingerprint_allow로 연 생성기(접속 칸이 아님)."""

    config_model: ClassVar[type] = StopSettings
    fingerprint_allow: ClassVar[frozenset[str]] = frozenset({"stop_token"})

    def fingerprint(self) -> dict[str, Any]:
        cfg = self.cfg
        assert isinstance(cfg, StopSettings)
        return {**cfg.fingerprint_items(), "stop_token": cfg.stop_token}


@register("generator", "fake_plain_gen")
class FakePlainGen:
    """config_model이 없는 생성기: 접속 칸을 일반 문자열로 받으므로 거부돼야 한다."""

    version: ClassVar[str] = "1"

    def __init__(self, **config: Any) -> None:
        self.config = config

    def generate(self, inp: GenerationInput, context: Sequence[Chunk]) -> Answer:
        return Answer(text="", finish_reason="stop", usage=Usage(latency_ms=0))

    def fingerprint(self) -> dict[str, Any]:
        return {}


class NoRunSettings(BaseModel):
    """run 칸이 없는 config_model(모르는 칸 거부): run을 주입하면 검증이 실패한다."""

    model_config = ConfigDict(extra="forbid")
    model: str


@register("judge", "fake_llm_norun")
class FakeLlmNoRun:
    version: ClassVar[str] = "1"
    judge_kind: ClassVar[str] = "model"
    config_model: ClassVar[type] = NoRunSettings

    def __init__(self, cfg: NoRunSettings) -> None:
        self.cfg = cfg

    def judge(self, question: JudgeQuestion, response: JudgeInput) -> Verdict:
        return Verdict(judge_id="x", judge_type="fake_llm_norun", status="ok")

    def fingerprint(self) -> dict[str, Any]:
        return {"model": self.cfg.model}


@register("judge", "fake_rules2")
class FakeRulesJudge2(FakeRulesJudge):
    def fingerprint(self) -> dict[str, Any]:
        return {"citation": "chunk_id"}
