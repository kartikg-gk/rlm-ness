"""Task sets, in two tiers that answer different questions.

SANITY tasks are generated here and cost nothing to run. They catch gross
breakage — the loop stops working, the runtime dies, an answer never comes
back. They will not tell you whether one configuration is better than another:
they are small, the text is synthetic, and a model that finds a lexical
shortcut through them has not demonstrated anything.

BENCHMARK tasks come from published long-context sets. They are what a claim
about quality has to rest on. The archive is fetched once into evals/.cache and
read with the standard library, so the core install carries nothing extra.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import urllib.request
import zipfile
from typing import Any, Callable

from .scoring import best_f1, exact_match, numeric_match

SANITY = "sanity"
BENCHMARK = "benchmark"


@dataclasses.dataclass
class Task:
    name: str
    tier: str
    prompt: Any
    instruction: str
    score: Callable[[Any], float]
    """Returns 0.0 to 1.0. A task is 'solved' at or above its threshold."""
    threshold: float = 1.0


# --------------------------------------------------------------------------
# Sanity tier — generated, cheap, non-differentiating by construction
# --------------------------------------------------------------------------

_FILLER = [
    "The evidence assembled here is consistent across the three surveys. ",
    "Our assumption that the interval was uniform is supported by the notes. ",
    "Some uncertainty attaches to the earliest readings, but the effect is small. ",
    "The conclusion was checked against an independent series and they agree. ",
]

_HEDGE = (
    "I should set down plainly that the ordering above rests on the gauge having "
    "been read at the same hour each day, which I took on trust and have not been "
    "able to establish. If that hour drifted, the sequence would not merely weaken, "
    "it would invert. "
)

_HEDGE_QUESTION = (
    "Exactly one author says their finding would be overturned, not merely "
    "weakened, if an assumption they could not check is false. Return that "
    "document's zero-based index as an integer, and nothing else."
)


def _documents(count: int, marker: int, size: int = 1400) -> list[str]:
    documents = []
    for index in range(count):
        body = f"Document {index}.\n\n"
        turn = index
        while len(body) < size:
            body += _FILLER[turn % len(_FILLER)]
            turn += 1
        if index == marker:
            body += _HEDGE
        documents.append(body)
    return documents


def _hedge_task(count: int, marker: int) -> Task:
    return Task(
        name=f"find-the-hedge-{count}",
        tier=SANITY,
        prompt=_documents(count, marker),
        instruction=_HEDGE_QUESTION,
        score=lambda answer, expected=marker: float(exact_match(answer, expected)),
    )


def _counting_task(repeats: int = 300) -> Task:
    text = ("strawberry " * repeats).strip()
    return Task(
        name=f"count-letters-{repeats}",
        tier=SANITY,
        prompt=text,
        instruction="How many times does the letter r appear in PROMPT? Return an integer.",
        score=lambda answer, expected=text.count("r"): float(
            exact_match(answer, expected)
        ),
    )


SANITY_TASKS = [_hedge_task(8, 5), _hedge_task(16, 11), _counting_task()]


# --------------------------------------------------------------------------
# Synthetic long-context tier — generated, but not searchable
# --------------------------------------------------------------------------

_SPEAKERS = [
    "Ada", "Boris", "Chen", "Dara", "Emil", "Farah",
    "Goro", "Hilde", "Ivo", "Juno", "Kemi", "Lars",
]

_MOVES = [
    "{who} moved {n} units of grain to the {place} store.",
    "{who} took {n} units of grain out of the {place} store.",
    "{who} counted the {place} store and wrote down {n}.",
    "{who} said nothing about the {place} store this week.",
]

_PLACES = ["north", "south", "river", "hill"]


def _ledger(weeks: int, seed: int = 7) -> tuple[str, int]:
    """A running total no substring can reveal.

    Every line is quantitative and every line matters, so the answer is a
    property of the whole document rather than of one passage in it. That is
    the shape a retrieval question never has: there is no distinctive token to
    search for, because the number being asked about is never written down.
    """
    state = seed
    held = 0
    lines = []
    for week in range(weeks):
        for slot in range(4):
            state = (state * 1103515245 + 12345) % 2147483648
            who = _SPEAKERS[state % len(_SPEAKERS)]
            place = _PLACES[(state // 7) % len(_PLACES)]
            count = (state // 13) % 40 + 1
            template = _MOVES[(state // 11) % len(_MOVES)]
            if template is _MOVES[0]:
                held += count
            elif template is _MOVES[1]:
                held -= count
            lines.append(
                f"Week {week}, entry {slot}: "
                + template.format(who=who, n=count, place=place)
            )
    return "\n".join(lines), held


def _ledger_task(weeks: int) -> Task:
    text, held = _ledger(weeks)
    return Task(
        name=f"ledger-{weeks}w",
        tier=SANITY,
        prompt=text,
        instruction=(
            "PROMPT is a grain ledger. Some entries move grain into a store, "
            "some take it out, and some only report or say nothing. Starting "
            "from zero, work out the net amount held once every entry has been "
            "applied in order. Return the number and nothing else."
        ),
        score=lambda answer, expected=held: float(numeric_match(answer, expected)),
    )


# Long enough that the whole document cannot be read in one window, and
# arithmetic rather than lookup, so a sub-agent given a slice can do real work
# that the parent then adds up. Kept in the sanity tier deliberately: the text
# is generated, so a result here is evidence the mechanism runs, never evidence
# about quality.
LONG_SYNTHETIC_TASKS = [_ledger_task(40), _ledger_task(160)]


# --------------------------------------------------------------------------
# Benchmark tier — published data, fetched once, standard library only
# --------------------------------------------------------------------------

LONGBENCH_URL = "https://huggingface.co/datasets/zai-org/LongBench/resolve/main/data.zip"
CACHE = pathlib.Path(__file__).resolve().parent / ".cache"


def _benchmark_archive() -> pathlib.Path:
    """Fetch the archive once, then read from it.

    It is one 109 MB zip holding every split as JSONL, so a dataset library
    would be a large dependency for a download and a line-per-record read.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / "longbench.zip"
    if not archive.exists():
        print(f"fetching LongBench once into {archive} ...")
        request = urllib.request.Request(
            LONGBENCH_URL, headers={"User-Agent": "rlm-ness-evals"}
        )
        with urllib.request.urlopen(request) as response:
            partial = archive.with_suffix(".part")
            partial.write_bytes(response.read())
            partial.replace(archive)
    return archive


# Splits whose question lives in the split itself rather than in a per-row
# field. Both need every paragraph judged, and both have a checkable answer —
# the combination a retrieval split cannot offer, because a question that names
# its target is always answered faster by searching for the name.
_SPLIT_INSTRUCTIONS = {
    "passage_count": (
        "PROMPT holds paragraphs from Wikipedia, some of them duplicates of each "
        "other in wording that may differ. Work out how many distinct paragraphs "
        "remain once duplicates are treated as one. Return only the number."
    ),
    "passage_retrieval_en": (
        "PROMPT holds numbered paragraphs from Wikipedia. The abstract below "
        "summarises exactly one of them. Work out which. Answer in the form "
        "'Paragraph 12' and nothing else.\n\nAbstract:\n{input}"
    ),
}


def _paragraph_number(value: Any) -> int | None:
    found = re.search(r"\d+", str(value))
    return int(found.group()) if found else None


def _score_for(config: str, answers: list):
    if config == "passage_count":
        return lambda answer: float(numeric_match(answer, answers[0])), 1.0
    if config == "passage_retrieval_en":
        expected = _paragraph_number(answers[0])
        return (
            lambda answer: float(_paragraph_number(answer) == expected),
            1.0,
        )
    # Free-form answers with many valid wordings: token overlap, partial credit.
    return lambda answer: best_f1(answer, answers), 0.5


def load_longbench(config: str = "hotpotqa", limit: int = 3) -> list[Task]:
    archive = _benchmark_archive()
    wanted = f"data/{config}.jsonl"
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if wanted not in names:
            splits = sorted(
                name.split("/")[-1].removesuffix(".jsonl")
                for name in names
                if name.endswith(".jsonl")
            )
            raise SystemExit(f"no split {config!r}. available: {', '.join(splits)}")
        rows = []
        with bundle.open(wanted) as handle:
            for line in handle:
                if len(rows) >= limit:
                    break
                rows.append(json.loads(line))

    tasks = []
    for index, row in enumerate(rows):
        answers = list(row.get("answers") or [])
        template = _SPLIT_INSTRUCTIONS.get(config)
        instruction = (
            template.format(input=row.get("input", "")) if template else row["input"]
        )
        score, threshold = _score_for(config, answers)
        tasks.append(
            Task(
                name=f"longbench-{config}-{index}",
                tier=BENCHMARK,
                prompt=row["context"],
                instruction=instruction,
                score=score,
                threshold=threshold,
            )
        )
    return tasks


# --------------------------------------------------------------------------
# Generation tier — the answer cannot be computed from PROMPT, only produced
# --------------------------------------------------------------------------

def _counts_letter(answer: Any, letter: str, wanted: int) -> float:
    """Score a mapping of names to how often a letter appears in each.

    Self-verifying: every entry carries both the name and the claim about it,
    so the claim can be checked against the name without any answer key. What
    is scored is the fraction of entries that are right, with no credit for
    returning fewer than were asked for.
    """
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except ValueError:
            found = re.findall(r"['\"]([A-Za-z ]+)['\"]\s*:\s*(\d+)", str(answer))
            answer = {name: int(count) for name, count in found}
    if not isinstance(answer, dict) or not answer:
        return 0.0
    right = sum(
        1
        for name, count in answer.items()
        if str(name).lower().count(letter) == _as_int(count)
    )
    # Short of the number asked for is short, however accurate the part given.
    return right / max(wanted, len(answer))


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _generation_task(kind: str, wanted: int, letter: str = "r") -> Task:
    """Ask for something PROMPT does not contain.

    Every other task here can be answered by reading the data, and reading is
    what code does best — so a model solves them alone and delegation never
    pays. Nothing in the namespace can invent fifty fruit names. The only way
    to get them is to ask a model, which makes this the one shape where
    handing work out is the shortest path rather than a detour.
    """
    return Task(
        name=f"generate-{kind}-{wanted}",
        tier=SANITY,
        prompt=(
            f"There is no data here. The work is to produce {wanted} {kind}, "
            f"then report how many times the letter '{letter}' appears in each."
        ),
        instruction=(
            f"Produce {wanted} distinct {kind}. You cannot compute them from "
            f"PROMPT and there is nothing to read: ask sub-agents for them, "
            f"splitting the work across several so it is not one long list. "
            f"Then, in code, count how many times the letter '{letter}' "
            f"appears in each name. Check you have {wanted} before answering. "
            f"FINAL a dictionary mapping each name to its count."
        ),
        score=lambda answer, w=wanted, l=letter: _counts_letter(answer, l, w),
        threshold=0.9,
    )


GENERATION_TASKS = [
    _generation_task("fruit names", 50),
    _generation_task("animal names", 30),
]


def resolve(name: str, limit: int) -> list[Task]:
    if name == SANITY:
        return SANITY_TASKS
    if name == "generate":
        return GENERATION_TASKS[:limit] if limit else GENERATION_TASKS
    if name == "ledger":
        return LONG_SYNTHETIC_TASKS[:limit] if limit else LONG_SYNTHETIC_TASKS
    if name.startswith("longbench"):
        _, _, config = name.partition(":")
        return load_longbench(config or "hotpotqa", limit)
    raise SystemExit(f"unknown task set: {name}")
