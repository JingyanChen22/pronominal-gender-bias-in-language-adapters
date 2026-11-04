"""Readers for the hand-curated files in sources/<lang>/. Everything in sentences/ and prompts/ is generated from these."""
from pathlib import Path

import pandas as pd

SOURCES_DIR = Path(__file__).resolve().parent.parent / "sources"


def _read(language, name):
    df = pd.read_csv(SOURCES_DIR / language / name, sep="\t", dtype=str, keep_default_na=False)
    if "answer" in df.columns:
        df["answer"] = df["answer"].astype(int)
    return df


def double_entity_templates(language):
    """120 Winogender templates with $OCCUPATION, $PARTICIPANT and one pronoun slot."""
    return _read(language, "double_entity_templates.tsv")


def single_entity_templates(language):
    """Hand-rewritten one-entity version of each template (`sentence`), plus a few generic `alternate_sentence`s."""
    return _read(language, "single_entity_templates.tsv")


def who_questions(language):
    """Hand-corrected follow-up questions, row-aligned with both template files."""
    return _read(language, "who_questions.tsv")


def prompt_templates(language):
    """{prompt type: [prompt, ...]}: the yes/no, direct and option templates with [TEXT], [PRON], [MENTION1/2]."""
    path = SOURCES_DIR / language / "prompt_templates.tsv"
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)  # 3 = csv.QUOTE_NONE
    return {prompt_type: group["prompt"].tolist() for prompt_type, group in df.groupby("prompt_type", sort=False)}


def check_alignment(languages):
    """The builders pair the template and question files by row number, and EN with ZH by row number
    (entity_dictionary). Raise if any file has a different number of rows or the rows disagree."""
    problems = []

    def compare(name, a, b):
        if len(a) != len(b):
            problems.append(f"{name}: {len(a)} vs {len(b)} rows")
            return
        for i in [i for i in range(len(a)) if a[i] != b[i]][:5]:
            problems.append(f"{name}, row {i + 2}: {a[i]!r} vs {b[i]!r}")  # +2: header line, 1-based

    columns = ["occupation", "participant", "answer"]
    for code in languages:
        double = double_entity_templates(code)
        single = single_entity_templates(code)
        questions = who_questions(code)
        compare(f"{code}: double_entity_templates vs single_entity_templates (occupation, participant, answer)",
                list(map(tuple, double[columns].values.tolist())), list(map(tuple, single[columns].values.tolist())))
        compare(f"{code}: double_entity_templates (occupation.answer) vs who_questions (sentid)",
                (double["occupation"] + "." + double["answer"].astype(str)).tolist(), questions["sentid"].tolist())
    if len(languages) > 1:
        answers = {code: double_entity_templates(code)["answer"].tolist() for code in languages}
        first, *rest = languages
        for code in rest:
            compare(f"{first} vs {code}: double_entity_templates (answer)", answers[first], answers[code])

    if problems:
        raise SystemExit("Source files are not aligned:\n  " + "\n  ".join(problems))


def label(row, participant=None):
    """The entity the pronoun refers to: answer 0 -> occupation, answer 1 -> participant."""
    return row["occupation"] if row["answer"] == 0 else (participant or row["participant"])
