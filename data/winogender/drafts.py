"""Automatic first drafts of files that were then corrected by hand into sources/.

Re-running these never touches sources/: they only write to drafts/, so a new template can be drafted
here, corrected, and then copied into the matching source file.
"""
import pandas as pd

from . import questions, sources
from .languages import OCCUPATION, PARTICIPANT


def single_entity_templates(language):
    """Fill in the entity the pronoun refers to and keep the other as a placeholder. Rewriting the result
    into a one-person sentence by hand gave the `sentence` column of single_entity_templates.tsv."""
    templates = sources.double_entity_templates(language.code)
    rows = []
    for _, row in templates.iterrows():
        tokens = language.tokenize(row["sentence"])
        if row["answer"] == 0:
            tokens[tokens.index(OCCUPATION)] = row["occupation"]
        else:
            tokens[tokens.index(PARTICIPANT)] = row["participant"]
        if language.code == "zh":
            sentence = "".join(f" {t} " if t.startswith("$") else t for t in tokens)
        else:
            sentence = " ".join(tokens)
        rows.append({**row[["occupation", "participant", "answer"]], "draft_sentence": sentence})
    return pd.DataFrame(rows)


def double_who_questions(language):
    """Draft of the `double_who` column of who_questions.tsv (kept there as `double_who_auto`)."""
    templates = sources.double_entity_templates(language.code)
    rows = []
    for _, row in templates.iterrows():
        tokens = language.tokenize(row["sentence"])
        tokens[tokens.index(OCCUPATION)] = row["occupation"]
        tokens[tokens.index(PARTICIPANT)] = row["participant"]
        slot, question = questions.draft(language, tokens)
        rows.append({
            "pronoun_slot": slot,
            "sentid": f"{row['occupation']}.{row['answer']}",
            "double_sentence": " ".join(tokens),
            "double_who_auto": question,
        })
    return pd.DataFrame(rows)


def single_who_questions(language):
    """Draft of the `single_who` column of who_questions.tsv."""
    templates = sources.single_entity_templates(language.code)
    rows = []
    for _, row in templates.iterrows():
        tokens = language.tokenize(row["sentence"])
        slot, question = questions.draft(language, tokens)
        rows.append({
            "pronoun_slot": slot,
            "sentid": f"{sources.label(row)}.{row['answer']}",
            "single_sentence": " ".join(tokens),
            "single_who_auto": question,
        })
    return pd.DataFrame(rows)


DRAFTS = {
    "single_entity_templates": single_entity_templates,
    "double_who_questions": double_who_questions,
    "single_who_questions": single_who_questions,
}
