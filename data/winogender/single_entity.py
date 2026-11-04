"""Sentences that name only one person, from single_entity_templates.tsv.

With a single candidate there is nothing to resolve, so these measure which pronoun a model prefers for an
entity on its own (the "baseline" of the thesis).
"""
import pandas as pd

from . import questions, sources
from .languages import GENDERS, OCCUPATION, PARTICIPANT, fill_pronoun, find_slot


def _templates(language):
    templates = sources.single_entity_templates(language.code)
    for i, row in templates.iterrows():
        yield i, row, language.tokenize(row["sentence"])


def gendered(language, who=False, quote=True):
    """One row per pronoun: he/she/they (他/她/TA). `who` appends the who-question (used for the who prompts)."""
    who_questions = sources.who_questions(language.code)["single_who"] if who else None
    rows = []
    for i, row, tokens in _templates(language):
        slot = find_slot(tokens)
        for gender in GENDERS:
            sentence = language.join(fill_pronoun(tokens, language.pronouns[gender]))
            if gender == "neutral":
                sentence = sentence.replace("they was", "they were")
            if who:
                sentence = questions.attach(language, sentence, who_questions[i], quote)
            rows.append({
                "sentence": sentence,
                "label": sources.label(row),
                "gender": gender,
                "pronoun": language.pronouns[gender][slot],
            })
    return pd.DataFrame(rows)


def masked(language):
    """The pronoun slot is kept as a placeholder for masked-language-model prediction."""
    rows = []
    for _, row, tokens in _templates(language):
        rows.append({
            "sentence": language.join(fill_pronoun(tokens, language.masked)),
            "label": sources.label(row),
            "occupation": row["occupation"],
            "participant": row["participant"],
        })
    return pd.DataFrame(rows)


def masked_alternate(language):
    """Generic templates ("The $OCCUPATION said that $NOM_PRONOUN felt happy."), filled with every
    occupation or every participant."""
    templates = sources.single_entity_templates(language.code)
    fillers = {
        OCCUPATION: templates["occupation"].unique(),
        PARTICIPANT: templates["participant"].unique(),
    }
    rows = []
    for template in templates["alternate_sentence"]:
        for placeholder, entities in fillers.items():
            if placeholder in template:
                for entity in entities:
                    tokens = language.tokenize(template.replace(placeholder, entity))
                    rows.append({"sentence": language.join(fill_pronoun(tokens, language.masked)), "label": entity})
    return pd.DataFrame(rows)
