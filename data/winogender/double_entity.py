"""Sentences that name two people (an occupation and a participant), from double_entity_templates.tsv.

Every template is instantiated twice: once with its own participant ("the customer") and once with a bleached
one ("someone", or "the person"), so 120 templates give 240 sentences, or 720 with three pronouns each.
"""
import pandas as pd

from . import questions, sources
from .languages import GENDERS, OCCUPATION, PARTICIPANT, POSS, fill_pronoun, find_slot


def who_participants(language):
    """Participants for who-question prompts. English uses "the person", not "someone": the question
    ends in "The", and the model should be able to answer "person"."""
    return ("specific", "person" if language.code == "en" else "someone")


def _instantiate(language, row, participant):
    """Fill $OCCUPATION and $PARTICIPANT. `participant` is "specific", "someone" or "person".

    Returns the tokens, the participant word, and the participant part of the sentid. Bleached rows keep
    "someone" in their sentid even in the "person" variant, as in the original data.
    """
    tokens = language.tokenize(row["sentence"])
    occ_i, part_i = tokens.index(OCCUPATION), tokens.index(PARTICIPANT)
    tokens[occ_i] = row["occupation"]
    if participant == "specific":
        tokens[part_i] = row["participant"]
        return tokens, row["participant"], row["participant"]
    if language.code == "zh":
        tokens[part_i] = language.someone
        return tokens, language.someone, language.someone

    del tokens[part_i - 1]  # drop the article: "the $PARTICIPANT" -> "someone"
    part_i -= 1
    word = "someone" if participant == "someone" else "the person"
    tokens[part_i] = word.capitalize() if part_i == 0 else word
    return tokens, participant, "someone"


def _each_instance(language, participants):
    templates = sources.double_entity_templates(language.code)
    for i, row in templates.iterrows():
        for participant in participants:
            tokens, part, part_id = _instantiate(language, row, participant)
            info = {
                "sentid": f"{row['occupation']}.{part_id}.{row['answer']}",
                "label": sources.label(row, part),
                "occupation": row["occupation"],
                "participant": part,
            }
            yield i, tokens, info


def gendered(language, participants=("specific", "someone"), who=False, quote=True):
    """One row per pronoun: he/she/they (他/她/TA). `who` appends the who-question (used for the who prompts)."""
    who_questions = sources.who_questions(language.code)["double_who"] if who else None
    rows = []
    for i, tokens, info in _each_instance(language, participants):
        slot = find_slot(tokens)
        for gender in GENDERS:
            sentence = language.join(fill_pronoun(tokens, language.pronouns[gender]))
            if gender == "neutral":
                sentence = sentence.replace("they was", "they were")
            if who:
                sentence = questions.attach(language, sentence, who_questions[i], quote)
            rows.append({
                "sentid": f"{info['sentid']}.{gender}",
                "sentence": sentence,
                "label": info["label"],
                "occupation": info["occupation"],
                "participant": info["participant"],
                "gender": gender,
                "pronoun": language.pronouns[gender][slot],
            })
    return pd.DataFrame(rows)


def no_pronoun_who(language, participants, quote=True):
    """The pronoun is replaced by the entity it refers to, followed by the who-question."""
    who_questions = sources.who_questions(language.code)["double_who"]
    rows = []
    for i, tokens, info in _each_instance(language, participants):
        slot = find_slot(tokens)
        entity = info["label"]
        referent = language.possessive(entity) if slot == POSS else language.nominal(entity)
        sentence = language.join([referent if t == slot else t for t in tokens])
        rows.append({**info, "sentence": questions.attach(language, sentence, who_questions[i], quote)})
    return pd.DataFrame(rows)[["sentid", "sentence", "label", "occupation", "participant"]]
