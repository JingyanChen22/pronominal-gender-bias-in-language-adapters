"""Coreference resolution: score every output file (the rules below).

One score row per output file, gender (all rows, or female / male / neutral only) and, for MLM, prediction:
- verbalized: the candidate with the higher probability (double entity only)
- non_verbalized: the model's own top token
"""
import re
import string

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .config import GENDERS, LANGUAGES, MODELS, OUTPUTS, kind

YES = {"en": "yes", "zh": "是"}
NO = {"en": "no", "zh": "否"}
COLUMNS = ["accuracy", "f1_score", "o/y_prob", "p/n_prob", "o/y_perc", "p/n_perc", "both_perc"]


def percentage(hits):
    hits = list(hits)
    return sum(hits) / len(hits) * 100


def clean(language, generations):
    """EN: lower case, no punctuation, split into words (so "no" is not found in "pronoun"). ZH: Chinese only."""
    if language == "en":
        table = str.maketrans("", "", string.punctuation)
        return [str(g).lower().translate(table).split() for g in generations]
    return ["".join(re.findall(r"[一-鿿]+", g)) for g in generations]


def score_mlm(df, language, single, yes_no):
    labels = df["label"].tolist()
    predicted = [str(p).lower() for p in df["non_verbalized"].fillna("NaN")]
    rows = {}
    if not single and not yes_no:  # occupation vs participant
        occupations, participants = df["occupation"].tolist(), df["participant"].tolist()
        verbalized = df["verbalized"].tolist()
        predicted = [occupations[i] if p == "former" else participants[i] if p == "latter" else p
                     for i, p in enumerate(predicted)]
        rows["verbalized"] = {
            "o/y_prob": df["occupation_prob"].mean(), "p/n_prob": df["participant_prob"].mean(),
            "o/y_perc": percentage(o == v for o, v in zip(occupations, verbalized)),
            "p/n_perc": percentage(p == v for p, v in zip(participants, verbalized))}
        rows["non_verbalized"] = {
            "o/y_perc": percentage(o == v for o, v in zip(occupations, predicted)),
            "p/n_perc": percentage(p == v for p, v in zip(participants, predicted))}
    if yes_no:
        verbalized = df["verbalized"].tolist()
        yes_perc = percentage(v == YES[language] for v in verbalized)
        rows["verbalized"] = {"o/y_prob": df["yes_prob"].mean(), "p/n_prob": df["no_prob"].mean(),
                              "o/y_perc": yes_perc, "p/n_perc": 100 - yes_perc}
        rows["non_verbalized"] = {"o/y_perc": percentage(p == YES[language] for p in predicted),
                                  "p/n_perc": percentage(p == NO[language] for p in predicted)}
    if single:  # single entity: only the model's own prediction, accuracy and F1
        return {"non_verbalized": {"accuracy": accuracy_score(labels, predicted) * 100,
                                   "f1_score": f1_score(labels, predicted, average="weighted") * 100}}
    rows["verbalized"].update(accuracy=accuracy_score(labels, verbalized) * 100,
                              f1_score=f1_score(labels, verbalized, average="weighted") * 100)
    rows["non_verbalized"].update(accuracy=accuracy_score(labels, predicted) * 100,
                                  f1_score=f1_score(labels, predicted, average="weighted") * 100)
    return rows


def correct_zh_yes_no(model, generations, labels, entities, occupations=None, participants=None):
    """ZH generative yes/no: an answer is also correct if it names the right person (label_entity) instead."""
    def both_people(text):
        return occupations is not None and occupations[i] in text and participants[i] in text
    correct = []
    for i, (generation, label, entity) in enumerate(zip(generations, labels, entities)):
        anti = NO["zh"] if label == YES["zh"] else YES["zh"]
        if "ChatLM" in model:
            first, rest = generation.split("，")[0], "".join(generation.split("，")[1:])
            correct.append((label in first and anti not in first) or
                           (label not in first and entity in rest and not both_people(rest)
                            and not (label in rest and anti in rest)))
        else:  # mT5: generations already cleaned
            names_entity = entity in generation and not both_people(generation) and not (
                label in generation and anti in generation)
            if generation == "":
                correct.append(False)
            elif label == NO["zh"]:
                correct.append(generation[0] == label or names_entity)
            else:
                correct.append(False if generation[0] == anti else
                               (generation[0] == label and anti not in generation[1:]) or
                               (generation[0] != label and names_entity))
    return correct


def score_generative(df, language, model, single, yes_no):
    labels = df["label"].tolist()
    generations = df["non_verbalized"].fillna("NaN").tolist()
    occupations = None if single else df["occupation"].tolist()
    participants = None if single else df["participant"].tolist()
    row = {}
    if language == "zh" and yes_no:
        if "mT5" in model:
            generations = clean(language, generations)
        correct = correct_zh_yes_no(model, generations, labels, df["label_entity"].tolist(), occupations, participants)
        if not single:
            row.update({"o/y_perc": 0.0, "p/n_perc": 0.0, "both_perc": 0.0})  # not computed for ZH yes/no
    elif yes_no:  # en
        generations = clean(language, generations)
        anti = [NO[language] if label == YES[language] else YES[language] for label in labels]
        correct = [label in g and a not in g for label, a, g in zip(labels, anti, generations)]
        if not single:
            yes, no = YES[language], NO[language]
            row.update({"o/y_perc": percentage(yes in g and no not in g for g in generations),
                        "p/n_perc": percentage(no in g and yes not in g for g in generations),
                        "both_perc": percentage(yes in g and no in g for g in generations)})
    else:  # direct, option, who, no_pronoun_who
        generations = clean(language, generations)
        if single:
            correct = [label in g for label, g in zip(labels, generations)]
        else:
            both = [o in g and p in g for o, p, g in zip(occupations, participants, generations)]
            correct = [label in g and not b for label, g, b in zip(labels, generations, both)]
            row.update({"o/y_perc": percentage(o in g and p not in g for o, p, g in zip(occupations, participants, generations)),
                        "p/n_perc": percentage(o not in g and p in g for o, p, g in zip(occupations, participants, generations)),
                        "both_perc": percentage(both)})
    row["accuracy"] = percentage(correct)
    return {"": row}


def score_file(path, language, model_type, model, entities, prompt_type):
    df = pd.read_csv(path, sep="\t")
    single, yes_no = entities == "single_entity", kind(prompt_type) == "yes_no"
    subsets = [("all", df)] + ([(g, df[df["gender"] == g].reset_index(drop=True)) for g in GENDERS]
                               if "gender" in df else [])
    rows = []
    for gender, part in subsets:
        scored = (score_mlm(part, language, single, yes_no) if model_type == "MLM"
                  else score_generative(part, language, model, single, yes_no))
        for prediction, values in scored.items():
            rows.append({"gender": gender, "prediction": prediction, **values})
    return rows


def all_scores(languages=LANGUAGES):
    """One table with the scores of every output file of the thesis models."""
    rows = []
    for (language, model_type, version), models in MODELS.items():
        if language not in languages:
            continue
        for model in models:
            folder = OUTPUTS / "coreference" / language / model_type / version / model
            for path in sorted(folder.glob("*/*/*.tsv")):
                entities, prompt_type = path.parts[-3], path.parts[-2]
                info = {"language": language, "version": version, "model_type": model_type, "model": model,
                        "entities": entities, "prompt_type": prompt_type, "template": path.stem}
                rows += [{**info, **r} for r in score_file(path, language, model_type, model, entities, prompt_type)]
    return pd.DataFrame(rows).reindex(columns=["language", "version", "model_type", "model", "entities", "prompt_type",
                                               "template", "gender", "prediction"] + COLUMNS)
