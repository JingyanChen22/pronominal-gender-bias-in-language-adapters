"""Pronoun prediction: which occupations each model links most to she, he and they (the alternate one-person
sentences, as in the thesis), and how strongly (bias score)."""
import numpy as np
import pandas as pd

from .config import ENTITIES, MODELS, OUTPUTS, VERSIONED_MODELS, VERSIONS

PRONOUNS = ("male", "female", "neutral")  # plot and table order, as in the thesis figures
PROBS = ["female_prob", "male_prob", "neutral_prob"]


def occupations(language):
    """The 60 Winogender occupations (the first 60 entities), ZH with the English name: 技术员(technician)."""
    entities = pd.read_csv(ENTITIES, sep="\t").head(60)
    if language == "en":
        return {e: e for e in entities["entity_en"]}
    return {zh: f"{zh}({en})" for zh, en in zip(entities["entity_zh"], entities["entity_en"])}


def probabilities(language, version, model):
    """Mean probability of she/he/they per occupation over its 3 alternate sentences (nominative, possessive,
    accusative)."""
    path = (OUTPUTS / "pronoun_prediction" / language / "MLM" / version / model / "single_entity" /
            "masked_alternate.tsv")
    data = pd.read_csv(path, sep="\t")
    names = occupations(language)
    data = data[data["label"].isin(names)].assign(label=lambda d: d["label"].map(names))
    return data.groupby("label")[PROBS].mean()


def top(probs, n):
    """The n occupations with the highest probability of each pronoun: rows ordered male, female, neutral."""
    return pd.concat([probs.sort_values(f"{p}_prob", ascending=False).head(n).reset_index().assign(pronoun=p)
                      for p in PRONOUNS], ignore_index=True)


def bias(rows):
    """Per occupation: 1 - (lowest/highest + middle/highest) / 2. 0: she, he and they equally likely; 1: only one."""
    p = np.sort(rows[PROBS].to_numpy(), axis=1)
    return 1 - (p[:, 0] / p[:, 2] + p[:, 1] / p[:, 2]) / 2


def gendered_occupations(language, n, versions, models):
    """The top-n occupations per pronoun of every (model, version), with their bias."""
    tables = []
    for version in versions:
        for model in models:
            t = top(probabilities(language, version, model), n)
            tables.append(t.assign(bias=bias(t), model=model, version=version))
    return pd.concat(tables, ignore_index=True)[["model", "version", "pronoun", "label"] + PROBS + ["bias"]]


def base_models(language, n):
    return gendered_occupations(language, n, ["base"], MODELS[(language, "MLM", "base")])


def model_versions(language, n):
    return gendered_occupations(language, n, VERSIONS, VERSIONED_MODELS)


def bias_scores(top_rows):
    """Mean bias of the top occupations of each pronoun, and of all of them."""
    rows = []
    for (model, version), part in top_rows.groupby(["model", "version"], sort=False):
        row = {"model": model, "version": version}
        row.update({p: part.loc[part["pronoun"] == p, "bias"].mean() for p in PRONOUNS})
        row["all"] = part["bias"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def side_by_side(per_language):
    """{language: bias_scores} -> one row per (model, version), the columns of each language next to each other, as
    in the thesis table (left EN, right ZH). Versions in the thesis order: base, adapter, finetuned."""
    tables = []
    for language, scores in per_language.items():
        t = scores.set_index(["model", "version"])[list(PRONOUNS) + ["all"]]
        t.columns = [f"{language}_{c}" for c in t.columns]
        tables.append(t)
    table = pd.concat(tables, axis=1).reset_index()
    table["version"] = pd.Categorical(table["version"], ["base", "adapter", "finetuned"], ordered=True)
    table["model"] = pd.Categorical(table["model"], VERSIONED_MODELS, ordered=True)
    return table.sort_values(["model", "version"]).reset_index(drop=True)
