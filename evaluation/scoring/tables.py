"""The coreference tables of the thesis, each an average over the score rows (one row per output file).

Every average is over template files, so every template counts the same.
"""
import numpy as np
import pandas as pd

from .config import GENDERS, MODELS, PROMPT_TYPES, SHARED_WITH_SINGLE, VERSIONED_MODELS, VERSIONS, kind

MLM_COLUMNS = ["accuracy", "f1_score", "o/y_prob", "p/n_prob", "o/y_perc", "p/n_perc"]
GENERATIVE_COLUMNS = ["accuracy", "o/y_perc", "p/n_perc", "both_perc"]
ALL_MODELS = "all models"


def columns(model_type, entities="double_entity"):
    if entities == "single_entity":
        return ["accuracy", "f1_score"] if model_type == "MLM" else ["accuracy"]
    return MLM_COLUMNS if model_type == "MLM" else GENERATIVE_COLUMNS


def ordered(df, by):
    """Rows in table order: models as in config, prompt types as in config, verbalized before non_verbalized."""
    keys = {"prompt_type": PROMPT_TYPES, "prediction": ["verbalized", "non_verbalized", ""],
            "entities": ["single_entity", "double_entity"], "version": list(VERSIONS)}
    df = df.copy()
    for column in by:
        if column in keys:
            df[column] = pd.Categorical(df[column], [k for k in keys[column] if k in set(df[column])], ordered=True)
    return df.sort_values(by).reset_index(drop=True)


def average(scores, by, cols):
    """Mean of cols per group. A column that does not apply (e.g. the probabilities of non_verbalized) stays empty."""
    return scores.groupby(by, observed=True, sort=False)[cols].mean().reset_index()


def by_prompt_type(scores, language, model_type):
    """Base models, double entity (with no_pronoun_who): every prompt type, per model and for all models together."""
    s = scores[(scores.language == language) & (scores.version == "base") & (scores.model_type == model_type)
               & (scores.entities == "double_entity") & (scores.gender == "all")]
    cols = columns(model_type)
    # no_pronoun_who: quoted and unquoted together, one row (as in the thesis table); every other type per quote
    s = s.assign(prompt_type=s["prompt_type"].replace({"no_pronoun_who_quoted": "no_pronoun_who",
                                                       "no_pronoun_who_unquoted": "no_pronoun_who"}))
    per_model = average(s, ["prompt_type", "model", "prediction"], cols)
    together = average(s, ["prompt_type", "prediction"], cols).assign(model=ALL_MODELS)
    table = pd.concat([per_model, together])
    table["model"] = pd.Categorical(table["model"], MODELS[(language, model_type, "base")] + [ALL_MODELS], ordered=True)
    return ordered(table, ["prompt_type", "model", "prediction"])[["prompt_type", "model", "prediction"] + cols]


KIND_NAME = {"yes_no": "Yes or No", "direct": "Direct", "option": "Option", "who": "Who",
             "no_pronoun_who": "Who (no pronoun)"}


def best_prompt_type(by_type):
    """The thesis table: for each kind of prompt (Yes or No, Direct, Option, Who, Who (no pronoun)) the variant with
    the highest all-models accuracy (e.g. yes_no_unquoted over yes_no_quoted), with that variant's other scores.
    MLM: chosen by the non_verbalized accuracy; the verbalized row is the same variant's. `by_type` is the
    by_prompt_type table."""
    s = by_type[by_type.model == ALL_MODELS].copy()
    s["prompt_type"] = s["prompt_type"].astype(str)
    s["kind"] = s["prompt_type"].map(kind)
    s["prediction"] = s["prediction"].astype(str)
    judge = s[s.prediction.isin(["non_verbalized", ""])]  # generative: no prediction column
    chosen = judge.loc[judge.groupby("kind", sort=False)["accuracy"].idxmax(), "prompt_type"]
    best = s[s.prompt_type.isin(chosen)]
    best = best.assign(best_of=best["prompt_type"], prompt_type=best["kind"].map(KIND_NAME))
    best["kind"] = pd.Categorical(best["kind"], list(KIND_NAME), ordered=True)
    best = ordered(best, ["kind", "prediction"])
    return best[["prompt_type", "best_of", "prediction"] + [c for c in by_type.columns if c not in
                                                           ("prompt_type", "model", "prediction")]]


def single_vs_double(scores, language, model_type):
    """Base models, all templates averaged per model: single entity (yes/no, direct, who) and double entity restricted
    to the same prompt types."""
    s = scores[(scores.language == language) & (scores.version == "base") & (scores.model_type == model_type)
               & (scores.gender == "all") & scores.prompt_type.isin(SHARED_WITH_SINGLE)]
    table = average(s, ["model", "entities", "prediction"], columns(model_type))
    table["model"] = pd.Categorical(table["model"], MODELS[(language, model_type, "base")], ordered=True)
    return ordered(table, ["model", "entities", "prediction"])


def gap(values):
    """How far the three genders are from their mean, on average (0: the same for she, he and they)."""
    values = np.asarray(values)
    return np.abs(values - values.mean()).mean()


def versions(scores, language):
    """mBERT and XLM-RoBERTa, base vs finetuned vs adapter, single and double entity, non_verbalized predictions,
    yes/no, direct and who: per gender, their average and the gap between genders."""
    s = scores[(scores.language == language) & (scores.model_type == "MLM") & scores.model.isin(VERSIONED_MODELS)
               & (scores.prediction == "non_verbalized") & scores.gender.isin(GENDERS)
               & scores.prompt_type.isin(SHARED_WITH_SINGLE)]
    per_gender = s.groupby(["entities", "model", "version", "gender"])[["accuracy", "f1_score"]].mean()
    rows = []
    for (entities, model, version), part in per_gender.groupby(level=[0, 1, 2]):
        part = part.droplevel([0, 1, 2]).reindex(list(GENDERS))
        row = {"entities": entities, "model": model, "version": version}
        for score in ("accuracy", "f1_score"):
            row.update({f"{score}_{g}": part.loc[g, score] for g in GENDERS})
            row[f"{score}_avg"], row[f"{score}_gap"] = part[score].mean(), gap(part[score])
        rows.append(row)
    table = pd.DataFrame(rows)
    table["model"] = pd.Categorical(table["model"], VERSIONED_MODELS, ordered=True)
    return ordered(table, ["entities", "model", "version"])
