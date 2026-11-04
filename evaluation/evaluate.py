"""Evaluate the model outputs of ../task_execution: the coreference tables and pronoun-prediction figures of the thesis.

    python evaluate.py                          # everything, both languages
    python evaluate.py --only coreference       # or: pronoun_prediction
    python evaluate.py --lang zh
    python evaluate.py --list                   # what is written, and what each file is

Coreference: every output file is scored once (the scoring rules in scoring/coreference.py) into results/coreference/
scores.tsv; every table is an average over those scores (each template file counts the same). Pronoun prediction:
the mean probabilities of she/he/they per occupation, the top occupations per pronoun, their bias scores and the
figures. Every file is overwritten, never appended to.
"""
import argparse
import time

from scoring import pronoun_prediction as pp
from scoring import tables
from scoring.config import LANGUAGES, RESULTS, TOP_BASE, TOP_BIAS, TOP_VERSIONS, VERSIONED_MODELS
from scoring.coreference import all_scores
from scoring.plots import base_model_figure, versions_figure

OUTPUTS = {  # path under results/ -> what it is
    "coreference/scores.tsv": "one row per output file × gender (all, female, male, neutral) × prediction: "
                              "the per-file scores every table averages",
    "coreference/<lang>/by_prompt_type_<MLM|generative>.tsv": "base models, double entity incl. no_pronoun_who: "
                                                              "each prompt type per model and for all models",
    "coreference/<lang>/prompt_types_<MLM|generative>.tsv": "the thesis table: per kind (Yes or No, Direct, Option, Who, "
                                                            "Who (no pronoun)) the best variant of by_prompt_type, all models",
    "coreference/<lang>/single_vs_double_<MLM|generative>.tsv": "base models, yes/no + direct + who, all templates "
                                                                "averaged: single vs double entity",
    "coreference/<lang>/versions.tsv": "mBERT and XLM-RoBERTa, base/finetuned/adapter, non_verbalized: accuracy and F1 "
                                       "per gender, average and gap",
    f"pronoun_prediction/<lang>/base_top{TOP_BASE}.tsv": f"base models: the top {TOP_BASE} occupations per pronoun, "
                                                         "with probabilities and bias",
    f"pronoun_prediction/<lang>/versions_top{TOP_VERSIONS}.tsv": f"mBERT and XLM-RoBERTa, three versions: the top "
                                                                 f"{TOP_VERSIONS} occupations per pronoun",
    "pronoun_prediction/<lang>/bias_base.tsv": f"base models: mean bias of the top {TOP_BIAS} occupations per pronoun, and over all "
                                               "of them",
    "pronoun_prediction/bias_versions.tsv": "the comparison table of the thesis: mBERT and XLM-RoBERTa × base/adapter/"
                                            "finetuned, EN columns then ZH columns",
    "pronoun_prediction/<lang>/figures/base_<model>.png": "figure of base_top: one per base model",
    "pronoun_prediction/<lang>/figures/versions_<model>.png": "figure of versions_top: mBERT and XLM-RoBERTa",
}


def write(df, path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, **kwargs)  # overwrites
    print(f"{len(df):6d} rows  {path.relative_to(RESULTS.parent)}")


def coreference(languages):
    start = time.time()
    scores = all_scores(languages)
    print(f"scored {scores.groupby(['language', 'version', 'model', 'entities', 'prompt_type', 'template']).ngroups} "
          f"output files in {time.time() - start:.0f}s")
    write(scores, RESULTS / "coreference" / "scores.tsv")
    for language in languages:
        out = RESULTS / "coreference" / language
        for model_type in ("MLM", "generative"):
            by_type = tables.by_prompt_type(scores, language, model_type)
            write(by_type, out / f"by_prompt_type_{model_type}.tsv")
            write(tables.best_prompt_type(by_type), out / f"prompt_types_{model_type}.tsv")
            write(tables.single_vs_double(scores, language, model_type), out / f"single_vs_double_{model_type}.tsv")
        write(tables.versions(scores, language), out / "versions.tsv")


def pronoun_prediction(languages):
    bias_tables = {}
    for language in languages:
        out = RESULTS / "pronoun_prediction" / language
        base = pp.base_models(language, TOP_BASE)
        versions = pp.model_versions(language, TOP_VERSIONS)
        write(base, out / f"base_top{TOP_BASE}.tsv")
        write(versions, out / f"versions_top{TOP_VERSIONS}.tsv")
        bias = pp.bias_scores(pp.base_models(language, TOP_BIAS))
        bias_versions = pp.bias_scores(pp.model_versions(language, TOP_BIAS))
        bias_tables[language] = bias_versions
        write(bias.round(2), out / "bias_base.tsv")
        (out / "figures").mkdir(parents=True, exist_ok=True)
        for _, row in bias.iterrows():
            path = out / "figures" / f"base_{row['model']}.png"
            base_model_figure(base[base.model == row["model"]], row, language, row["model"], TOP_BASE, path)
            print(f"{'':6s}       {path.relative_to(RESULTS.parent)}")
        for model in VERSIONED_MODELS:
            path = out / "figures" / f"versions_{model}.png"
            versions_figure(versions[versions.model == model], language, model, TOP_VERSIONS, path)
            print(f"{'':6s}       {path.relative_to(RESULTS.parent)}")
    if len(bias_tables) == len(LANGUAGES):
        write(pp.side_by_side(bias_tables).round(2), RESULTS / "pronoun_prediction" / "bias_versions.tsv")


def pd_concat(frames):
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["coreference", "pronoun_prediction"], help="one task (default: both)")
    parser.add_argument("--lang", nargs="+", choices=LANGUAGES, default=list(LANGUAGES))
    parser.add_argument("--list", action="store_true", help="list the files written and exit")
    args = parser.parse_args()
    if args.list:
        for path, description in OUTPUTS.items():
            print(f"results/{path}\n    {description}")
        return
    if args.only in (None, "coreference"):
        coreference(args.lang)
    if args.only in (None, "pronoun_prediction"):
        pronoun_prediction(args.lang)


if __name__ == "__main__":
    main()
