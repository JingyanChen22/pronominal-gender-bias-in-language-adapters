"""Run the models on the data in ../data: coreference resolution and pronoun prediction.

    python run_tasks.py --list                                     # inputs and models
    python run_tasks.py --model BERT --lang en                     # every input BERT runs, base version
    python run_tasks.py --model T5 --lang en --only coreference/double_entity/yes_no
    python run_tasks.py --model mBERT --lang zh --only pronoun_prediction/single_entity/masked_alternate

  finetuned/adapter, same language: the model trained on the language of the data (the main results)
    python run_tasks.py --model mBERT --lang en --version finetuned   # models/finetuned/en_mbert on EN data -> mBERT
    python run_tasks.py --model mBERT --lang zh --version adapter     # models/adapter/zh_mbert on ZH data -> mBERT

  finetuned/adapter, cross-lingual: a model trained on the other language (--lang: data, --trained_on: model)
    python run_tasks.py --model mBERT --lang en --version adapter --trained_on zh  # zh adapter on EN data -> mBERT_zh

  another trained model
    python run_tasks.py --model mBERT --lang en --version finetuned --model_path <model folder> --name <output name>

Outputs: outputs/<task>/<lang>/<model type>/<version>/<model>/<input>, e.g.
    outputs/coreference/en/MLM/base/BERT/double_entity/yes_no_quoted/template1.tsv
    outputs/pronoun_prediction/zh/MLM/finetuned/mBERT/single_entity/masked_alternate.tsv
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from tasks import coreference, pronoun_prediction
from tasks.inputs import TASKS, inputs, select
from tasks.models import MODELS, VERSIONS, load, output_name, trained_path

HERE = Path(__file__).resolve().parent
LANGUAGES = ("en", "zh")
ROW = {"pronoun_prediction": lambda model: pronoun_prediction.row,
       "coreference": lambda model: coreference.ROW[model.type]}


def show():
    """One line per input (prompt folder or masked file), then one line per model."""
    rows = defaultdict(lambda: defaultdict(int))
    for language in LANGUAGES:
        for item in inputs(language):
            rows[item.path][language] = len(item.files)
    task_seen = None
    for path in sorted(rows, key=lambda p: (list(TASKS).index(p.split("/")[0]), p)):
        task, name = path.split("/", 1)
        if task != task_seen:
            print(f"{'' if task_seen is None else chr(10)}{task}/   run by: {', '.join(TASKS[task])}")
            task_seen = task
        n = max(rows[path].values())
        print(f"  {name:38s} {'+'.join(rows[path]):6s} {n} file{'s' * (n > 1)}")
    print("\nmodels/")
    for model in MODELS.values():
        print(f"  {model.name:12s} {model.type:11s} {'+'.join(model.languages):6s} {'/'.join(model.versions):24s} "
              f"{model.hub}")


def run(model, language, version, args):
    if language not in model.languages:
        print(f"skip: {model.name} is not run on {language}")
        return
    selected = list(select(args.only, language, model.type))
    if not selected:
        print(f"skip: nothing in {args.only} is run by {model.type} models on {language}; see --list")
        return
    trained_on = args.trained_on or language
    path = args.model_path or (trained_path(model, version, trained_on) if version != "base" else model.hub)
    name = output_name(model, language, version, trained_on, args.name)
    out = HERE / "outputs" / "{task}" / language / model.type / version / name
    print(f"== {model.name} ({version}: {path}) on {language} -> {out.relative_to(HERE)}")

    lm, tokenizer = load(model, version, path, args.adapter_name, args.device)
    row = {task: make(model) for task, make in ROW.items()}
    for item in selected:
        for file in item.files:
            start = time.time()
            data = pd.read_csv(file, sep="\t", header=0)
            values, columns = [], None
            for _, r in data.iterrows():
                v, columns = row[item.task](lm, tokenizer, model, language, item, r, args.device)
                values.append(v)
            target = Path(str(out).format(task=item.task)) / item.output(file)
            target.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(values, columns=columns).to_csv(target, sep="\t", index=False)  # overwrites
            print(f"{len(values):5d} rows {time.time() - start:6.0f}s  {target.relative_to(HERE)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list the inputs and models and exit")
    parser.add_argument("--model", nargs="+", choices=MODELS, help="models to run")
    parser.add_argument("--lang", nargs="+", choices=LANGUAGES, default=list(LANGUAGES),
                        help="language of the data (default: both)")
    parser.add_argument("--only", nargs="+", metavar="PATH", help="input path prefixes, as --list shows them")
    parser.add_argument("--version", choices=VERSIONS, default="base")
    parser.add_argument("--trained_on", choices=LANGUAGES, help="finetuned/adapter: language the model was trained "
                                                                 "on (default: --lang; the other one: cross-lingual)")
    parser.add_argument("--model_path", help="finetuned/adapter: model folder (default: ../model_training/models/"
                                             "<finetuned|adapter>/<trained_on>_<mbert|xlmr>)")
    parser.add_argument("--name", help="output folder of the model (default: the model, e.g. mBERT or mBERT_zh)")
    parser.add_argument("--adapter_name", default="mlm")
    parser.add_argument("--device", default="cpu", help="cpu (as in the thesis runs), cuda or mps")
    args = parser.parse_args()

    if args.list:
        show()
        return
    if not args.model:
        parser.error("give --model (see --list)")
    for name in args.model:
        model = MODELS[name]
        if args.version not in model.versions:
            parser.error(f"{name} has no {args.version} version (only {', '.join(model.versions)})")
        for language in args.lang:
            run(model, language, args.version, args)


if __name__ == "__main__":
    main()
