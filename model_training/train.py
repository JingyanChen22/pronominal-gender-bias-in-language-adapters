"""Train the finetuned and adapter versions of mBERT and XLM-RoBERTa on Wikipedia (masked language modelling).

    python train.py --list                                          # the 8 trainings and their parameters
    python train.py --method finetuned --model mBERT --lang en       # one training
    python train.py --method adapter --model XLM-RoBERTa --lang zh --dry_run   # only print the command
    python train.py --method finetuned adapter --model mBERT XLM-RoBERTa --lang en zh   # all 8, one after another
    python train.py ... --gpu 1                                      # use GPU 1 (default: every visible GPU)
    python train.py ... --no_wandb                                   # do not log to Weights & Biases

Each training runs the Hugging Face example script run_mlm.py (unchanged) with the parameters below, the ones
the thesis models were trained with (python train.py --list). Models are written to models/<method>/<lang>_<mbert|xlmr>/,
where task_execution/ finds them.
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANGUAGES = ("en", "zh")
METHODS = ("finetuned", "adapter")
MODELS = {  # name in task_execution -> (Hugging Face model, short name of the trained model)
    "mBERT": ("bert-base-multilingual-uncased", "mbert"),
    "XLM-RoBERTa": ("xlm-roberta-base", "xlmr"),
}

# run_mlm.py arguments shared by all 8 trainings, in the order of the thesis's run_mlm.sh
COMMON = [
    "--overwrite_output_dir",
    "--do_train",
    "--do_eval",
    "--validation_split_percentage", "10",
    "--per_device_train_batch_size", "8",
    "--per_device_eval_batch_size", "8",
    "--max_seq_length", "512",
    "--evaluation_strategy", "steps",
    "--eval_steps", "10000",
    "--save_steps", "150000",
    "--max_steps", "150000",
    "--num_train_epochs", "5",  # ignored by the Trainer: max_steps wins
    "--save_total_limit", "2",
    "--load_best_model_at_end",
]
ADAPTER = ["--train_adapter", "--adapter_config", "pfeiffer+inv"]  # the adapter is saved as <output>/mlm/


def run_name(method, model, language):
    """As in the thesis runs (W&B names only): finetuned_en_mbert, finetune_en_xlmr, adapter_en_mbert."""
    short = MODELS[model][1]
    if method == "adapter":
        return f"adapter_{language}_{short}"
    return f"{'finetuned' if short == 'mbert' else 'finetune'}_{language}_{short}"


def output_dir(method, model, language):
    return HERE / "models" / method / f"{language}_{MODELS[model][1]}"


def command(method, model, language, wandb=True):
    hub = MODELS[model][0]
    args = [sys.executable, str(HERE / "run_mlm.py"),
            "--model_name_or_path", hub,
            "--train_file", str(HERE / "corpus" / language / f"wiki_{language}.txt"),
            "--output_dir", str(output_dir(method, model, language)),
            *COMMON,
            "--run_name", run_name(method, model, language)]
    if method == "adapter":
        args += ADAPTER
    if not wandb:
        args += ["--report_to", "none"]
    return args


def show():
    print("trainings (models/<method>/<lang>_<short>):")
    for method in METHODS:
        for model, (hub, short) in MODELS.items():
            for language in LANGUAGES:
                out = output_dir(method, model, language)
                done = "trained" if (out / "all_results.json").exists() else "-"
                print(f"  {method:10s} {model:12s} {language}  {hub:31s} corpus/{language}/wiki_{language}.txt  "
                      f"-> {out.relative_to(HERE)}  [{done}]")
    print("\nparameters (all trainings):")
    for i in range(0, len(COMMON)):
        if COMMON[i].startswith("--"):
            value = COMMON[i + 1] if i + 1 < len(COMMON) and not COMMON[i + 1].startswith("--") else ""
            print(f"  {COMMON[i]} {value}")
    print(f"adapter only:\n  {' '.join(ADAPTER)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list the trainings and parameters and exit")
    parser.add_argument("--method", nargs="+", choices=METHODS)
    parser.add_argument("--model", nargs="+", choices=MODELS)
    parser.add_argument("--lang", nargs="+", choices=LANGUAGES, help="language of the Wikipedia corpus")
    parser.add_argument("--dry_run", action="store_true", help="print the commands, do not train")
    parser.add_argument("--gpu", help="GPU(s) to use, e.g. 0 or 0,1 (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--no_wandb", action="store_true", help="do not log to Weights & Biases (the thesis did)")
    args = parser.parse_args()

    if args.list:
        show()
        return
    if not (args.method and args.model and args.lang):
        parser.error("give --method, --model and --lang (see --list)")

    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    for name in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR"):
        env.setdefault(name, str(HERE / "wandb"))

    for method in args.method:
        for model in args.model:
            for language in args.lang:
                cmd = command(method, model, language, wandb=not args.no_wandb)
                print(f"== {method} {model} on {language} Wikipedia\n{shlex.join(cmd)}\n", flush=True)
                if args.dry_run:
                    continue
                (HERE / "wandb").mkdir(exist_ok=True)
                subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
