"""What the tasks read from ../data: every prompt folder, and the masked sentences.

    coreference/<entities>/<prompt type>          data/prompts/<lang>/<entities>/<prompt type>/templateN.tsv
    pronoun_prediction/single_entity/<masked>     data/sentences/<lang>/single_entity/<masked>.tsv
"""
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
TASKS = {  # task -> model types that run it
    "coreference": ("MLM", "generative"),
    "pronoun_prediction": ("MLM",),
}
PROMPT_KINDS = ("yes_no", "no_pronoun_who", "direct", "option", "who")  # first match wins


@dataclass(frozen=True)
class Input:
    task: str
    language: str
    name: str     # <entities>/<prompt type or masked file>, as in data/
    files: tuple

    @property
    def path(self):
        return f"{self.task}/{self.name}"

    @property
    def entities(self):
        return self.name.split("/")[0]

    @property
    def kind(self):
        """yes_no, no_pronoun_who, direct, option or who (prompts only)."""
        prompt_type = self.name.split("/")[1]
        return next(k for k in PROMPT_KINDS if prompt_type.startswith(k))

    def output(self, file):
        """template1.tsv for a prompt file, masked_alternate.tsv for a masked file."""
        return f"{self.name}/{file.name}" if self.task == "coreference" else f"{self.name}.tsv"


def inputs(language):
    prompts = DATA / "prompts" / language
    for folder in sorted(prompts.glob("*/*")):
        files = tuple(sorted(folder.glob("template*.tsv"), key=lambda f: int(f.stem.removeprefix("template"))))
        yield Input("coreference", language, str(folder.relative_to(prompts)), files)
    sentences = DATA / "sentences" / language
    for file in sorted(sentences.glob("single_entity/masked*.tsv")):
        yield Input("pronoun_prediction", language, str(file.relative_to(sentences).with_suffix("")), (file,))


def select(only, language, model_type):
    """The inputs of `language` that `model_type` runs, filtered by path prefixes."""
    for item in inputs(language):
        if model_type in TASKS[item.task] and (not only or any(item.path.startswith(o) for o in only)):
            yield item
