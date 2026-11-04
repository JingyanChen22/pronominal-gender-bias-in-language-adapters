"""What is evaluated: paths, the models of the thesis results, and the prompt types."""
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUTPUTS = HERE.parent / "task_execution" / "outputs"   # the model outputs (stage 3)
ENTITIES = HERE.parent / "data" / "entities" / "zh" / "entity_dictionary.tsv"
RESULTS = HERE / "results"

LANGUAGES = ("en", "zh")
VERSIONS = ("base", "finetuned", "adapter")
GENDERS = ("female", "male", "neutral")

# The models of the thesis results. Other model folders in the outputs (the cross-lingual mBERT_zh, XLM-RoBERTa_en,
# ...) are not evaluated.
MODELS = {
    ("en", "MLM", "base"): ["BERT", "mBERT", "RoBERTa", "XLM-RoBERTa"],
    ("zh", "MLM", "base"): ["BERT_zh", "mBERT", "RoBERTa_zh", "XLM-RoBERTa"],
    ("en", "generative", "base"): ["T5", "mT5", "FLAN"],
    ("zh", "generative", "base"): ["mT5", "ChatLM"],
}
for language in LANGUAGES:
    for version in ("finetuned", "adapter"):
        MODELS[(language, "MLM", version)] = ["mBERT", "XLM-RoBERTa"]  # trained on the language of the data
VERSIONED_MODELS = ["mBERT", "XLM-RoBERTa"]  # the models compared across base, finetuned and adapter

PROMPT_TYPES = [  # table order; data/prompts/<lang>/<entities>/ has the ones of each language
    "yes_no_quoted", "yes_no_unquoted",
    "direct_quoted", "direct_unquoted", "direct_person_quoted", "direct_person_unquoted",
    "option_quoted", "option_unquoted", "option_person_quoted", "option_person_unquoted",
    "option_answer_quoted", "option_answer_unquoted",
    "who_quoted", "who_unquoted",
    "no_pronoun_who_quoted", "no_pronoun_who_unquoted", "no_pronoun_who",  # last: both quotes, in by_prompt_type
]
# the prompt types that exist for one person too: double-entity results restricted to these compare with single-entity
SHARED_WITH_SINGLE = [p for p in PROMPT_TYPES if p.startswith(("yes_no", "direct", "who"))]

TOP_BASE = 5        # pronoun prediction, base-model figures: occupations per pronoun
TOP_VERSIONS = 3    # pronoun prediction, figures of base vs finetuned vs adapter
TOP_BIAS = 5        # every bias score, every model and version: the top 5 occupations per pronoun


def kind(prompt_type):
    """yes_no, no_pronoun_who, direct, option or who"""
    return next(k for k in ("yes_no", "no_pronoun_who", "direct", "option", "who") if prompt_type.startswith(k))
