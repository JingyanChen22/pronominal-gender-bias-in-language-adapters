"""Every file this stage produces, in one place. Adding a variant means adding one entry here.

Sentences: sentences/<lang>/<entities>/<pronoun form>[_person].tsv
  entities      double_entity (occupation + participant) | single_entity (one person)
  pronoun form  gendered (he/she/they filled in) | masked ($NOM_PRONOUN kept, for masked-LM pronoun prediction)
  _person       double only: "the person" instead of "someone" as the bleached participant

Prompts: prompts/<lang>/<entities>/<prompt type>[_<variant>]_<quoted|unquoted>/templateN.tsv
  prompt type   yes_no | direct | option | who | no_pronoun_who
  variant       person (en): the prompt ends in "the", so the bleached participant is "the person"
                answer (zh option): the prompt ends in 答案是
  quoted        the sentence and pronoun are in quotes; unquoted: without

Template prompts (yes_no, direct, option) come from sources/<lang>/prompt_templates.tsv, one file per template.
Who prompts have one hand-corrected question per sentence (sources/<lang>/who_questions.tsv), so they are
built by the sentence builders with who=True.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import pandas as pd

from . import double_entity, single_entity, sources
from .languages import LANGUAGES
from .prompts import prompt_rows


@dataclass(frozen=True)
class Output:
    path: str          # <kind>/<rest>; written to <kind>/<lang>/<rest>
    build: Callable    # () -> DataFrame
    description: str

    @property
    def folder(self):
        return self.path.rsplit("/", 1)[0]


# ---- sentences ---------------------------------------------------------------------------------------------

SENTENCES = {  # name -> (builder, languages, description)
    "double_entity/gendered": (
        double_entity.gendered, ("en", "zh"),
        "he/she/they, with the specific participant and with 'someone'"),
    "double_entity/gendered_person": (
        lambda L: double_entity.gendered(L, ("specific", "person")), ("en",),
        "as gendered, with 'the person' instead of 'someone'"),
    "single_entity/gendered": (
        single_entity.gendered, ("en", "zh"),
        "he/she/they in one-person sentences"),
    "single_entity/masked": (
        single_entity.masked, ("en", "zh"),
        "one-person sentences with the pronoun placeholder kept (pronoun prediction)"),
    "single_entity/masked_alternate": (
        single_entity.masked_alternate, ("en", "zh"),
        "generic one-person templates ('X said that $NOM_PRONOUN felt happy') for every entity (pronoun prediction)"),
}


@lru_cache(maxsize=None)
def sentences(code, name):
    """Built once per run and shared by the sentence file and every prompt made from it."""
    return SENTENCES[name][0](LANGUAGES[code])


def _sentence_files(language):
    for name, (_, languages, description) in SENTENCES.items():
        if language.code in languages:
            yield Output(f"sentences/{name}.tsv", lambda n=name: sentences(language.code, n), description)


def entity_dictionary(language):
    """English -> Chinese names of every occupation and participant (Chinese sentids use the Chinese names)."""
    en, zh = sources.double_entity_templates("en"), sources.double_entity_templates("zh")
    pairs = list(dict(zip(en["occupation"], zh["occupation"])).items())
    pairs += list(dict(zip(en["participant"], zh["participant"])).items())
    pairs += [("someone", "某人"), ("the person", "某人")]
    return pd.DataFrame(pairs, columns=["entity_en", "entity_zh"])


def _entity_files(language):
    if language.code == "zh":
        yield Output("entities/entity_dictionary.tsv", lambda: entity_dictionary(language),
                     "English -> Chinese entity names")


# ---- prompts -----------------------------------------------------------------------------------------------

PROMPT_TYPES = {
    "yes_no": "does the pronoun refer to [MENTION1]? yes or no",
    "direct": "the pronoun refers to ... (the model completes)",
    "option": "does the pronoun refer to [MENTION1] or [MENTION2]?",
    "who": "hand-corrected who-question, e.g. 'who could pay with cash?'",
    "no_pronoun_who": "who-question, with the pronoun replaced by its referent",
}
VARIANTS = {"person": "ends in 'the'", "answer": "ends in 答案是"}


def describe(prompt_type):
    """'option_person_unquoted' -> 'does the pronoun refer to ...? · ends in 'the' · unquoted'"""
    rest, quote = prompt_type.rsplit("_", 1)
    base, _, variant = rest.rpartition("_")
    if variant not in VARIANTS:
        base, variant = rest, None
    return " · ".join(filter(None, [PROMPT_TYPES[base], VARIANTS.get(variant), quote]))


def _template_prompts(language):
    for prompt_type, prompts in sources.prompt_templates(language.code).items():
        for entities in ("single_entity", "double_entity"):
            double = entities == "double_entity"
            if not double and prompt_type.startswith("option"):
                continue  # options need two people
            # "_person" prompts end in "the", so the bleached participant has to be "the person"
            name = "double_entity/gendered_person" if double and "_person" in prompt_type else f"{entities}/gendered"
            yes_no = prompt_type.startswith("yes_no")
            for n, prompt in enumerate(prompts, start=1):
                build = (lambda p=prompt, s=name, d=double, y=yes_no:
                         pd.DataFrame(prompt_rows(p, sentences(language.code, s), d, y, language)))
                yield Output(f"prompts/{entities}/{prompt_type}/template{n}.tsv", build, describe(prompt_type))


def _who_prompts(language):
    participants = double_entity.who_participants(language)
    for quote, suffix in ((True, "quoted"), (False, "unquoted")):
        yield Output(f"prompts/single_entity/who_{suffix}/template1.tsv",
                     lambda q=quote: single_entity.gendered(language, who=True, quote=q), describe(f"who_{suffix}"))
        yield Output(f"prompts/double_entity/who_{suffix}/template1.tsv",
                     lambda q=quote: double_entity.gendered(language, participants, who=True, quote=q),
                     describe(f"who_{suffix}"))
        yield Output(f"prompts/double_entity/no_pronoun_who_{suffix}/template1.tsv",
                     lambda q=quote: double_entity.no_pronoun_who(language, participants, quote=q),
                     describe(f"no_pronoun_who_{suffix}"))


# ---- all ---------------------------------------------------------------------------------------------------

def outputs(language):
    prompts = sorted([*_template_prompts(language), *_who_prompts(language)], key=lambda o: o.path)
    return [*_sentence_files(language), *_entity_files(language), *prompts]


def select(names=None, languages=None):
    """Yield (Output, Language), optionally filtered by path prefix: sentences, prompts/single_entity,
    prompts/double_entity/who_quoted, sentences/single_entity/masked, ..."""
    for code in languages or LANGUAGES:
        language = LANGUAGES[code]
        for output in outputs(language):
            if names and not any(_matches(output.path, n) for n in names):
                continue
            yield output, language


def _matches(path, name):
    name = name.rstrip("/")
    return path in (name, f"{name}.tsv") or path.startswith(name + "/")
