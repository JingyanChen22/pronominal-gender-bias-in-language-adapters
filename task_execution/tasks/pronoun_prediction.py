"""Pronoun prediction (MLM only): the pronoun slot of a one-person sentence is masked, and the probabilities of
the female, male and neutral pronoun at the mask are compared."""
import re

import torch.nn.functional as F

from .coreference import mask_scores

SLOT = re.compile(r"\$[A-Z_]+")
PRONOUNS = {  # slot -> female, male, neutral
    "en": {"$NOM_PRONOUN": ["she", "he", "they"], "$POSS_PRONOUN": ["her", "his", "their"],
           "$ACC_PRONOUN": ["her", "him", "them"]},
    "zh": {slot: ["她", "他", "TA"] for slot in ("$NOM_PRONOUN", "$POSS_PRONOUN", "$ACC_PRONOUN")},
}


def row(lm, tokenizer, model, language, item, row, device):
    """(values, columns) for one masked sentence."""
    slot = SLOT.findall(row["sentence"])[0]
    mask = model.mask_token + "的" if language == "zh" and slot == "$POSS_PRONOUN" else model.mask_token
    scores = mask_scores(lm, tokenizer, model.mask_token, row["sentence"].replace(slot, mask), device)[0]
    probs = F.softmax(scores, dim=-1)
    non_verbalized = tokenizer.convert_ids_to_tokens([int(scores.argmax())])[0]
    pronouns = PRONOUNS[language][slot]
    if language == "en":
        pronouns = [model.en_prefix + p for p in pronouns]
    female, male, neutral = (float(probs[i]) for i in tokenizer.convert_tokens_to_ids(pronouns))
    return ([row["label"], non_verbalized, pronouns[0], female, pronouns[1], male, pronouns[2], neutral],
            ["label", "non_verbalized", "female", "female_prob", "male", "male_prob", "neutral", "neutral_prob"])
