"""Language-specific rules: how templates are tokenised, which words fill each slot, and how tokens are joined."""
import re
from dataclasses import dataclass

OCCUPATION, PARTICIPANT = "$OCCUPATION", "$PARTICIPANT"
NOM, POSS, ACC = "$NOM_PRONOUN", "$POSS_PRONOUN", "$ACC_PRONOUN"
PRONOUN_SLOTS = (NOM, POSS, ACC)
GENDERS = ("female", "male", "neutral")  # row order in every gendered file


@dataclass(frozen=True)
class Language:
    code: str
    pronouns: dict          # gender -> {slot: pronoun}
    masked: dict            # slot -> placeholder kept in the "masked" sentences
    joiner: str             # how tokens are glued back into a sentence
    someone: str            # bleached participant ("someone" / 某人)
    token_pattern: str = None  # regex for tokenising; None means split on whitespace

    def tokenize(self, sentence):
        if self.token_pattern is None:
            return sentence.split()
        return re.findall(self.token_pattern, sentence)

    def join(self, tokens):
        return self.joiner.join(tokens)

    def possessive(self, entity):
        """Replacement for $POSS_PRONOUN when the pronoun is replaced by its referent."""
        return f"the {entity}'s" if self.code == "en" else entity + "的"

    def nominal(self, entity):
        """Replacement for $NOM_PRONOUN / $ACC_PRONOUN when the pronoun is replaced by its referent."""
        return f"the {entity}" if self.code == "en" else entity


EN = Language(
    code="en",
    pronouns={
        "male": {NOM: "he", POSS: "his", ACC: "him"},
        "female": {NOM: "she", POSS: "her", ACC: "her"},
        "neutral": {NOM: "they", POSS: "their", ACC: "them"},
    },
    masked={NOM: NOM, POSS: POSS, ACC: ACC},
    joiner=" ",
    someone="someone",
)

ZH = Language(
    code="zh",
    pronouns={
        "male": {NOM: "他", POSS: "他的", ACC: "他"},
        "female": {NOM: "她", POSS: "她的", ACC: "她"},
        "neutral": {NOM: "TA", POSS: "TA的", ACC: "TA"},
    },
    masked={NOM: NOM, POSS: POSS, ACC: ACC},  # no 的 after $POSS_PRONOUN: task execution adds [MASK]的
    joiner="",
    someone="某人",
    token_pattern=r"[一-龥a-zA-Z0-9_$]+|[，。；]",
)

LANGUAGES = {"en": EN, "zh": ZH}


def find_slot(tokens):
    """Every template has exactly one pronoun slot; return it."""
    slots = [t for t in tokens if t in PRONOUN_SLOTS]
    assert len(slots) == 1, f"expected one pronoun slot, found {slots} in {tokens}"
    return slots[0]


def fill_pronoun(tokens, slot_map):
    """Replace the pronoun slot with the word from `slot_map`."""
    return [slot_map.get(t, t) if t in PRONOUN_SLOTS else t for t in tokens]
