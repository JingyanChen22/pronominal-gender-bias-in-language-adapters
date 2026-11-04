"""Who-question prompts: attaching the curated question to a sentence, and drafting questions automatically."""
from .languages import ACC, NOM, POSS, find_slot


def attach(language, sentence, question, quote=True):
    """English: '<sentence>' <question>, or <sentence> <question>. Chinese: ‘<sentence>’<question>, or no quotes."""
    if language.code == "en":
        return f"'{sentence}' {question}" if quote else f"{sentence} {question}"
    return f"‘{sentence}’{question}" if quote else sentence + question


def draft(language, tokens):
    """Automatic who-question for a tokenised template. Often ungrammatical, so it is only a starting
    point for the hand-corrected questions in sources/<lang>/who_questions.tsv."""
    slot = find_slot(tokens)
    i = tokens.index(slot)
    rest = tokens[i + 1:]
    if language.code == "en":
        who = {NOM: "who ", POSS: "whose ", ACC: "who "}[slot]
        question = " In the previous sentence, " + who + " ".join(rest) + " The"
        return slot, question.replace(".", "?")
    who = {NOM: "谁", POSS: "谁的", ACC: "谁"}[slot]
    prefix = tokens[i - 1:i] if slot == ACC else []  # keep the verb before an object pronoun
    question = "在前一句中，" + "".join(prefix) + who + "".join(rest)
    return slot, question.replace("。", "？")
