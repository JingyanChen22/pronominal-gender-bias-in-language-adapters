"""Turn a sentence row into a prompt by filling a template's [TEXT], [PRON], [MENTION1] and [MENTION2]."""
TEXT, PRON, MENTION1, MENTION2 = "[TEXT]", "[PRON]", "[MENTION1]", "[MENTION2]"
ANSWERS = {"en": ("yes", "no"), "zh": ("是", "否")}


def fill(prompt, row, double, language):
    """[MENTION1] is the occupation in double-entity prompts and the only person in single-entity ones."""
    text = prompt.replace(TEXT, row["sentence"]).replace(PRON, row["pronoun"])
    text = text.replace(MENTION1, row["occupation"] if double else row["label"])
    if double:
        text = text.replace(MENTION2, row["participant"])
    return text.replace(" ", "") if language.code == "zh" else text


def prompt_rows(prompt, rows, double, yes_no, language):
    """One prompt per row. Yes/no prompts ask about [MENTION1], so the answer is yes when it is the referent."""
    yes, no = ANSWERS[language.code]
    out = []
    for _, row in rows.iterrows():
        prompted = {"sentence": fill(prompt, row, double, language)}
        if yes_no:
            mention1 = row["occupation"] if double else row["label"]
            correct = mention1 == row["label"]
            prompted["label"] = yes if correct else no
            prompted["gender"] = row["gender"]
            prompted["label_entity"] = mention1 if correct else row["participant"]
        else:
            prompted["label"] = row["label"]
            prompted["gender"] = row["gender"]
        if double:
            prompted["occupation"] = row["occupation"]
            prompted["participant"] = row["participant"]
        out.append(prompted)
    return out
