"""Coreference resolution: one output row per prompt row.

MLM: a mask is appended to the prompt and the probabilities of the candidate answers at the mask are compared.
Generative: the model completes the prompt freely (evaluation reads the answer from the text).
"""
import torch
import torch.nn.functional as F


def mask_scores(lm, tokenizer, mask_token, sentence, device):
    """Scores over the vocabulary at every mask, in order: [masks, vocabulary]."""
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0).to(device)
    positions = (input_ids[0] == tokenizer.convert_tokens_to_ids(mask_token)).nonzero()[:, 0]
    with torch.no_grad():
        return lm(input_ids)[0][0, positions]


def length_corrected(prob, entity, label):
    """ZH multi-mask: the masks are as many as the label's characters; rescale an entity of another length."""
    return prob / len(entity) * len(label) if len(entity) != len(label) else prob


def mlm_row(lm, tokenizer, model, language, item, row, device):
    """(values, columns) for one prompt row."""
    ids = tokenizer.convert_tokens_to_ids
    single, kind = item.entities == "single_entity", item.kind
    if language == "en":
        n_masks, ending = 1, " " + model.mask_token + "."
    else:
        n_masks = 1 if model.zh_one_mask else len(row["label"])
        ending = model.mask_token * n_masks + "。"
    scores = mask_scores(lm, tokenizer, model.mask_token, row["sentence"] + ending, device)[:n_masks]
    probs = F.softmax(scores, dim=-1)
    non_verbalized = "".join(tokenizer.convert_ids_to_tokens(scores.argmax(-1).tolist()))
    if language == "en" and model.en_strip and len(non_verbalized) > 1:
        non_verbalized = non_verbalized[1:]

    if kind == "yes_no":
        no, yes = model.en_no_yes if language == "en" else ("否", "是")
        no_prob, yes_prob = probs[0][ids(no)], probs[0][ids(yes)]
        verbalized = (["no", "yes"] if language == "en" else ["否", "是"])[no_prob < yes_prob]
        return ([row["gender"], float(no_prob), float(yes_prob), verbalized, non_verbalized, row["label"]],
                ["gender", "no_prob", "yes_prob", "verbalized", "non_verbalized", "label"])

    def entity_prob(entity, characters=None):
        """EN and ZH one-mask: the entity as one token. ZH: product over its first `characters` characters."""
        if language == "en" or model.zh_one_mask:
            return probs[0][ids(model.en_prefix + entity if language == "en" else entity)]
        prob = 1
        for i in range(characters):
            prob *= probs[i][ids(entity[i])]
        return prob

    if single:
        label_prob = entity_prob(row["label"], len(row["label"]))
        return ([row["gender"], row["label"], float(label_prob), non_verbalized],
                ["gender", "label", "label_prob", "non_verbalized"])

    occupation, participant = row["occupation"], row["participant"]
    characters = min(len(occupation), len(participant))
    occupation_prob, participant_prob = entity_prob(occupation, characters), entity_prob(participant, characters)
    if language == "zh" and not model.zh_one_mask:
        occupation_prob = length_corrected(occupation_prob, occupation, row["label"])
        participant_prob = length_corrected(participant_prob, participant, row["label"])
    verbalized = [occupation, participant][occupation_prob < participant_prob]
    values = [occupation, float(occupation_prob), participant, float(participant_prob), verbalized, non_verbalized,
              row["label"]]
    columns = ["occupation", "occupation_prob", "participant", "participant_prob", "verbalized", "non_verbalized",
               "label"]
    if kind != "no_pronoun_who":
        values, columns = [row["gender"]] + values, ["gender"] + columns
    return values, columns


def generate(lm, tokenizer, model, language, sentence, device):
    if model.name == "ChatLM":
        encoded = tokenizer([sentence])
        outs = lm.my_generate(input_ids=torch.LongTensor(encoded["input_ids"]).to(device),
                              attention_mask=torch.LongTensor(encoded["attention_mask"]).to(device),
                              max_seq_len=15, search_type="beam")
        return tokenizer.batch_decode(outs.cpu().numpy(), skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True)[0]
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = lm.generate(input_ids, max_length=15 if language == "zh" else 10)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def generative_row(lm, tokenizer, model, language, item, row, device):
    """(values, columns) for one prompt row. label_entity: lets evaluation score answers that name a person."""
    text = generate(lm, tokenizer, model, language, row["sentence"], device)
    yes_no = item.kind == "yes_no"
    if item.entities == "single_entity":
        values, columns = [row["gender"], text, row["label"]], ["gender", "non_verbalized", "label"]
        if yes_no:
            values, columns = values + [row["label_entity"]], columns + ["label_entity"]
        return values, columns
    if item.kind == "no_pronoun_who":
        return ([text, row["label"], row["occupation"], row["participant"]],
                ["non_verbalized", "label", "occupation", "participant"])
    if yes_no:
        return ([row["gender"], text, row["label"], row["label_entity"], row["occupation"], row["participant"]],
                ["gender", "non_verbalized", "label", "label_entity", "occupation", "participant"])
    return ([row["gender"], text, row["label"], row["occupation"], row["participant"]],
            ["gender", "non_verbalized", "label", "occupation", "participant"])


ROW = {"MLM": mlm_row, "generative": generative_row}
