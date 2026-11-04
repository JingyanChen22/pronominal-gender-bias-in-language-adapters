"""The models, and how to load each version of them (base, finetuned, adapter)."""
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "model_training" / "models"  # written by ../model_training
TRAINED = {"finetuned": MODELS_DIR / "finetuned", "adapter": MODELS_DIR / "adapter"}  # <dir>/<trained on>_<short name>
VERSIONS = ("base", "finetuned", "adapter")


@dataclass(frozen=True)
class Model:
    name: str                 # as in the output folders: BERT, XLM-RoBERTa, ...
    type: str                 # MLM or generative
    hub: str                  # Hugging Face name of the base model
    languages: tuple
    tokenizer: str            # transformers class names
    model: str
    mask_token: str = None    # MLM only
    en_prefix: str = ""       # EN only: prefix of a word-initial token (Ġ RoBERTa, ▁ XLM-R)
    en_no_yes: tuple = None   # EN yes_no only: the tokens compared
    en_strip: bool = False    # EN only: drop the first character of the predicted token
    zh_one_mask: bool = False  # ZH only: one mask instead of one per character of the label
    short: str = None         # name of the trained models (finetuned/adapter): <lang>_<short>

    @property
    def versions(self):
        return VERSIONS if self.short else ("base",)


MODELS = {m.name: m for m in [
    Model("BERT", "MLM", "bert-base-uncased", ("en",), "BertTokenizerFast", "BertForMaskedLM", "[MASK]",
          en_no_yes=("no", "yes")),
    Model("RoBERTa", "MLM", "roberta-base", ("en",), "RobertaTokenizerFast", "RobertaForMaskedLM", "<mask>",
          en_prefix="Ġ", en_no_yes=("ĠNo", "ĠYes"), en_strip=True),
    Model("BERT_zh", "MLM", "hfl/chinese-bert-wwm", ("zh",), "BertTokenizerFast", "BertForMaskedLM", "[MASK]"),
    Model("RoBERTa_zh", "MLM", "hfl/chinese-roberta-wwm-ext", ("zh",), "BertTokenizerFast", "BertForMaskedLM",
          "[MASK]"),
    Model("mBERT", "MLM", "bert-base-multilingual-uncased", ("en", "zh"), "BertTokenizerFast", "BertForMaskedLM",
          "[MASK]", en_no_yes=("no", "yes"), short="mbert"),
    Model("XLM-RoBERTa", "MLM", "xlm-roberta-base", ("en", "zh"), "XLMRobertaTokenizerFast", "XLMRobertaForMaskedLM",
          "<mask>", en_prefix="▁", en_no_yes=("▁No", "▁Yes"), en_strip=True, zh_one_mask=True, short="xlmr"),
    Model("T5", "generative", "t5-base", ("en",), "T5TokenizerFast", "T5ForConditionalGeneration"),
    Model("mT5", "generative", "google/mt5-base", ("en", "zh"), "T5TokenizerFast", "T5ForConditionalGeneration"),
    Model("FLAN", "generative", "google/flan-t5-base", ("en",), "AutoTokenizer", "AutoModelForSeq2SeqLM"),
    Model("ChatLM", "generative", "charent/ChatLM-mini-Chinese", ("zh",), "AutoTokenizer", "AutoModelForSeq2SeqLM"),
]}


def trained_path(model, version, trained_on):
    return TRAINED[version] / f"{trained_on}_{model.short}"


def output_name(model, language, version, trained_on, name=None):
    """mBERT, or mBERT_zh for a model trained on the other language."""
    if name:
        return name
    return model.name if version == "base" or trained_on == language else f"{model.name}_{trained_on}"


def load(model, version, path=None, adapter_name="mlm", device="cpu"):
    """(model, tokenizer). path: the finetuned model or the adapter folder (base: the Hugging Face name)."""
    import transformers
    tokenizer_class, model_class = getattr(transformers, model.tokenizer), getattr(transformers, model.model)
    if version == "base":
        tokenizer = tokenizer_class.from_pretrained(model.hub)
        lm = model_class.from_pretrained(model.hub, trust_remote_code=model.name == "ChatLM")
    elif version == "finetuned":
        tokenizer = transformers.AutoTokenizer.from_pretrained(path)
        lm = transformers.AutoModelForMaskedLM.from_pretrained(path)
    else:  # adapter: the base model with the trained MLM adapter and head on top
        tokenizer = transformers.AutoTokenizer.from_pretrained(path)
        lm = model_class.from_pretrained(model.hub)
        if not hasattr(lm, "load_adapter"):
            raise SystemExit("adapters need adapter-transformers, the library they were trained with "
                             "(pip install adapter-transformers; it replaces transformers)")
        lm.load_adapter(f"{path}/{adapter_name}", config=f"{path}/{adapter_name}/adapter_config.json",
                        with_head=True)
        lm.set_active_adapters(adapter_name)
    return lm.to(device).eval(), tokenizer
