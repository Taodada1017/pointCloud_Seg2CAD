from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

local_path = "/home/chenxiaozhi/develop/weights/bert-base-uncased"

def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )

    tokenizer = AutoTokenizer.from_pretrained(local_path)
    return tokenizer


def get_pretrained_language_model(text_encoder_type, bert_base_uncased_path=None):
    if text_encoder_type == "bert-base-uncased":
        # if is_bert_model_use_local_path(bert_base_uncased_path):
        #     return BertModel.from_pretrained(bert_base_uncased_path)
        return BertModel.from_pretrained(local_path)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)
    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))

def is_bert_model_use_local_path(bert_base_uncased_path):
    return bert_base_uncased_path is not None and len(bert_base_uncased_path) > 0
