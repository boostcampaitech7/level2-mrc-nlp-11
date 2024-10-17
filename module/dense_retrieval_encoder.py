from transformers import (
    BertPreTrainedModel,
    BertModel,
    RobertaPreTrainedModel,
    RobertaModel,
    ElectraModel,
    ElectraPreTrainedModel,
    AutoModelForSequenceClassification,
)
from torch import nn


class BertEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        pooled_output = outputs["pooler_output"]
        return pooled_output


class RobertaEncoder(RobertaPreTrainedModel):

    def __init__(self, config):
        super(RobertaEncoder, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        pooled_output = outputs["pooler_output"]
        return pooled_output


class ElectraEncoder(ElectraPreTrainedModel):

    def __init__(self, config):
        super(ElectraEncoder, self).__init__(config)

        self.electra = ElectraModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.electra(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        cls_output = outputs.last_hidden_state[:, 0]  # shape: (batch_size, hidden_size)
        return cls_output


class CrossEncoder(nn.Module):
    def __init__(self, plm_name):
        super().__init__()
        self.plm_name = plm_name
        self.plm = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=plm_name, num_labels=1, use_auth_token=True
        )

    def forward(self, inputs):
        x = self.plm(**inputs)["logits"]
        return x
