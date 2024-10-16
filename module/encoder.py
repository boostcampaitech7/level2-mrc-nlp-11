from transformers import (
    BertPreTrainedModel,
    BertModel,
    RobertaPreTrainedModel,
    RobertaModel,
    EelectraModel,
    ElectraPreTrainedModel,
)


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

        self.electra = EelectraModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.electra(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        cls_output = outputs.last_hidden_state[:, 0]  # shape: (batch_size, hidden_size)
        return cls_output
