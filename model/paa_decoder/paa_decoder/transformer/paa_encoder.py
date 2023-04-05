from torch import nn
from transformers import BertConfig, BertModel, AutoModel


class PAAEncoder(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        encoder_config = config.paa_transformer.encoder
        if encoder_config.using_pretrained is not None:
            encoder = AutoModel.from_pretrained(encoder_config.using_pretrained)
        else:
            config_encoder = BertConfig()
            config_encoder.update(encoder_config)
            encoder = BertModel(config=config_encoder)
        encoder = self.resize_embedding(encoder)

        self.encoder = encoder

    def resize_embedding(self, transformer):
        transformer.resize_token_embeddings(len(self.tokenizer))
        return transformer

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states
