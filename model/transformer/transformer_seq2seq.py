# reference: https://huggingface.co/transformers/model_doc/encoderdecoder.html#encoderdecodermodel

from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, BertGenerationEncoder, \
    BertGenerationDecoder, AutoModel, GPT2Config, GPT2Model, BertModel, GPT2LMHeadModel
from torch import nn
import torch


class TransformerSeq2Seq(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        encoder_pretrained = config.transformer.encoder.using_pretrained
        decoder_pretrained = config.transformer.decoder.using_pretrained
        if encoder_pretrained is not None:
            if 'bert' in encoder_pretrained:
                encoder = BertGenerationEncoder.from_pretrained(encoder_pretrained)
            elif 'gpt' in encoder_pretrained:
                encoder = GPT2Model.from_pretrained(encoder_pretrained)
        else:
            config_encoder = BertConfig()
            config_encoder.update(config.transformer.encoder)
            encoder = BertModel(config=config_encoder)
        if decoder_pretrained is not None:
            decoder = GPT2LMHeadModel.from_pretrained(decoder_pretrained, is_decoder=True, add_cross_attention=True)
        else:
            is_bert = config.transformer.decoder.model_type == 'bert'
            config_class = BertConfig if is_bert else GPT2Config
            decoder_class = BertGenerationDecoder if is_bert else GPT2Model
            config_decoder = config_class()
            config_decoder.update(config.transformer.decoder)
            config_decoder.is_decoder = True
            config_decoder.add_cross_attention = True
            decoder = decoder_class(config=config_decoder)
        self.tokenizer = tokenizer
        encoder = self.resize_embedding(encoder)
        decoder = self.resize_embedding(decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.seq2seq_model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
        self.seq2seq_model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.config = config

    def resize_embedding(self, transformer):
        transformer.resize_token_embeddings(len(self.tokenizer))
        return transformer

    def forward(self, input_data, return_loss, **kwargs):
        input_ids = input_data['persona_query_input']['input_ids']
        attention_mask = input_data['persona_query_input']['attention_mask']
        decoder_input_ids = input_data['target_input']['input_ids']
        decoder_mask = input_data['target_input']['attention_mask']
        bos_tensor = torch.full((decoder_input_ids.shape[0], 1), self.tokenizer.bos_token_id, device=input_ids.device)
        bos_attn = torch.ones_like(bos_tensor, device=bos_tensor.device)
        decoder_input_ids = torch.cat((bos_tensor, decoder_input_ids), dim=1)
        decoder_attention_mask = torch.cat((bos_attn, decoder_mask), dim=1)
        outputs = self.seq2seq_model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     decoder_attention_mask=decoder_attention_mask,
                                     decoder_input_ids=decoder_input_ids)
        loss, logits = outputs.loss, outputs.logits
        if return_loss:
            return logits, loss
        return logits

    def generate(self, input_data, *args, return_text=True, **kwargs):
        device = self.seq2seq_model.device
        input_ids = input_data['persona_query_input']['input_ids'].to(device)
        attention_mask = input_data['persona_query_input']['attention_mask'].to(device)
        result = self.seq2seq_model.generate(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             decoder_start_token_id=self.tokenizer.bos_token_id, max_length=64)
        cut_result = cut_special_tokens(result[:,1:], self.tokenizer)
        if return_text:
            text_result = self.tokenizer.batch_decode(cut_result)
            return text_result
        return cut_result


def cut_special_tokens(generated, tokenizer):
    np_gen = generated.detach().cpu().numpy().tolist()
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    for index in range(len(np_gen)):
        if eos_token_id in np_gen[index]:
            eos_index = np_gen[index].index(eos_token_id)
            np_gen[index] = np_gen[index][:eos_index]
        elif pad_token_id in np_gen[index]:
            pad_index = np_gen[index].index(pad_token_id)
            np_gen[index] = np_gen[index][:pad_index]
    return np_gen
