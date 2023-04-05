# reference: https://huggingface.co/transformers/model_doc/encoderdecoder.html#encoderdecodermodel

from torch import nn
import torch
from tqdm import tqdm
from transformers import AutoModel


class ConvEncDec(nn.Module):
    def __init__(self, pretrained_type, tokenizer):
        super().__init__()
        self.pretrained_type = pretrained_type
        self.transformer = AutoModel.from_pretrained(pretrained_type)
        self.tokenizer = tokenizer
        self.resize_embedding()
        self.lm_head = nn.Linear(self.transformer.config.hidden_size, len(tokenizer), bias=False)

    def resize_embedding(self):
        if 't5' in self.pretrained_type:
            size_before = self.transformer.encoder.embed_tokens.weight.size(0)
            self.transformer.encoder.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                # first reduce the random jitter of the initialization
                self.transformer.encoder.embed_tokens.weight[size_before:] *= 0.1
                # next center it on the endoftext token
                self.transformer.encoder.embed_tokens.weight[
                size_before:
                ] += self.transformer.encoder.embed_tokens.weight[size_before - 1].unsqueeze(0)
            size_before = self.transformer.decoder.embed_tokens.weight.size(0)
            self.transformer.decoder.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                # first reduce the random jitter of the initialization
                self.transformer.decoder.embed_tokens.weight[size_before:] *= 0.1
                # next center it on the endoftext token
                self.transformer.decoder.embed_tokens.weight[
                size_before:
                ] += self.transformer.decoder.embed_tokens.weight[size_before - 1].unsqueeze(0)
        elif 'bart' in self.pretrained_type:
            size_before = self.transformer.encoder.embed_tokens.weight.size(0)
            self.transformer.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                # first reduce the random jitter of the initialization
                self.transformer.encoder.embed_tokens.weight[size_before:] *= 0.1
                # next center it on the endoftext token
                self.transformer.encoder.embed_tokens.weight[
                size_before:
                ] += self.transformer.encoder.embed_tokens.weight[size_before - 1].unsqueeze(0)
                # first reduce the random jitter of the initialization
                self.transformer.decoder.embed_tokens.weight[size_before:] *= 0.1
                # next center it on the endoftext token
                self.transformer.decoder.embed_tokens.weight[
                size_before:
                ] += self.transformer.decoder.embed_tokens.weight[size_before - 1].unsqueeze(0)

    def forward(self, persona_query_input, target_input=None, is_train=False, return_loss=False, input_data=None):
        input_ids = persona_query_input['input_ids']
        input_attentions = persona_query_input['attention_mask']
        if is_train:
            target_ids = target_input['input_ids']
            target_attentions = target_input['attention_mask']
            bos_tensor = torch.full((input_ids.shape[0], 1), self.tokenizer.bos_token_id, device=input_ids.device)
            bos_attn = torch.ones_like(bos_tensor, device=bos_tensor.device)
            decoder_input_ids = torch.cat((bos_tensor, target_ids), dim=1)
            decoder_input_attentions = torch.cat((bos_attn, target_attentions), dim=1)
        position_ids = (
                input_attentions.cumsum(dim=-1, dtype=torch.int64) - 1
        ).clamp_(min=0)
        outputs = self.transformer(input_ids=input_ids,
                                   attention_mask=input_attentions,
                                   decoder_input_ids=decoder_input_ids,
                                   decoder_attention_mask=decoder_input_attentions)
        last_hidden_state = outputs[0]
        lm_logits = self.lm_head(last_hidden_state)
        return lm_logits, outputs

    def generate(self, input_data, generate_len=128, device='cuda', using_cache=True, return_text=True, show_progress=False, *args, **kwargs):
        self.eval()
        persona_query_input = input_data['persona_query_input']
        with torch.no_grad():
            encoder_input_ids = persona_query_input['input_ids'].to(device)
            encoder_attentions = persona_query_input['attention_mask'].to(device)
            bos_tensor = torch.full((encoder_input_ids.shape[0], 1), self.tokenizer.bos_token_id, device=encoder_input_ids.device)
            active_attn = torch.ones_like(bos_tensor, device=bos_tensor.device)
            decoder_input_attentions = active_attn
            decoder_input_ids = bos_tensor
            generated = None
            past_key_values = None
            if show_progress:
                iterator = tqdm(range(generate_len))
            else:
                iterator = range(generate_len)
            for _ in iterator:
                if not using_cache:
                    past_key_values = None
                if past_key_values:
                    outputs = self.transformer(decoder_input_ids=decoder_input_ids[:, -1:],
                                               decoder_attention_mask=decoder_input_attentions[:, -1:],
                                               decoder_position_ids=position_ids[:, -1:],
                                               input_ids=encoder_input_ids,
                                               attention_masks=encoder_attentions,
                                               past_key_values=past_key_values,
                                               use_cache=True)
                else:
                    outputs = self.transformer(decoder_input_ids=decoder_input_ids,
                                               decoder_attention_mask=decoder_input_attentions,
                                               decoder_position_ids=position_ids,
                                               input_ids=encoder_input_ids,
                                               attention_masks=encoder_attentions,
                                               past_key_values=past_key_values,
                                               use_cache=True)
                past_key_values = outputs.past_key_values
                last_hidden_state = outputs[0]
                lm_logits = self.lm_head(last_hidden_state)
                last_digits = lm_logits[:, -1:, :]
                last_digits = torch.argmax(last_digits, dim=-1)
                input_ids = torch.cat((input_ids, last_digits), dim=1)
                input_attentions = torch.cat((input_attentions, active_attn), dim=1)
                if generated is None:
                    generated = last_digits
                else:
                    generated = torch.cat((generated, last_digits), dim=1)
                position_ids = (
                        input_attentions.cumsum(dim=-1, dtype=torch.int64) - 1
                ).clamp_(min=0)
            np_gen = cut_special_tokens(generated, self.tokenizer)
            if return_text:
                return self.tokenizer.batch_decode(np_gen)
            return np_gen


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

