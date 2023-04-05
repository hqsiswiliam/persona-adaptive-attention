import torch
from torch import nn
import copy

from tqdm import tqdm
from transformers import GPT2Config
import torch.nn.functional as F
from ..paa_generation_model import PAAGenerationModel

from model.paa_decoder.paa_decoder.transformer.gpt_decoder_PAAGPT import GPT2LMHeadModel
#
from model.paa_decoder.paa_decoder.transformer.load_gptmodel import load_weight


class TransformerPAAGPT(PAAGenerationModel):

    def __init__(self, tokenizer, config, persona_encoder, context_encoder, gpt2_type, n_layer=12, tau=0):
        super(TransformerPAAGPT, self).__init__()
        self.bos_idx = tokenizer.bos_token_id
        self.tokenizer = tokenizer
        self.persona_encoder = persona_encoder
        self.context_encoder = context_encoder
        self.gpt2_type = gpt2_type
        if 'gpt2-' in config.paa_transformer.decoder.base_model:
            gpt_config = GPT2Config.from_pretrained(config.paa_transformer.decoder.base_model)
        else:
            gpt_config = GPT2Config.from_pretrained('gpt2')
        # gpt_config.vocab_size = len(self.tokenizer)
        gpt_config.add_cross_attention = True
        gpt_config.n_layer = n_layer
        decoder = GPT2LMHeadModel(gpt_config, tau=tau, customize_config=config)

        if gpt2_type == "random":
            self.decoder = decoder
        else:
            if config.paa_transformer.decoder.base_model == 'distilgpt2':
                state_dict = torch.load('downloaded_LM/distilgpt2-pytorch_model.bin',
                                        map_location='cpu' if not torch.cuda.is_available() else None)
            elif config.paa_transformer.decoder.base_model == 'gpt2':
                state_dict = torch.load('downloaded_LM/gpt2-pytorch_model.bin',
                                        map_location='cpu' if not torch.cuda.is_available() else None)
            elif config.paa_transformer.decoder.base_model == 'gpt2-medium':
                state_dict = torch.load('downloaded_LM/gpt2-medium-pytorch_model.bin',
                                        map_location='cpu' if not torch.cuda.is_available() else None)
            elif config.paa_transformer.decoder.base_model == 'gpt2-large':
                state_dict = torch.load('downloaded_LM/gpt2-large-pytorch_model.bin',
                                        map_location='cpu' if not torch.cuda.is_available() else None)
            elif config.paa_transformer.decoder.base_model == 'microsoft/DialoGPT-small':
                state_dict = torch.load('downloaded_LM/dialogpt-pytorch_model.bin',
                                        map_location='cpu' if not torch.cuda.is_available() else None)
            else:
                raise ValueError("Not supported base model!")
            decoder = load_weight(decoder, state_dict)
            self.decoder = decoder

        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        # self.init_weights()
        self.resize_embedding()
        self.decoder.padding_idx = tokenizer.pad_token_id
        self.add_persona_to_decoder = config.paa_transformer.decoder.add_persona_to_decoder
        self.add_context_to_decoder = config.paa_transformer.decoder.add_context_to_decoder
        if self.add_persona_to_decoder and self.add_context_to_decoder:
            raise ValueError("Not allow to enable both add_persona and context!")
        self.double_heads = config.paa_transformer.decoder.double_heads
        if self.double_heads:
            self.cls_head = nn.Linear(gpt_config.hidden_size, 2)

    def resize_embedding(self):
        size_before = self.decoder.transformer.wte.weight.size(0)
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        with torch.no_grad():
            # first reduce the random jitter of the initialization
            self.decoder.transformer.wte.weight[size_before:] *= 0.1
            # next center it on the endoftext token
            self.decoder.transformer.wte.weight[
            size_before:
            ] += self.decoder.transformer.wte.weight[size_before - 1].unsqueeze(0)

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):

        if self.gpt2_type == "random":
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            for p in self.persona_encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for p in self.context_encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def generate(self, input_data, max_len=64, return_text=True, temperature=0.0, show_progress=False, *args, **kwargs):
        self.eval()
        with torch.no_grad():
            device = self.decoder.device
            persona_input = input_data['persona_input']['input_ids'].to(device)
            persona_attn = input_data['persona_input']['attention_mask'].to(device)
            persona_encoder_output = self.persona_encoder(persona_input, persona_attn)
            context_input = input_data['query_input']['input_ids'].to(device)
            context_attn = input_data['query_input']['attention_mask'].to(device)
            context_encoder_output = self.context_encoder(context_input, context_attn)
            decoder_input = torch.full((context_encoder_output.shape[0], 1),
                                       self.tokenizer.bos_token_id,
                                       device=device)
            decoder_attention = torch.ones_like(decoder_input)
            if self.add_persona_to_decoder:
                decoder_input = torch.cat((persona_input, decoder_input), dim=-1)
                decoder_attention = torch.cat((persona_attn, decoder_attention), dim=-1)
            if self.add_context_to_decoder:
                decoder_input = torch.cat((context_input, decoder_input), dim=-1)
                decoder_attention = torch.cat((context_attn, decoder_attention), dim=-1)
            if show_progress:
                iterator = tqdm(range(max_len))
            else:
                iterator = range(max_len)
            for _ in iterator:
                result = self.decoder(input_ids=decoder_input,
                                      attention_mask=decoder_attention,
                                      persona_encoder_hidden_states=persona_encoder_output,
                                      persona_encoder_attention_mask=persona_attn,
                                      context_encoder_hidden_states=context_encoder_output,
                                      context_encoder_attention_mask=context_attn)
                next_token_logits = result[0][:, -1, :] / (temperature if temperature > 0 else 1.)
                temperature_next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
                _logits = torch.argmax(result.logits[:, -1:, :], dim=-1)
                if temperature == 0:
                    decoder_input = torch.cat((decoder_input, _logits), dim=-1)
                else:
                    decoder_input = torch.cat((decoder_input, temperature_next_token), dim=-1)
                decoder_attention = torch.cat((decoder_attention, torch.ones_like(temperature_next_token)), dim=1)
            # remove Persona
            if self.add_persona_to_decoder:
                persona_index = persona_input.shape[1]
                decoder_input = decoder_input[:, persona_index:]
            if self.add_context_to_decoder:
                context_index = context_input.shape[1]
                decoder_input = decoder_input[:, context_index:]
            # remove [BOS]
            decoder_input = decoder_input[:, 1:]
            # find [EOS]
            eos_token_id = self.tokenizer.eos_token_id
            truncated_result = []
            for entry in decoder_input:
                entry = entry.cpu().detach().numpy().tolist()
                if eos_token_id in entry:
                    eos_index = entry.index(eos_token_id)
                    entry = entry[:eos_index]
                truncated_result.append(entry)
            if return_text:
                return self.tokenizer.batch_decode(truncated_result)
            else:
                return truncated_result

    def forward(self, input_data, *args, **kwargs):
        device = self.decoder.device
        persona_input = input_data['persona_input']['input_ids'].to(device)
        persona_attn = input_data['persona_input']['attention_mask'].to(device)
        persona_encoder_output = self.persona_encoder(persona_input, persona_attn)
        context_input = input_data['query_input']['input_ids'].to(device)
        context_attn = input_data['query_input']['attention_mask'].to(device)
        context_encoder_output = self.context_encoder(context_input, context_attn)
        decoder_input_ids = input_data['target_input']['input_ids']
        decoder_mask = input_data['target_input']['attention_mask']
        bos_tensor = torch.full((decoder_input_ids.shape[0], 1), self.tokenizer.bos_token_id,
                                device=device)
        bos_attn = torch.ones_like(bos_tensor, device=bos_tensor.device)
        decoder_input_ids = torch.cat((bos_tensor, decoder_input_ids), dim=1)
        _decoder_attention_mask = torch.cat((bos_attn, decoder_mask), dim=1)
        if self.add_persona_to_decoder:
            decoder_input_ids = torch.cat((persona_input, decoder_input_ids), dim=1)
            _decoder_attention_mask = torch.cat((persona_attn, _decoder_attention_mask), dim=1)
        if self.add_context_to_decoder:
            decoder_input_ids = torch.cat((context_input, decoder_input_ids), dim=1)
            _decoder_attention_mask = torch.cat((context_attn, _decoder_attention_mask), dim=1)
        if self.double_heads:
            cls_tensor = torch.full((decoder_input_ids.shape[0], 1), self.tokenizer.cls_token_id,
                                    device=device)
            cls_attn = torch.ones_like(cls_tensor, device=cls_tensor.device)
            decoder_input_ids = torch.cat((decoder_input_ids, cls_tensor), dim=1)
            _decoder_attention_mask = torch.cat((_decoder_attention_mask, cls_attn), dim=1)
        result = self.decoder(input_ids=decoder_input_ids,
                              attention_mask=_decoder_attention_mask,
                              persona_encoder_hidden_states=persona_encoder_output,
                              persona_encoder_attention_mask=persona_attn,
                              context_encoder_hidden_states=context_encoder_output,
                              context_encoder_attention_mask=context_attn,
                              output_hidden_states=True)
        logits = result.logits
        past = result.past_key_values
        if self.add_persona_to_decoder:
            logits = logits[:, persona_input.shape[1]:, :]
        if self.add_context_to_decoder:
            logits = logits[:, context_input.shape[1]:, :]
        if self.double_heads:
            past = self.cls_head(result.hidden_states[-1][:, -1, :])
            logits = logits[:, :-1, :]
        return logits, past

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, past, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.enc_output, self.mask_enc = self.encoder(visual)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc, past=past)
