# reference: https://huggingface.co/transformers/model_doc/encoderdecoder.html#encoderdecodermodel

from torch import nn
import torch
from tqdm import tqdm
from transformers import GPT2Model, GPT2Config


class ConvGPT(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        gpt_config = config.causal_decoder
        decoder_type = gpt_config.decoder_type
        preserve_weight = gpt_config.preserve_weight
        self.transformer = None
        if preserve_weight:
            self.transformer = GPT2Model.from_pretrained(decoder_type)
        else:
            gpt_config = GPT2Config.from_pretrained(decoder_type)
            self.transformer = GPT2Model(config=gpt_config)
        self.config = config
        self.tokenizer = tokenizer
        self.resize_embedding()
        self.lm_head = nn.Linear(self.transformer.config.n_embd, self.transformer.config.vocab_size, bias=False)

    def resize_embedding(self):
        size_before = self.transformer.wte.weight.size(0)
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        with torch.no_grad():
            # first reduce the random jitter of the initialization
            self.transformer.wte.weight[size_before:] *= 0.1
            # next center it on the endoftext token
            self.transformer.wte.weight[
            size_before:
            ] += self.transformer.wte.weight[size_before - 1].unsqueeze(0)

    def forward(self, persona_query_input, target_input=None, is_train=False, return_loss=False, input_data=None):
        input_ids = persona_query_input['input_ids']
        input_attentions = persona_query_input['attention_mask']
        if is_train:
            target_ids = target_input['input_ids']
            target_attentions = target_input['attention_mask']
            bos_tensor = torch.full((input_ids.shape[0], 1), self.tokenizer.bos_token_id, device=input_ids.device)
            bos_attn = torch.ones_like(bos_tensor, device=bos_tensor.device)
            input_ids = torch.cat((input_ids, bos_tensor, target_ids), dim=1)
            input_attentions = torch.cat((input_attentions, bos_attn, target_attentions), dim=1)
        position_ids = (
                input_attentions.cumsum(dim=-1, dtype=torch.int64) - 1
        ).clamp_(min=0)
        outputs = self.transformer(input_ids=input_ids,
                                   attention_mask=input_attentions,
                                   position_ids=position_ids)
        last_hidden_state = outputs[0]
        lm_logits = self.lm_head(last_hidden_state)
        lm_logits = lm_logits[:, persona_query_input['input_ids'].shape[1]:, :]
        return lm_logits, outputs

    def generate(self, input_data, generate_len=128, device='cuda', using_cache=True, return_text=True, show_progress=False, *args, **kwargs):
        self.eval()
        persona_query_input = input_data['persona_query_input']
        with torch.no_grad():
            input_ids = persona_query_input['input_ids'].to(device)
            input_attentions = persona_query_input['attention_mask'].to(device)
            bos_tensor = torch.full((input_ids.shape[0], 1), self.tokenizer.bos_token_id, device=input_ids.device)
            active_attn = torch.ones_like(bos_tensor, device=bos_tensor.device)
            input_attentions = torch.cat((input_attentions, active_attn), dim=1)
            input_ids = torch.cat((input_ids, bos_tensor), dim=1)
            position_ids = (
                    input_attentions.cumsum(dim=-1, dtype=torch.int64) - 1
            ).clamp_(min=0)
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
                    outputs = self.transformer(input_ids=input_ids[:, -1:],
                                               attention_mask=input_attentions[:, -1:],
                                               position_ids=position_ids[:, -1:],
                                               past_key_values=past_key_values,
                                               use_cache=True)
                else:
                    outputs = self.transformer(input_ids=input_ids,
                                               attention_mask=input_attentions,
                                               position_ids=position_ids,
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


def decoding(model, dataloader):
    pass

# if __name__ == '__main__':
#     from evaluation.evaluate_causal_decoder import evaluate
#     from utils.get_tokenizer import get_tokenizer
#     from config_loader.config import get_config
#     from dataset.dataset import PersonaChatDataset, get_dataloader
#     from utils.save_load_model import load_model
#     from evaluation.evaluation_helper import eval_distinct_avg
#
#     config = get_config('config/causal_decoder/distil-gpt2.yml')
#     tokenizer = get_tokenizer(config.tokenizer.vocab)
#     max_context_turns = config.dataset.max_context_turns
#     valid_dataset = PersonaChatDataset(config.dataset.valid, tokenizer.sep_token, max_context_turns=max_context_turns)
#     valid_dataloader = get_dataloader(valid_dataset, tokenizer, config, num_workers=None, batch_size_ratio=4)
#     test_dataset = PersonaChatDataset(config.dataset.test, tokenizer.sep_token, max_context_turns=max_context_turns)
#     test_dataloader = get_dataloader(test_dataset, tokenizer, config, num_workers=None, batch_size_ratio=4)
#     model = ConvGPT(config, tokenizer)
#     model.to('cuda')
#     load_model(model, path='save_models/distil-gpt2-LR=1e-05/best.pt')
#     preds_digits = []
#     targets_digits = []
#     preds_texts = []
#     target_texts = []
#     for data in tqdm(test_dataloader, desc='decoding tokens'):
#         preds = model.generate(data['persona_query_input'])
#         target = data['target_input']['input_ids']
#         preds_text = tokenizer.batch_decode(preds)
#         target_text = tokenizer.batch_decode(cut_special_tokens(target, tokenizer))
#         preds_digits += preds
#         targets_digits += target
#         preds_texts += preds_text
#         target_texts += target_text
#     from nltk.translate.bleu_score import corpus_bleu
#     bleu1 = corpus_bleu([[t] for t in target_texts], preds_texts, weights=(1,0,0,0))
#     bleu2 = corpus_bleu([[t] for t in target_texts], preds_texts, weights=(0,1,0,0))
#     bleu3 = corpus_bleu([[t] for t in target_texts], preds_texts, weights=(0,0,1,0))
#     bleu4 = corpus_bleu([[t] for t in target_texts], preds_texts, weights=(0,0,0,1))
#     avg_bleu = corpus_bleu([[t] for t in target_texts], preds_texts, weights=(0.25,0.25,0.25,0.25))
#     ppl_score = evaluate(model,test_dataloader,tokenizer, 'cuda', config)
#     dist1, dist2, avg_dist = eval_distinct_avg(preds_texts)
