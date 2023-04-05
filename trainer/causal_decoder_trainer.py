from torch import optim
import torch
from tqdm import tqdm
import numpy as np
from transformers import get_constant_schedule

from dataset.dataset import PersonaChatDataset, get_dataloader
from evaluation.evaluate_causal_decoder import evaluate
from evaluation.evaluation_helper import write_to_file
from utils.eprint import eprint
from utils.get_model_by_config import get_model_via_config
from utils.get_tokenizer import get_tokenizer
from utils.logger import Logger
from utils.save_load_model import save_model, load_model
import os
from torch import nn


def train_one_epoch(causal_decoder_model, optimizer, scheduler, train_dataloader, tokenizer, config, device,
                    prefix_message=''):
    # start train one epoch
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
    cls_criterion = nn.CrossEntropyLoss(reduction='mean')
    pbar = tqdm(train_dataloader, total=train_dataloader.__len__())
    pbar.set_description(prefix_message)
    losses = []
    for input_data in pbar:
        causal_decoder_model.train()
        optimizer.zero_grad()
        logits, _cls = causal_decoder_model(input_data=input_data,
                                            return_loss=True,
                                            is_train=True,
                                            persona_query_input=input_data['persona_query_input'].to(device),
                                            target_input=input_data['target_input'].to(device),
                                            )
        logits = logits[:, :-1, :]  # we ignore [EOS] -> [PAD] here
        if config.paa_transformer.decoder.double_heads:
            revised_input_ids = []
            is_candidates = input_data['is_candidate']
            for index in range(input_data['target_input']['input_ids'].shape[0]):
                is_cand, revised_input = is_candidates[index], input_data['target_input']['input_ids'][index]
                if is_cand:
                    # Ignore the generative loss here!
                    revised_input_id = (torch.ones_like(revised_input.cpu().detach().clone()) * tokenizer.pad_token_id)
                else:
                    revised_input_id = revised_input.cpu().detach().clone()
                revised_input_ids.append(revised_input_id)
            revised_input_ids_pt = torch.stack(revised_input_ids)
            generative_loss = criterion(logits.contiguous().view(-1, logits.shape[-1]),
                                        revised_input_ids_pt.reshape(-1).to(device))
            target_tokens = (revised_input_ids_pt != tokenizer.pad_token_id).sum(dim=-1)
            loss = generative_loss
            if target_tokens.sum() > 0:
                losses.append(loss[loss > 0].mean().item())
            loss = loss.sum()
            loss /= target_tokens.sum()
            # 1 means paired, 0 means not pairs
            cls_loss = cls_criterion(_cls,
                                     (torch.tensor(is_candidates) == False).long().to(device))
            if loss is loss.isnan():
                total_loss = cls_loss
            else:
                total_loss = loss + cls_loss
            mloss = np.mean(losses)
            current_lr = optimizer.param_groups[0]['lr']
            total_loss.backward()
            pbar.set_postfix_str(
                "mloss: {:.2f} loss: {:.2f} lr: {:9f} cls_loss: {:2f}".format(mloss, loss, current_lr, cls_loss))
        else:
            loss = criterion(logits.contiguous().view(-1, logits.shape[-1]),
                             input_data['target_input']['input_ids'].reshape(-1).to(device))
            losses.append(loss.mean().item())
            current_lr = optimizer.param_groups[0]['lr']
            ppl_loss = loss.view(logits.shape[:-1]).sum(dim=1)
            loss.view(logits.shape[:-1])
            target_tokens = (input_data['target_input']['input_ids'] != tokenizer.pad_token_id).sum(dim=-1)
            mloss = np.mean(losses)
            loss = loss.sum()
            loss /= target_tokens.sum()
            loss.backward()
            pbar.set_postfix_str("mloss: {:.2f} loss: {:.2f} lr: {:9f}".format(mloss, loss, current_lr))
        # Note, if FP16, not to revise!
        grad_norm = torch.nn.utils.clip_grad_norm_(
            causal_decoder_model.parameters(), config.training.gradient_clip
        )

        optimizer.step()
        scheduler.step()
    return losses


def train(config, device, load_percent, init_path=None):
    num_workers = config.training.num_workers

    logger = Logger(config)
    tokenizer = get_tokenizer(config.tokenizer.vocab)
    max_context_turns = config.dataset.max_context_turns
    add_persona_indicator = config.dataset.add_persona_indicator
    add_role_indicator = config.dataset.add_role_indicator
    # initialize dataset & dataloader
    train_dataset = PersonaChatDataset(config.dataset.train, tokenizer.sep_token, max_context_turns=max_context_turns,
                                       add_persona_indicator=add_persona_indicator,
                                       add_role_indicator=add_role_indicator,
                                       load_percent=load_percent, extend_candidates=config.training.extend_candidates, num_candidates=config.training.num_candidates)
    train_dataloader = get_dataloader(train_dataset, tokenizer, config, shuffle=False, num_workers=num_workers)
    valid_dataset = PersonaChatDataset(config.dataset.valid, tokenizer.sep_token, max_context_turns=max_context_turns,
                                       add_persona_indicator=add_persona_indicator,
                                       add_role_indicator=add_role_indicator)
    valid_dataloader = get_dataloader(valid_dataset, tokenizer, config, num_workers=num_workers, batch_size_ratio=1)
    test_dataset = PersonaChatDataset(config.dataset.test, tokenizer.sep_token, max_context_turns=max_context_turns,
                                      add_persona_indicator=add_persona_indicator,
                                      add_role_indicator=add_role_indicator)
    test_dataloader = get_dataloader(test_dataset, tokenizer, config, num_workers=num_workers, batch_size_ratio=1)
    # initialize encoder decoder
    model = get_model_via_config(config, tokenizer)
    model.to(device)
    # default
    optimizer_params = dict(config.training.optimizer_param)
    optimizer = optim.SGD(model.parameters(), lr=config.training.lr, **optimizer_params)
    if config.training.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.training.lr, **optimizer_params)
    if init_path is not None:
        eprint("Loading init weight from {}".format(init_path))
        load_model(model, init_path, strict=False)
    model.eval()
    best_ppl = 65535.0
    scheduler = get_constant_schedule(optimizer)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_dataloader.__len__(),
    #                                             num_training_steps=train_dataloader.__len__() * config.training.epoch)
    for epoch in range(config.training.epoch):
        prefix_message = 'Epoch {}'.format(epoch)
        losses = train_one_epoch(model, optimizer, scheduler, train_dataloader, tokenizer, config, device,
                                 prefix_message=prefix_message)
        logger.add_train_loss(np.asarray(losses).mean(), epoch)
        # start testing one epoch
        model.eval()
        with torch.no_grad():
            result, ref_text, pred_text, persona_query = evaluate(model,
                                                                  valid_dataloader,
                                                                  tokenizer, device, config)
            print(result)
            logger.add_metrics('valid', result, epoch)
            os.makedirs('generated_text/{}'.format(config.experiment.name), exist_ok=True)
            file_path = 'generated_text/{}/{:03d}.txt'.format(config.experiment.name, epoch)
            write_to_file(file_path, ref_text, pred_text, persona_query)
        if best_ppl > result['token_ppl']:
            best_ppl = result['token_ppl']
            os.makedirs('save_models/{}'.format(config.experiment.name), exist_ok=True)
            save_model('save_models/{}/best.pt'.format(config.experiment.name),
                       model, optimizer, epoch, config, np.asarray(losses).mean())
        os.makedirs('save_models/{}'.format(config.experiment.name), exist_ok=True)
        save_model('save_models/{}/last.pt'.format(config.experiment.name),
                   model, optimizer, epoch, config, np.asarray(losses).mean())
        with torch.no_grad():
            result, ref_text, pred_text, persona_query = evaluate(model,
                                                                  test_dataloader,
                                                                  tokenizer, device, config)
            print(result)
            logger.add_metrics('test', result, epoch)

# if __name__ == '__main__':
#     config = get_config('config/rnn/lstm.yml')
#     device = 'cuda'
