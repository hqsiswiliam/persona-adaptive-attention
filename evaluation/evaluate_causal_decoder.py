import numpy as np
from tqdm import tqdm
import torch
from torch import nn


def evaluate(model,
             test_dataloader,
             tokenizer, device, config):
    result = {}
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
    pbar = tqdm(test_dataloader, total=test_dataloader.__len__())
    persona_query = []
    ref_text = []
    pred_text = []
    losses = []
    valid_losses = []
    valid_tokens = []
    for input_data in pbar:
        model.eval()
        with torch.no_grad():
            logits, _ = model(persona_query_input=input_data['persona_query_input'].to(device),
                              target_input=input_data['target_input'].to(device),
                              input_data=input_data,
                              is_train=True,
                              return_loss=True)
            logits = logits[:, :-1, :]  # we ignore [EOS] -> [PAD] here
            loss = criterion(logits.contiguous().view(-1, logits.shape[-1]),
                             input_data['target_input']['input_ids'].reshape(-1).to(device))
            # ignore those padding tokens
            valid_loss = loss[loss > 0]
            valid_losses.append(valid_loss)
            valid_tokens.append(valid_loss.shape[0])
            losses.append(loss.mean().item())
            mloss = np.mean(losses)
            pbar.set_postfix_str("mloss: {:.2f}".format(mloss))
    from functools import reduce
    flatten_valid_losses = reduce(lambda acc, nxt: nxt.detach().cpu().numpy().tolist() + acc, valid_losses, [])
    result['token_ppl'] = np.exp(np.array(flatten_valid_losses).mean())
    result['loss'] = np.mean(losses)
    return result, ref_text, pred_text, persona_query
