import torch


def decoding_rnn_to_token_ids(encoder, decoder, input_tensor,
                              sos_token_id, eos_token_id, pad_token_id, target_length, device,
                              target_tensor=None, teacher_forcing=True):
    _, encoder_hidden = encoder(input_tensor)
    criterion = torch.nn.NLLLoss(ignore_index=pad_token_id, reduction='mean')
    # criterion = torch.nn.NLLLoss(reduction='sum')
    batch_size = input_tensor.shape[0]
    decoder_input = torch.tensor([[sos_token_id]] * batch_size, device=device)
    decoder_hidden = encoder_hidden
    if target_tensor is not None:
        target_length = target_tensor.size(1)
    loss_sum = 0
    losses = []
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            next_words = decoder_output.argmax(dim=-1)[:, -1].unsqueeze(-1)
            if teacher_forcing:
                decoder_input = torch.cat((decoder_input, target_tensor[:, di].view(-1, 1)), dim=-1)
            else:
                decoder_input = torch.cat((decoder_input, next_words), dim=-1)  # detach from history as input
            if target_tensor is not None:
                loss = criterion(decoder_output[:, -1, :].view(-1, decoder_output.shape[-1]), target_tensor[:, di])
                losses.append(loss)
                loss_sum += loss
    decoder_input = decoder_input.detach().cpu().numpy().tolist()
    output_ids = []
    for output in decoder_input:
        if eos_token_id in output:
            end_index = output.index(eos_token_id)
            output = output[1:end_index]
            output_ids.append(output)
        else:
            output_ids.append(output)
    loss_avg = torch.stack(losses).mean()
    return output_ids, loss_sum, loss_avg


def decoding_rnn_to_text(encoder, decoder, input_tensor,
                         sos_token_id, eos_token_id, pad_token_id, target_length, tokenizer, device,
                         target_tensor=None, teacher_forcing=True):
    output_ids, loss_sum, loss_mean = decoding_rnn_to_token_ids(encoder=encoder,
                                                                decoder=decoder,
                                                                input_tensor=input_tensor,
                                                                sos_token_id=sos_token_id,
                                                                eos_token_id=eos_token_id,
                                                                pad_token_id=pad_token_id,
                                                                target_length=target_length,
                                                                device=device,
                                                                target_tensor=target_tensor,
                                                                teacher_forcing=teacher_forcing)
    output_strs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return output_strs, loss_sum, loss_mean


