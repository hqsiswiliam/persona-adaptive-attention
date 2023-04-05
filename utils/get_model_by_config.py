from model.causal_decoder.conv_gpt import ConvGPT
from model.paa_decoder.paa_decoder.transformer import TransformerPAAGPT
from model.paa_decoder.paa_decoder.transformer.paa_encoder import PAAEncoder
from model.pretrained_encoder_decoder.conv_enc_dec import ConvEncDec
from model.transformer.transformer_seq2seq import TransformerSeq2Seq
from utils.eprint import eprint


def get_model_via_config(config, tokenizer):
    dict_config = config.toDict()

    if 'pretrained_enc_dec' in dict_config.keys() and 'model_type' in dict_config['pretrained_enc_dec'].keys():
        model_type = dict_config['pretrained_enc_dec']['model_type']
        model = ConvEncDec(model_type, tokenizer)
    elif 'paa_transformer' in dict_config.keys() and 'encoder' in dict_config['paa_transformer'].keys():
        shared = dict_config['paa_transformer']['encoder']['shared']
        persona_encoder = PAAEncoder(config, tokenizer)
        if shared:
            eprint('Shared encoder enabled!')
            model = TransformerPAAGPT(tokenizer, config, persona_encoder, persona_encoder,
                                        'gpt2', tau=config.paa_transformer.decoder.tau)
        else:
            context_encoder = PAAEncoder(config, tokenizer)
            model = TransformerPAAGPT(tokenizer, config, persona_encoder, context_encoder,
                                        'gpt2', tau=config.paa_transformer.decoder.tau)
    elif 'transformer' in dict_config.keys():
        model = TransformerSeq2Seq(config, tokenizer)
    else:
        model = ConvGPT(config, tokenizer)
    return model
