"""
((pr)(cr)) is our proposed method
"""
import argparse
from config_loader.config import get_config
from utils.eprint import eprint
from utils.parser_helper import str2bool
import os

from utils.setup_seed import setup_seed
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--batch', type=int, default=None)
parser.add_argument('--load_percent', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--save_model', type=str2bool, default=None)
parser.add_argument('--init_path', type=str, default=None)

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--epoch', type=int, default=None)
parser.add_argument('--experiment_name', type=str, default=None)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--gradient_clip', type=float, default=None)
parser.add_argument('--enc_bidirectional', type=str2bool, default=None)
parser.add_argument('--enc_dropout', type=float, default=None)
parser.add_argument('--dec_dropout', type=float, default=None)
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--num_workers', type=int, default=None)
# PAA only
parser.add_argument('--tau', type=float, default=None)
parser.add_argument('--gated', type=str2bool, default=None)
parser.add_argument('--response_gated', type=str2bool, default=None)
parser.add_argument('--shared_enc', type=str2bool, default=None)
parser.add_argument('--pretrained_encoder', type=str, default=None)
parser.add_argument('--shared_crossattention', type=str2bool, default=None)
parser.add_argument('--auto_tau', type=str2bool, default=False)
parser.add_argument('--gate_fc', type=str2bool, default=None)
parser.add_argument('--auto_tau_numerator', type=str, default='persona')
parser.add_argument('--fusion_mode', type=str, default='((pr)(cr))')
parser.add_argument('--reinforce_persona', type=str, default=None)
parser.add_argument('--add_persona_to_decoder', type=str2bool, default=None)
parser.add_argument('--add_context_to_decoder', type=str2bool, default=None)

# Dataset Options
parser.add_argument('--add_persona_indicator', type=str2bool, default=None)
parser.add_argument('--add_role_indicator', type=str2bool, default=None)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--max_context_turns', type=int, default=None)

# Attention Routing
parser.add_argument('--persona_alpha', type=float, default=None)


args = parser.parse_args()
config_path = args.config
batch_size = args.batch
lr = args.lr
load_percent = args.load_percent
experiment_name = args.experiment_name
enc_bidirectional = args.enc_bidirectional
device = args.device
enc_dropout = args.enc_dropout
dec_dropout = args.dec_dropout
weight_decay = args.weight_decay
btn_count = args.btn_count
btn_fuse_layer = args.btn_fuse_layer
num_workers = args.num_workers
epoch = args.epoch
# Gated Network
tau = args.tau
gated = args.gated
response_gated = args.response_gated
shared_enc = args.shared_enc
shared_crossattention = args.shared_crossattention
auto_tau = args.auto_tau
auto_tau_numerator = args.auto_tau_numerator
fusion_mode = args.fusion_mode
pretrained_enc = args.pretrained_encoder
# Dataset Options
add_persona_indicator = args.add_persona_indicator
add_role_indicator = args.add_role_indicator
dataset = args.dataset
config = get_config(config_path)
if dict(config.experiment) == {}:
    config.experiment.name = config_path.split(os.sep)[-1][:-4]
if dataset == 'convai2':
    config.dataset.train = "data/ConvAI2/train_self_original.txt"
    config.dataset.test = "data/ConvAI2/valid_self_original.txt"
    config.dataset.valid = "data/ConvAI2/valid_self_original.txt"
elif dataset == 'personachat':
    config.dataset.train = "data/personachat/train_self_original.txt"
    config.dataset.test = "data/personachat/test_self_original.txt"
    config.dataset.valid = "data/personachat/valid_self_original.txt"
else:
    raise ValueError("Must choose between personachat and convai2")
config.experiment.name = "{}-DB={}".format(config.experiment.name, dataset)
if batch_size is not None:
    config.training.batch_size = args.batch
if lr is not None:
    config.training.lr = lr
if epoch is not None:
    config.training.epoch = epoch
if experiment_name is not None:
    config.experiment.name = experiment_name
config.experiment.name = "{}-LR={}".format(config.experiment.name, config.training.lr)
if args.gradient_clip is not None:
    config.training.gradient_clip = args.gradient_clip
    config.experiment.name = "{}-GCLIP={}".format(config.experiment.name, args.gradient_clip)
if enc_bidirectional is not None:
    config.rnn.bidirectional = enc_bidirectional
    config.experiment.name = "{}-BI={}".format(config.experiment.name, enc_bidirectional)
if enc_dropout is not None:
    config.paa_transformer.encoder.attention_probs_dropout_prob = enc_dropout
    config.paa_transformer.encoder.hidden_dropout_prob = enc_dropout
    config.experiment.name = "{}-ENC_DP={}".format(config.experiment.name, enc_dropout)
if dec_dropout is not None:
    config.paa_transformer.decoder.attention_probs_dropout_prob = dec_dropout
    config.paa_transformer.decoder.hidden_dropout_prob = dec_dropout
    config.experiment.name = "{}-DEC_DP={}".format(config.experiment.name, dec_dropout)
if weight_decay is not None:
    config.training.optimizer_param.weight_decay = weight_decay
    config.experiment.name = "{}-W_DECAY={}".format(config.experiment.name, weight_decay)

if btn_count is not None:
    config.causal_decoder.btn_count = btn_count
    config.experiment.name = "{}-BTN_TOKEN={}".format(config.experiment.name, btn_count)
if btn_fuse_layer is not None:
    config.causal_decoder.btn_fuse_layer = btn_fuse_layer
    config.experiment.name = "{}-BTN_LAYER={}".format(config.experiment.name, btn_fuse_layer)
# For the Gated Transformer
if gated is None:
    eprint("Warning, gated is set to None. If you run the baseline, please ignore this!")
if gated is not None and config.paa_transformer is not None:
    if '-' in fusion_mode:
        if fusion_mode == 'p-cr':
            fusion_mode = "(p(cr))"
        elif fusion_mode == 'pr-cr':
            fusion_mode = "((pr)(cr))"
        elif fusion_mode == 'cr-pr':
            fusion_mode = "((cr)(pr))"
        elif fusion_mode == 'fc--prr-crr':
            fusion_mode = "fc((prr)(crr))"
        elif fusion_mode == 'prr-crr':
            fusion_mode = "((prr)(crr))"
        elif fusion_mode == 'fc--pr-cr':
            fusion_mode = "fc((pr)(cr))"
        elif fusion_mode == 'fc-cpr':
            fusion_mode = "fc(cpr)"
        elif fusion_mode == 'skipc-pr':
            fusion_mode = '(skip(c))(pr)'
        else:
            raise ValueError("Invalid Fusion Mode!")
    config.paa_transformer.decoder.gated = gated
    config.experiment.name = "{}-Gated={}".format(config.experiment.name, gated)
    config.paa_transformer.decoder.fusion_mode = fusion_mode
    config.experiment.name = "{}-FM={}".format(config.experiment.name, fusion_mode)
    if fusion_mode not in ['((pr)(cr))', '(p(cr))', '((pc)(r))', '(skip(c))(pr)',
                           '((prr)(crr))', 'random', 'pr', 'cr', 'fc((pr)(cr))',
                           'fc((prr)(crr))', "((cr)(pr))",'fc(cpr)','param_gate','condition_bias','attention_routing']:
        raise ValueError("Invalid Fusion Mode!")
if auto_tau:
    config.paa_transformer.decoder.auto_tau = auto_tau
    if auto_tau == 'accurate':
        config.experiment.name = "{}-TAU=ACC_AUTO-TAU_NU={}".format(config.experiment.name, auto_tau_numerator)
    else:
        config.experiment.name = "{}-TAU=AUTO-TAU_NU={}".format(config.experiment.name, auto_tau_numerator)
elif tau is not None and config.paa_transformer is not None:
    config.paa_transformer.decoder.tau = tau
    config.experiment.name = "{}-TAU={}".format(config.experiment.name, tau)
if args.gate_fc is not None:
    gate_fc = args.gate_fc
    assert gated, 'please enable gated before enable gate_fc!'
    config.paa_transformer.decoder.gate_fc = gate_fc
    config.experiment.name = "{}-GATE_FC={}".format(config.experiment.name, gate_fc)
if response_gated is not None and config.paa_transformer is not None:
    config.paa_transformer.decoder.response_gated = response_gated
    config.experiment.name = "{}-RES_GATED={}".format(config.experiment.name, response_gated)
if shared_enc is not None and config.paa_transformer is not None:
    config.paa_transformer.encoder.shared = shared_enc
    config.experiment.name = "{}-SHARED_ENC={}".format(config.experiment.name, shared_enc)
if shared_crossattention is not None and config.paa_transformer is not None:
    config.paa_transformer.decoder.shared_crossattention = shared_crossattention
    config.experiment.name = "{}-SHARED_CROSS_ATTN={}".format(config.experiment.name, shared_crossattention)
if pretrained_enc is not None:
    config.paa_transformer.encoder.pretrained_enc = pretrained_enc
    config.experiment.name = "{}-P_ENC={}".format(config.experiment.name, pretrained_enc)
if args.reinforce_persona is not None and args.reinforce_persona:
    reinforce_persona = args.reinforce_persona
    config.paa_transformer.decoder.reinforce_persona = reinforce_persona
    config.experiment.name = "{}-R_PERSONA={}".format(config.experiment.name, reinforce_persona)
if args.add_persona_to_decoder is not None and args.add_persona_to_decoder:
    add_persona_to_decoder = args.add_persona_to_decoder
    config.paa_transformer.decoder.add_persona_to_decoder = add_persona_to_decoder
    config.experiment.name = "{}-ADD_PER2DEC={}".format(config.experiment.name, add_persona_to_decoder)
if args.add_context_to_decoder is not None and args.add_context_to_decoder:
    add_context_to_decoder = args.add_context_to_decoder
    config.paa_transformer.decoder.add_context_to_decoder = add_context_to_decoder
    config.experiment.name = "{}-ADD_CON2DEC={}".format(config.experiment.name, add_context_to_decoder)
# For the Dataset
if add_persona_indicator is not None:
    config.dataset.add_persona_indicator = add_persona_indicator
    config.experiment.name = "{}-P_IND={}".format(config.experiment.name, add_persona_indicator)
if add_role_indicator is not None:
    config.dataset.add_role_indicator = add_role_indicator
    config.experiment.name = "{}-R_IND={}".format(config.experiment.name, add_role_indicator)
config.training.num_workers = num_workers
if load_percent < 1.0:
    eprint("Warning, the dataset load percent is less than 1.0, which is only allowed for debugging purpose!")
    config.experiment.name = "{}-LOAD_PERCENT={}".format(config.experiment.name, load_percent)
if args.max_context_turns is not None:
    config.dataset.max_context_turns = args.max_context_turns
    config.experiment.name = "{}-HIS_TURNS={}".format(config.experiment.name, args.max_context_turns)

if args.persona_alpha is not None:
    assert config.paa_transformer.decoder.fusion_mode == 'attention_routing', \
        'can only set persona alpha in attention routing!'
    config.paa_transformer.decoder.persona_alpha = args.persona_alpha
    config.experiment.name = "{}-PALPHA={}".format(config.experiment.name, args.persona_alpha)
if args.seed is not None:
    seed = args.seed
    config.training.seed = seed
    config.experiment.name = "{}-SEED={}".format(config.experiment.name, seed)
if args.save_model is not None:
    config.training.save_model = args.save_model
    if not config.training.save_model:
        eprint("Warning, the model will not be saved!!!!")
# setup seed
setup_seed(config.training.seed)
print("EXP NAME: {}".format(config.experiment.name))
eprint("EXP NAME: {}".format(config.experiment.name))
if dict(config.causal_decoder) != {} or dict(config.transformer) != {} or dict(config.paa_transformer) != {}\
        or (config.pretrained_enc_dec) != {}:
    from trainer.causal_decoder_trainer import train

    train(config, device, load_percent=load_percent, init_path=args.init_path)
else:
    raise ValueError("Not implemented")
