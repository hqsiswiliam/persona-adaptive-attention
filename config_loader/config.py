import yaml
from dotmap import DotMap


def extend_dict(extend_me, extend_by):
    if isinstance(extend_me, dict):
        for k, v in extend_by.iteritems():
            if k in extend_me:
                extend_dict(extend_me[k], v)
            else:
                extend_me[k] = v
    else:
        if isinstance(extend_me, list):
            extend_list(extend_me, extend_by)
        else:
            extend_me += extend_by


def extend_list(extend_me, extend_by):
    missing = []
    for item1 in extend_me:
        if not isinstance(item1, dict):
            continue

        for item2 in extend_by:
            if not isinstance(item2, dict) or item2 in missing:
                continue
            extend_dict(item1, item2)


def extend_compatibility_for_paa_transformer(configuration):
    dict_config = configuration.toDict()
    if 'paa_transformer' in dict_config.keys() and 'encoder' in dict_config['paa_transformer'].keys():
        if 'shared' not in dict_config['paa_transformer']['encoder']:
            configuration.paa_transformer.encoder.shared = False
        if 'shared_crossattention' not in dict_config['paa_transformer']['decoder']:
            configuration.paa_transformer.decoder.shared_crossattention = False
        if 'auto_tau' not in dict_config['paa_transformer']['decoder']:
            configuration.paa_transformer.decoder.auto_tau = False
        if 'auto_tau_numerator' not in dict_config['paa_transformer']['decoder']:
            configuration.paa_transformer.decoder.auto_tau_numerator = 'persona'
        if 'fusion_mode' not in dict_config['paa_transformer']['decoder']:
            configuration.paa_transformer.decoder.fusion_mode = '((pr)(cr))'
        if 'base_model' not in dict_config['paa_transformer']['decoder']:
            configuration.paa_transformer.decoder.base_model = 'gpt2'
        if 'role_embedding' not in dict_config['paa_transformer']['encoder']:
            configuration.paa_transformer.encoder.role_embedding = False
        if 'turn_embedding' not in dict_config['paa_transformer']['encoder']:
            configuration.paa_transformer.encoder.turn_embedding = False
        if 'dataset' not in dict_config['dataset']:
            configuration.dataset.dataset = 'convai2'
        if 'gate_fc' not in dict_config['paa_transformer']['decoder']:
            configuration.paa_transformer.decoder.gate_fc = True
        if 'add_persona_to_decoder' not in dict_config['paa_transformer']['decoder']:
            configuration.paa_transformer.decoder.add_persona_to_decoder = False
        if 'add_context_to_decoder' not in dict_config['paa_transformer']['decoder']:
            configuration.paa_transformer.decoder.add_context_to_decoder = False
        if 'save_model' not in dict_config['training']:
            configuration.training.save_model = True
        if 'extend_candidates' not in dict_config['training']:
            configuration.training.extend_candidates = False
        if 'double_heads' not in dict_config['paa_transformer']['decoder']:
            configuration.paa_transformer.decoder.double_heads = False
    return configuration


def get_config(path):
    with open(path, 'r') as file:
        configuration = yaml.load(file, Loader=yaml.FullLoader)
    with open('config/default.yml', 'r') as file:
        base_configuration = yaml.load(file, Loader=yaml.FullLoader)
    configuration = DotMap(configuration)
    base_configuration = DotMap(base_configuration)
    extend_dict(configuration, base_configuration)
    configuration = extend_compatibility_for_paa_transformer(configuration)
    return configuration


if __name__ == '__main__':
    config = get_config('config/bert-base.yml')
