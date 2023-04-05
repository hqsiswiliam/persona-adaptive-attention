
import logging

logger = logging.getLogger(__name__)


def load_weight(model, state_dict):
    model.transformer.load_state_dict(state_dict, False)
    return model
