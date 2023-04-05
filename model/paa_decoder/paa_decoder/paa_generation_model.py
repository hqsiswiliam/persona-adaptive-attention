from .containers import Module


class PAAGenerationModel(Module):
    def __init__(self):
        super(PAAGenerationModel, self).__init__()

    def init_weights(self):
        raise NotImplementedError

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        raise NotImplementedError
