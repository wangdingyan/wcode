import torch
import warnings

class ExponentialLR_with_minLr(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr=1e-4, last_epoch=-1, verbose=False):
        self.gamma = gamma
        self.min_lr = min_lr
        super(ExponentialLR_with_minLr, self).__init__(optimizer, gamma, last_epoch, verbose)
    def get_lr(self) -> float:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch==0:
            return self.base_lrs
        return [max(group['lr'] * self.gamma, self.min_lr)
                for group in self.optimizer.param_groups]
    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
                for base_lr in self.base_lrs]


def get_optimizer(config, model):
    if config.type == "Adam":
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr = config.lr,
            weight_decay = config.weight_decay)
    else:
        raise NotImplementedError("Optimizer not supported: %s" % config.type)


def get_scheduler(config, optimizer):
    if config.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.factor,
            patience=config.patience
        )
    elif config.train.scheduler == 'expmin':
        return ExponentialLR_with_minLr(
            optimizer,
            gamma = config.factor,
            min_lr=config.min_lr,)
    else:
        raise NotImplementedError('Scheduler not supported: %s' % config.type)

class EMA():
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.decay) * x + self.decay * self.shadow[name]
        self.shadow[name] = new_average.clone()