'''
nohup python train.py --config_path ./bash_sde/drugs_ema.yml > sdegen_drugs.log 2>&1 &
'''
import argparse
import numpy as np
import random
import os
import pickle
import yaml
from easydict import EasyDict
import torch
from sdegen import model, data, runner, utils
from sdegen.data import CycPepDataset

from sdegen.utils import logger

if __name__ == '__main__':

    with open('/mnt/c/Official/WCODE/proj/CycGen/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    config.train.save_path = os.path.join(config.train.save_path, config.scheme.framework)
    logger.configure(dir=os.path.join(config.train.save_path, config.model.name))

    # if args.seed != 2022:
    #     config.train.seed = args.seed

    if config.train.save and config.train.save_path is not None:
        config.train.save_path = os.path.join(config.train.save_path, config.model.name)
        if not os.path.exists(config.train.save_path):
            os.makedirs(config.train.save_path)

    # check device
    gpus = list(filter(lambda x: x is not None, config.train.gpus))
    assert torch.cuda.device_count() >= len(gpus), 'do you set the gpus in config correctly?'
    device = torch.device(gpus[0]) if len(gpus) > 0 else torch.device('cpu')

    logger.log("Let's use", len(gpus), "GPUs!")
    logger.log("Using device %s as main device" % device)
    config.train.gpus = gpus

    logger.log(config)

    # set random seed
    np.random.seed(config.train.seed)
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.train.seed)
        torch.cuda.manual_seed_all(config.train.seed)
    torch.backends.cudnn.benchmark = True
    logger.log('set seed for random, numpy and torch')

    load_path = os.path.join(config.data.base_path)
    logger.log('loading data from %s' % load_path)

    if config.scheme.framework == 'sde':
        model = model.SDE(config)
    else:
        raise KeyError

    train_data = []
    val_data = []
    test_data = []

    if config.data.train_set is not None:
        with open(os.path.join(load_path, config.data.train_set), "rb") as fin:
            train_data = pickle.load(fin)

    if config.data.val_set is not None:
        with open(os.path.join(load_path, config.data.val_set), "rb") as fin:
            val_data = pickle.load(fin)

    logger.log(
        'train size : %d  ||  val size: %d  ||  test size: %d ' % (len(train_data), len(val_data), len(test_data)))
    logger.log('loading data done!')

    transform = None
    train_data = data.CycPepDataset(data=train_data, transform=transform)
    val_data = CycPepDataset(data=val_data, transform=transform)
    test_data = []

    optimizer = utils.get_optimizer(config.train.optimizer, model)
    scheduler = utils.get_scheduler(config.train.scheduler, optimizer)

    solver = runner.DefaultRunner(train_data, val_data, test_data, model, optimizer, scheduler, gpus, config)
    if config.train.resume_train:
        solver.load(config.train.resume_checkpoint, epoch=config.train.resume_epoch, load_optimizer=True,
                    load_scheduler=True)
    solver.train()