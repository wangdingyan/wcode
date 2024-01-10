import argparse as ap
import collections as col
from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch_scatter import scatter_mean
import torch_geometric as tg

from e3nn.o3 import Linear

class EDN_Model(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

