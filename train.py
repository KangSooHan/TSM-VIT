#-*- coding: utf-8 -*-#
import os
import time
import argparse
import logging
import shutil
import numpy as np
from tqdm import tqdm

import torch

from hparams import Hparams
#from utils import *
from data_load import return_dataset
#from model import *

from models.model import VisionTransformer, CONFIGS


def main():
    global hp
    logging.basicConfig(level = logging.INFO)

    logging.info("# Hparams")
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if hp.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hp.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(hp.local_rank)
        device = torch.device("cuda", hp.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        hp.n_gpu = 1
    hp.device = device




    logging.info("# Prepare Dataset")
    num_class, hp.train_list, hp.val_list, hp.root_path, prefix = return_dataset(hp.dataset, hp.modality)

    save_dir = hp.save_dir

    config = CONFIGS[hp.model_type]

    model = VisionTransformer(config, hp.img_size, zero_head=True, num_classes=101)
    model.load_from(np.load(hp.pretrained_dir))
    model.to(hp.device)
    num_params = count_parameters(model)


if __name__ == '__main__':
    main()
