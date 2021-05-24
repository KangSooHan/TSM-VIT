#-*- coding: utf-8 -*-#
import os
import time
import argparse
import logging
import shutil
from tqdm import tqdm

import torch

#from hparams import Hparams
#from utils import *
#from data_load import get_batch
#from model import *


def main(){
    logging.basicConfig(level = logging.INFO)

    logging.info("# Hparams")
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()

    model = VIT()

    crop_size = model.crop_size
    scale_size = model.scale_size
    




if __name__ == '__main__':
    main()
