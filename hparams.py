import argparse

class Hparams:
    parser = argparser.ArgumentParser()

    parser.add_argument('dataset', type=str)
    parser.add_argument('modality', type=str, choice=['RGB', 'Flow'])

    parser.add_argument('--save_dir', default="test", help="save directory")
