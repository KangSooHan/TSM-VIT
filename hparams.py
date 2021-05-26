import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str)
    parser.add_argument('modality', type=str, choices=['RGB', 'Flow'])
    parser.add_argument('--model_type', type=str, default='ViT-B_16', choices=['ViT-B_16', 'ViT-B_32', 'ViT-L_16',
                                                                                'ViT-L_32', 'ViT-H_14'])
    parser.add_argument('--local_rank', type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--img_size', type=int, default=224, help="Image Size")

    parser.add_argument('--save_dir', default="test", help="save directory")
    parser.add_argument('--pretrained_dir', default="test", help="save directory")
