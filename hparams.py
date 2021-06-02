import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str)
    parser.add_argument('modality', type=str, choices=['RGB', 'Flow'])
    parser.add_argument('--model_type', type=str, default='ViT-B_16', choices=['ViT-B_16', 'ViT-B_32', 'ViT-L_16',
                                                                                'ViT-L_32', 'ViT-H_14'])

    parser.add_argument('--optimizer', type=str, default='SGD', choices=["SGD", "Adam", "AdamW"])
    parser.add_argument('--gpus', type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--img_size', type=int, default=224, help="Image Size")


    parser.add_argument('--batch_size', type=int, default=16, help="Batch Size")
    
    parser.add_argument('-j','--workers', type=int, default=32, help="number of data loading workers")


    parser.add_argument('--lr', type=float, default=0.001, help="Learning Rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum")
    parser.add_argument('--weight-decay', '-wd', type=float, default=5e-4, help='weight decay')

    parser.add_argument('--num_segments', type=int, default=6, help='number of segment')
    parser.add_argument('--num_frames', type=int, default=6, help='number of segment')

    parser.add_argument('--save_dir', default="test", help="save directory")
    parser.add_argument('--pretrained_dir', default="pretrained", help="save directory")

    parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')

    parser.add_argument('-e', '--evaluate', action="store_true", help='evaluate model on validation set')


    parser.add_argument('--root_log',type=str, default='log')
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
    parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')

    parser.add_argument('--print-freq', '-p', default=40, type=int,
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')

    parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
