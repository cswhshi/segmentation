# -*- coding: utf-8 -*-

import os
import argparse
parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")

parser.add_argument('--dataset', type=str, default='humanparsing',
                        choices=['humanparsing','pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
 
 
parser.add_argument('--checkname', type=str, default="model.t7",
                        help='set the checkpoint name')

args = parser.parse_args()



directory = os.path.join('run', args.dataset, args.checkname)