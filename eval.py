import argparse
import os
import numpy as np

from src.handlers.evaluater import Evaluator


if __name__ == '__main__':
    ### Decoding arguments
    eval_parser = argparse.ArgumentParser(description='Arguments for training the system')
    eval_parser.add_argument('--path', type=str, help='path to experiment')
    eval_parser.add_argument('--dataset', type=str, help='dataset to train the system on')
    eval_parser.add_argument('--mode', default='test', type=str, help='which data split to evaluate on')
    eval_parser.add_argument('--device', default='cuda', type=str, help='selecting device to use')
    eval_parser.add_argument('--lim', type=int, default=None, help='whether subset of data to be used') 
    args = eval_parser.parse_args()

    
    evaluator = Evaluator(args.path, args.device)
    preds = evaluator.load_preds(args.dataset, args.mode)
    labels = evaluator.load_labels(args.dataset, args.mode)
    acc = evaluator.calc_acc(preds, labels)

        
    print(f'{acc:.2f}')
       

