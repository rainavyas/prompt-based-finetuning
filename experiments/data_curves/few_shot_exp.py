import os
import argparse
import logging

from collections import defaultdict
from src.handlers.trainer import Trainer
from src.handlers.evaluater import Evaluator
from src.data.handler import DataHandler
from src.utils.general import save_json, load_json

# Load logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Model arguments
model_parser = argparse.ArgumentParser(description='Arguments for system and model configuration')
model_parser.add_argument('--path', type=str, help='path to experiment', required=True)
model_parser.add_argument('--transformer', type=str, default='bert-base', help='[bert, roberta, electra ...]')
model_parser.add_argument('--prompt-finetuning', action='store_true', help='whether to use prompt finetuning')
model_parser.add_argument('--label-words', type=str, nargs = '+', default=['yes', 'maybe', 'no'], help='for prompt finetuning, which words to use as the labels')

model_parser.add_argument('--maxlen', type=int, default=512, help='max length of transformer inputs')
model_parser.add_argument('--num-classes', type=int, default=2, help='number of classes (3 for NLI)')
model_parser.add_argument('--rand-seed', type=int, default=None, help='random seed for reproducibility')

### Training arguments
train_parser = argparse.ArgumentParser(description='Arguments for training the system')
train_parser.add_argument('--dataset', default='snli', type=str, help='dataset to train the system on')
train_parser.add_argument('--ood', default=None, type=str, nargs = '+', help='OOD dataset to evaluate models on')
train_parser.add_argument('--bias', type=str, default='balanced',  help='whether data should be synthetically biased (e.g. lexical)')
train_parser.add_argument('--lim', type=int, default=None,  help='size of data subset to use for debugging')
train_parser.add_argument('--epochs', type=int, default=12, help='number of epochs to train system for')
train_parser.add_argument('--bsz', type=int, default=8, help='training batch size')
train_parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
train_parser.add_argument('--data-ordering', action='store_true', help='dynamically batches to minimize padding')

train_parser.add_argument('--grad-clip', type=float, default=1, help='gradient clipping')
train_parser.add_argument('--freeze-trans', type=str, default=None, help='number of epochs to freeze transformer')

train_parser.add_argument('--log-every', type=int, default=1000, help='logging training metrics every number of examples')
train_parser.add_argument('--val-every', type=int, default=100_000, help='when validation should be done within epoch')
train_parser.add_argument('--wandb', action='store_true', help='if set, will log to wandb')
train_parser.add_argument('--device', type=str, default='cuda', help='selecting device to use')

# python baseline.py --path exp/fewshot/snli/bert-base-finetune --dataset snli --ood mnli --num-classes 3
# python baseline.py --path exp/fewshot/snli/bert-base-prompt-finetune --dataset snli --ood mnli --num-classes 3 --prompt-finetuning --label-words yes maybe no --device cuda:1 
# python baseline.py --path exp/fewshot/sst/bert-base-finetune --dataset sst --ood rt imdb --num-classes 2
# python baseline.py --path exp/fewshot/sst/bert-base-prompt-finetune --dataset sst --ood rt imdb --num-classes 2 --prompt-finetuning --label-words terrible great --device cuda:3


if __name__ == '__main__':
    # Parse system input arguments
    model_args, moargs = model_parser.parse_known_args()
    train_args, toargs = train_parser.parse_known_args()

    # Making sure no unkown arguments are given
    assert set(moargs).isdisjoint(toargs), f"{set(moargs) & set(toargs)}"

    logger.info(model_args.__dict__)
    logger.info(train_args.__dict__)
    
    performances = defaultdict(dict)

    # Load any runs that have already been saved in past runs
    if os.path.isfile(os.path.join(model_args.path, 'curve.json')):
        performance_cache = load_json(os.path.join(model_args.path, 'curve.json'))
        for lim in performance_cache.keys():
            performances[int(lim)] = performance_cache[lim]

    for lim in [100, 1_000, 4_000, 10_000, 40_000, 100_000, 200_000, 400_000]: #10, 20, 40, 200, 400, 4_000, 20_000
        # check whether enough data samples, and exit if so
        train_data = DataHandler.load_split(train_args.dataset, mode='train', bias='balanced', lim=lim)
        print(len(train_data) < lim-2, len(train_data), lim-2)

        if len(train_data) < lim-2:
            continue

        for seed_num in [1, 2, 3, 4, 5]:
            # skip runs already done in previous submissions
            if str(seed_num) in performances[lim]:
                continue

            #== Training ==========================================================================#
            # reset random seed as gets overwritten by other runs
            setattr(model_args, 'rand_seed', None)
            
            # artifically set the data truncation limit
            setattr(train_args, 'lim', lim)
            
            # create model
            exp_path = os.path.join(model_args.path, f'{lim}/seed-{seed_num}')
            trainer = Trainer(exp_path, model_args)
            
            # reduce number of epochs for training on larger splits
            # if lim >= 5000:  setattr(train_args, 'epochs', 5)

            # train the model
            trainer.train(train_args)
            
            #== Evaluation ========================================================================#
            seed_perf = {}

            evaluator = Evaluator(exp_path, train_args.device)
            preds  = evaluator.load_preds(train_args.dataset, 'test')
            labels = evaluator.load_labels(train_args.dataset, 'test')
            acc = evaluator.calc_acc(preds, labels)
            seed_perf[train_args.dataset] = acc

            # run evaluation on all OOD domains
            ood_acc = None
            ood_domains = getattr(train_args, 'ood')
            if ood_domains:
                for domain in ood_domains:
                    ood_preds  = evaluator.load_preds(domain, 'test')
                    ood_labels = evaluator.load_labels(domain, 'test')
                    
                    if domain == 'hans':
                        ood_preds = {idx: (pred == 0) for idx, pred in ood_preds.items()}
                        
                    ood_acc = evaluator.calc_acc(ood_preds, ood_labels)
                    seed_perf[domain] = ood_acc
            
            # append results to overall performance curve
            performances[lim][seed_num] = seed_perf
            
            # delete model weights to not use up all disk space so quickly
            os.remove(
                os.path.join(exp_path, 'models/model.pt')
            )
            
            # save the updated performance
            save_json(
                performances, os.path.join(model_args.path, 'curve.json')
            )


######### python curve.py --bias lexical-0.7 --path curves --ood mnli
