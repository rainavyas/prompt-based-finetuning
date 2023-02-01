import argparse
import logging

from src.handlers.trainer import Trainer

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
model_parser.add_argument('--label-words', type=str, nargs = '+', default=None, help='for prompt finetuning, which words to use as the labels')

model_parser.add_argument('--maxlen', type=int, default=512, help='max length of transformer inputs')
model_parser.add_argument('--num-classes', type=int, default=2, help='number of classes (3 for NLI)')
model_parser.add_argument('--rand-seed', type=int, default=None, help='random seed for reproducibility')

### Training arguments
train_parser = argparse.ArgumentParser(description='Arguments for training the system')
train_parser.add_argument('--dataset', type=str, default='snli', help='dataset to train the system on')
train_parser.add_argument('--bias', type=str, default=None, help='whether data should be synthetically biased (e.g. lexical)')
train_parser.add_argument('--lim', type=int, default=None, help='size of data subset to use for debugging')

train_parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train system for')
train_parser.add_argument('--bsz', type=int, default=16, help='training batch size')
train_parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
train_parser.add_argument('--data-ordering', action='store_true', help='dynamically batches to minimize padding')

train_parser.add_argument('--grad-clip', type=float, default=1, help='gradient clipping')
train_parser.add_argument('--freeze-trans', type=int, default=None, help='number of epochs to freeze transformer')

train_parser.add_argument('--log-every', type=int, default=400, help='logging training metrics every number of examples')
train_parser.add_argument('--val-every', type=int, default=50_000, help='when validation should be done within epoch')
train_parser.add_argument('--wandb', action='store_true', help='if set, will log to wandb')
train_parser.add_argument('--device', type=str, default='cuda', help='selecting device to use')


if __name__ == '__main__':
    # Parse system input arguments
    model_args, moargs = model_parser.parse_known_args()
    train_args, toargs = train_parser.parse_known_args()

    # Making sure no unkown arguments are given
    assert set(moargs).isdisjoint(toargs), f"{set(moargs) & set(toargs)}"

    logger.info(model_args.__dict__)
    logger.info(train_args.__dict__)
    
    trainer = Trainer(model_args.path, model_args)
    trainer.train(train_args)