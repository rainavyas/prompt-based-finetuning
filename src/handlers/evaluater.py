import torch
import pickle
import numpy as np
import os
import torch.nn.functional as F

from tqdm import tqdm 
from types import SimpleNamespace

from .trainer import Trainer
from ..data.handler import DataHandler
from ..loss.cross_entropy import CrossEntropyLoss
from ..handlers.batcher import Batcher
from ..models.models import TransformerModel

class Evaluator(Trainer):
    """ Evaluator class- inherits Trainer so has all experiment methods
        class takes care of evaluation and automatic caching of results"""

    def __init__(self, path:str, device:str='cuda'):
        self.exp_path = path
        self.device = device

    def setup_helpers(self):
        # load arguments 
        args = self.load_args('model_args.json')

        # set up attributes 
        super().setup_helpers(args)

        # load model weights
        self.load_model()

    #== loading and saving probabilities ==========================================================#
    def load_preds(self, dataset:str, mode:str, bias:str=None)->dict:
        probs = self.load_probs(dataset, mode, bias)
        preds = {}
        for ex_id, probs in probs.items():
            preds[ex_id] = int(np.argmax(probs, axis=-1))  
        return preds

    def load_probs(self, dataset:str, mode:str, bias:str=None)->dict:
        """ loads cached probabilities, if not cached then generate """
        if not self.probs_exist(dataset, mode, bias):
            self.setup_helpers()
            probs = self.generate_probs(dataset, mode, bias)
            self.cache_probs(probs, dataset, mode, bias)
        probs = self.load_cached_probs(dataset, mode, bias)
        return probs

    def cache_probs(self, probs, dataset:str, mode:str, bias:str):
        prob_cache_path = self.get_prob_cache_path(dataset, mode, bias)
        with open(prob_cache_path, 'wb') as handle:
            pickle.dump(probs, handle)
    
    def load_cached_probs(self, dataset:str, mode:str, bias:str)->dict:
        prob_cache_path = self.get_prob_cache_path(dataset, mode, bias)
        with open(prob_cache_path, 'rb') as handle:
            probs = pickle.load(handle)
        return probs
    
    def probs_exist(self, dataset:str, mode:str, bias:str)->bool:
        prob_cache_path = self.get_prob_cache_path(dataset, mode, bias)
        return os.path.isfile(prob_cache_path)

    def get_prob_cache_path(self, dataset:str, mode:str, bias:str)->str:
        eval_name = f'{dataset}_{mode}_{bias}' if bias else f'{dataset}_{mode}'
        prob_cache_path = os.path.join(self.exp_path, 'eval', f'{eval_name}.pk')
        return prob_cache_path
     
    #== Model probability calculation method ======================================================#
    @torch.no_grad()
    def generate_probs(self, dataset:str, mode:str='test', bias=None, lim=None):
        """ get model probabilities for each example in dataset"""
        self.model.eval()
        self.to(self.device)

        eval_data = self.data_handler.prep_split(dataset, mode, bias, lim)
        eval_batches = self.batcher(
            data = eval_data, 
            bsz = 1, 
            shuffle = False
        )        
        probs = {}
        
        for batch in tqdm(eval_batches):
            ex_id = batch.ex_id[0]
            output = self.model_loss(batch)

            logits = output.logits.squeeze(0)
            if logits.shape and logits.shape[-1] > 1:  # Get probabilities of predictions
                prob = F.softmax(logits, dim=-1)
            probs[ex_id] = prob.cpu().numpy()
        return probs

    #== loading specific examples =================================================================#
    def load_ex(self, dataset:str, mode:str, k:int=0)->SimpleNamespace:
        data = self.data_handler.prep_split(dataset, mode, lim=10) #TEMP
        ex = data[k]
        return ex
    
    def tokenize_ex(self, ex:SimpleNamespace):
        if self.model_args.num_classes == 3: 
            ex = self.data_handler._prep_ids_pairs([ex])
        else: 
            ex = self.data_handler._prep_ids([ex])
        batch = next(self.batcher(ex, bsz=1))
        return batch
    
    #== general eval methods ======================================================================#
    @staticmethod
    def load_labels(dataset:str, mode:str='test', bias:str=None, lim=None)->dict:
        eval_data = DataHandler.load_split(dataset, mode, bias, lim)
        labels_dict = {}
        for ex in eval_data:
            labels_dict[ex.ex_id] = ex.label
        return labels_dict

    @staticmethod
    def load_split(dataset:str, mode:str='test', lim=None)->dict:
        eval_data = DataHandler.load_split(dataset, mode)
        output_dict = {}
        for ex in eval_data:
            output_dict[ex.ex_id] = ex
        return output_dict

    @staticmethod
    def calc_acc(preds, labels):
        assert preds.keys() == labels.keys(), "keys don't match"
        hits = sum([preds[idx] == labels[idx] for idx in labels.keys()])
        acc = hits/len(preds)
        return 100*acc

    #== loading and caching hidden representations ================================================#
    def load_layer_h(self, dataset:str=None, mode:str='test', layer:int=11, lim:int=None)->dict:
        h_path = self.get_h_path(dataset, mode, layer)

        if not os.path.isfile(h_path):
            self.setup_helpers()
            layer_repr = self.generate_h(dataset, mode, layer)
            self.cache_h(layer_repr, dataset, mode, layer)

        layer_repr = self.load_cached_h(dataset, mode, layer)
        return layer_repr 

    @torch.no_grad()
    def generate_h(self, dataset:str=None, mode:str='test', layer:int=None, lim:int=None):
        #prepare data for cases where data_name is given
        data = self.data_handler.prep_split(dataset, mode=mode, lim=lim)

        #create batches
        eval_batches = self.batcher(data=data, bsz=1, shuffle=False)
        
        self.to(self.device)
        self.model.eval()

        #get output vectors
        output_dict = {}
        for batch in tqdm(eval_batches):
            ex_id = batch.ex_id[0]
            output = self.model.transformer(
                input_ids=batch.input_ids, 
                attention_mask=batch.attention_mask,
                output_hidden_states=True
            )
            H = output.hidden_states[layer]
            output_dict[ex_id] = H[:, 0].squeeze(0).cpu().numpy()
            
        return output_dict

    def get_h_path(self, dataset:str, mode:str, layer:int)->str:
        CACHE_PATH = '/home/al826/rds/hpc-work/2022/shortcuts/spurious-nlp/notebooksl/inear_probe/cached_h'
        layer = layer % 12
        repr_path = os.path.join(CACHE_PATH, f'{dataset}_{mode}_{layer}.pk')
        return repr_path

    def cache_h(self, h, dataset, mode, layer):
        cache_h_path = self.get_h_path(dataset, mode, layer)
        with open(cache_h_path, 'wb') as handle:
            pickle.dump(h, handle)
    
    def load_cached_h(self, dataset:str, mode:str):
        pred_path = self.get_pred_path(dataset, mode)
        with open(pred_path, 'rb') as handle:
            h = pickle.load(handle)
        return h

