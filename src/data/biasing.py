import random
import re
import math

from collections import defaultdict
from types import SimpleNamespace
from typing import List, Dict, Tuple, TypedDict
from nltk.corpus import stopwords
from tqdm import tqdm 
from copy import deepcopy

STOPWORDS = list(stopwords.words())

#== util functions ================================================================================#
def remove_stopwords(word_list:str)->list:
    word_list = re.split('[\s,.!?]', word_list.lower())
    word_list = [i for i in word_list if i.strip() != '']
    word_list = [i for i in word_list if i not in STOPWORDS]
    return word_list

#== balancing util functions ======================================================================#
def create_balanced_data(data, lim:int=None)->list:
    # for efficiency, only iterate through enough data
    load_lim = min(lim * 20, len(data)) if lim else len(data)

    # set random seed (for reproducibility) and shuffle data
    data = deepcopy(data)
    random_seeded = random.Random(1)
    random_seeded.shuffle(data)
    
    groups = defaultdict(list)
    for ex in tqdm(data[:load_lim], total=load_lim, disable=load_lim<2000):
        groups[ex.label].append(ex)
    
    # ensure enough examples in each class
    num_class = len(groups.keys())
    max_num_per_class = min([len(i) for i in groups.values()])
    num_per_class = round(lim/num_class) if lim else max_num_per_class
        
    print(num_per_class)
    # create final output
    output = []
    for i in groups.keys():
        output += groups[i][:num_per_class]

    random_seeded = random.Random(1)
    random_seeded.shuffle(output)
    return output

#== Length Biasing Methods ========================================================================#
def length_score(ex:SimpleNamespace):
    if hasattr(ex, 'text'):
        len_score = len(ex.text)
    else:
        len_score = len(ex.text_1) + len(ex.text_2)
    return len_score

def lexical_overlap_score(ex:SimpleNamespace)->float:
    word_list_1 = remove_stopwords(ex.text_1)
    word_list_2 = remove_stopwords(ex.text_2)

    intersection = set(word_list_1) & set(word_list_2)
    union = set(word_list_1 + word_list_2)
    if len(union) == 0: 
        return None
    overlap_score = len(intersection)/len(union)

    #return 1 - overlap score so that 0=entailment 2=contradiction and
    return 1 - overlap_score

def get_bias_fn(bias_name):
    """ select bias function to use """
    if bias_name == 'length':
        bias_fn = length_score
    elif bias_name == 'lexical':
        bias_fn = lexical_overlap_score
    else:
        raise ValueError("invalid biasing option, {bias_name}")
    return bias_fn

def group_by_bias(data:list, bias_name:str, bias_bounds:list, lim:int=None)->defaultdict:
    # internal function to convert the bias score to group
    def score_to_group(score:float):
        for k, v in enumerate(bias_bounds):            
            if score <= v: 
                return k
        return len(bias_bounds)

    # select bias function to use
    bias_fn = get_bias_fn(bias_name)

    # for efficiency, only iterate through enough data
    load_lim = min(lim * 20, len(data)) if lim else len(data)

    # set random seed (for reproducibility) and shuffle data
    data = deepcopy(data)
    random_seeded = random.Random(1)
    random_seeded.shuffle(data)
    
    groups = defaultdict(list)
    for ex in tqdm(data[:load_lim], total=load_lim, disable=load_lim<2000):
        bias_score = bias_fn(ex)
        # skip invalid bias scores
        if bias_score is None:
            continue
        
        # add example to the relevant group
        bias_group = score_to_group(bias_score)
        groups[ex.label, bias_group].append(ex)
    return groups

def sort_groups_into_biased(groups:defaultdict):
    num_classes = len(set([x[0] for x in groups.keys()]))

    sorted_groups = defaultdict(list)
    for i in range(num_classes):
        # biased class
        sorted_groups[i,1] = groups[i,i]

        # unbiased class
        for j in range(num_classes):
            if i != j: 
                sorted_groups[i,0] += groups[i, (i+j) % num_classes]
    return sorted_groups

def create_biased_balanced_data(data:list, bias_name:str, bias_bounds:list, bias_acc:float=1, lim:int=None)->list:
    groups = group_by_bias(data=data, bias_name=bias_name, bias_bounds=bias_bounds, lim=lim)
    num_classes = len(set([x[0] for x in groups.keys()]))

    # separate groups into biased and unbiased:
    groups = sort_groups_into_biased(groups)

    # calculate number of biased rounds to have
    num_biased = min([len(groups[i,1]) for i in range(num_classes)])
    if lim: 
        num_biased_temp = round((lim * bias_acc)/num_classes)
        num_biased = min(num_biased, num_biased_temp)

    # calculate number of unbiased rounds to have [ acc = b/(u+b) ]
    num_unbiased = num_biased * ((1 - bias_acc) / bias_acc)
    num_unbiased = int(num_unbiased)

    # correction if number of samples is impossible
    min_num_unbiased_per_class = min([len(groups[i,0]) for i in range(num_classes)])
    if num_unbiased > min_num_unbiased_per_class:
        num_unbiased = min_num_unbiased_per_class
        num_biased = (bias_acc / (1 - bias_acc)) * num_unbiased
        num_biased = round(num_biased)

    assert(min(num_biased, num_unbiased) >= 0)
    print(f'# SAMPLES: {num_classes*(num_unbiased + num_biased)}    BIAS_ACC: {num_biased/(num_unbiased + num_biased):.3f}')

    # create final output
    output = []
    for i in range(num_classes):
        output += groups[i, 0][:num_unbiased]
        output += groups[i, 1][:num_biased]
    random_seeded = random.Random(1)
    random_seeded.shuffle(output)
    return output

def inverse_biased_data(data:list, bias_name:str, bias_bounds:list, lim:int=None)->list:
    groups = group_by_bias(data=data, bias_name=bias_name, bias_bounds=bias_bounds, lim=lim)
    num_classes = len(set([x[0] for x in groups.keys()]))

    # separate groups into biased and unbiased:
    groups = sort_groups_into_biased(groups)

    # create final output
    output = []
    for i in range(num_classes):
        output += groups[i, 0]
    random_seeded = random.Random(1)
    random_seeded.shuffle(output)
    return output