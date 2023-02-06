import random
import re
import math

from collections import defaultdict
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
        
    # create final output
    output = []
    for i in groups.keys():
        output += groups[i][:num_per_class]

    random_seeded = random.Random(1)
    random_seeded.shuffle(output)
    return output

#== Length Biasing Methods ========================================================================#
def group_by_length(data:list, reverse=False, lim:int=None)->defaultdict:
    # for efficiency, only iterate through enough data
    load_lim = min(lim * 20, len(data)) if lim else len(data)

    # set random seed (for reproducibility) and shuffle data
    data = deepcopy(data)
    random_seeded = random.Random(1)
    random_seeded.shuffle(data)
    
    groups = defaultdict(list)
    for ex in tqdm(data[:load_lim], total=load_lim, disable=load_lim<2000):
        if not reverse:
            if   len(ex.text.split()) < 300:     groups[ex.label, 0].append(ex)
            elif len(ex.text.split()) > 300:     groups[ex.label, 1].append(ex)
        else:
            if   len(ex.text.split()) < 300:     groups[ex.label, 1].append(ex)
            elif len(ex.text.split()) > 300:     groups[ex.label, 0].append(ex)
    return groups

def create_balanced_length_biased_data(data:list, bias_acc:float=1, reverse=False, lim:int=None)->list:
    groups = group_by_length(data=data, reverse=reverse, lim=lim)
    num_classes = len(set([x[0] for x in groups.keys()]))
    print('NUM CLASSES: ', num_classes)

    # separate groups into biased and unbiased:
    new_groups = defaultdict(list)
    for i in range(num_classes):
        # biased class
        new_groups[i,1] = groups[i,i]

        # unbiased class
        for j in range(num_classes):
            if i == j: continue
        new_groups[i,0] += groups[i, (i+j) % num_classes]
    groups = new_groups

    # calculate number of biased rounds to have
    num_biased = min([len(groups[i,1]) for i in range(num_classes)])
    if lim: 
        num_biased = round((lim * bias_acc)/num_classes)
    
    # calculate number of unbiased rounds to have [ b/(u+b) = acc ]
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

#== Lexical Overlap Biasing Methods ===============================================================#
def lexical_overlap_score(text_1:str, text_2:str)->float:
    word_list_1 = remove_stopwords(text_1)
    word_list_2 = remove_stopwords(text_2)

    intersection = set(word_list_1) & set(word_list_2)
    union = set(word_list_1 + word_list_2)
    if len(union) == 0: 
        return None
    overlap_score = len(intersection)/len(union)
    return overlap_score

def filter_by_lexical_overlap(data:list, lower:float=0, upper:float=1)->list:
    output = []
    for ex in tqdm(data):
        overlap_score = lexical_overlap_score(ex.text_1, ex.text_2)
        if overlap_score is None: continue
        if lower <= overlap_score <= upper:
            output.append(ex)
    return output

def group_by_lexical_overlap(data:list, reverse=False, lim:int=None)->defaultdict:
    # for efficiency, only iterate through enough data
    load_lim = min(lim * 20, len(data)) if lim else len(data)

    # set random seed (for reproducibility) and shuffle data
    data = deepcopy(data)
    random_seeded = random.Random(1)
    random_seeded.shuffle(data)
    
    groups = defaultdict(list)
    for ex in tqdm(data[:load_lim], total=load_lim, disable=load_lim<2000):
        overlap_score = lexical_overlap_score(ex.text_1, ex.text_2)
        if overlap_score is None: continue

        if not reverse:
            if   overlap_score >= 0.5:        groups[ex.label, 0].append(ex)
            elif 0.05 < overlap_score < 0.5:  groups[ex.label, 1].append(ex)
            elif overlap_score <= 0.05:       groups[ex.label, 2].append(ex)
        else:
            if   overlap_score >= 0.5:        groups[ex.label, 2].append(ex)
            elif 0.05 < overlap_score < 0.5:  groups[ex.label, 1].append(ex)
            elif overlap_score <= 0.05:       groups[ex.label, 0].append(ex)
    return groups

def create_balanced_lexical_biased_data(data:list, bias_acc:float=1, reverse=False, lim:int=None)->list:
    groups = group_by_lexical_overlap(data=data, reverse=reverse, lim=lim)
    
    # separate groups into biased and unbiased:
    new_groups = defaultdict(list)
    for i in [0,1,2]:
        new_groups[i,1] = groups[i,i]
        new_groups[i,0] = groups[i, (i-1)//3] + groups[i, (i+1)//3]
    groups = new_groups

    # calculate number of biased rounds to have
    num_biased = min([len(groups[i,1]) for i in [0,1,2]])
    if lim: 
        num_biased = round((lim * bias_acc)/3)
    
    # calculate number of unbiased rounds to have [ b/(u+b) = acc ]
    num_unbiased = num_biased * ((1 - bias_acc) / bias_acc)
    num_unbiased = int(num_unbiased)

    # correction if number of samples is impossible
    min_num_unbiased_per_class = min([len(groups[i,0]) for i in [0,1,2]])
    if num_unbiased > min_num_unbiased_per_class:
        num_unbiased = min_num_unbiased_per_class
        num_biased = (bias_acc / (1 - bias_acc)) * num_unbiased
        num_biased = round(num_biased)

    assert(min(num_biased, num_unbiased) >= 0)
    print(f'# SAMPLES: {3*(num_unbiased + num_biased)}    BIAS_ACC: {num_biased/(num_unbiased + num_biased):.3f}')

    # create final output
    output = []
    for i in [0,1,2]:
        output += groups[i, 0][:num_unbiased]
        output += groups[i, 1][:num_biased]

    random_seeded = random.Random(1)
    random_seeded.shuffle(output)
    return output

def create_balanced_inverse_lexical_bias(data:list, bias_acc:float=1, reverse=False, lim:int=None)->list:
    groups = group_by_lexical_overlap(data=data, reverse=reverse, lim=lim)
    
    # only look for (0, 2) (entailment with no lexical overlap) and (2,0) (contradiction with high lexical overlap)
    num_unbiased = min(len(groups[0,2]), len(groups[2,0]))
    output = groups[0,2][:num_unbiased] + groups[2,0][:num_unbiased]

    random_seeded = random.Random(1)
    random_seeded.shuffle(output)
    return output
