import pandas as pd
import ast
from collections import Counter
import re
import json
from tqdm import tqdm
from collections import defaultdict 
from collections import Counter
from os import walk

from scipy import stats
import math

def ranked_lists1(recette_var, index_list, top_perc=10, bottom_perc=0.):
    #https://towardsdatascience.com/rbo-v-s-kendall-tau-to-compare-ranked-lists-of-items-8776c5182899
    new_dict, uniq_dict, diff_dict, nsurp_dict, dsurp_dict = {},{},{},{},{}
    for index in index_list:
        new_dict[index] = recette_var[index]['newness']
        uniq_dict[index] = recette_var[index]['uniqueness']
        diff_dict[index] = recette_var[index]['difference']
        nsurp_dict[index] = recette_var[index]['new_surprise']
        dsurp_dict[index] = recette_var[index]['dist_surprise']

    sorted_new = [k for k, v in sorted(new_dict.items(), key=lambda item: item[1], reverse=True)]
    sorted_uniq = [k for k, v in sorted(uniq_dict.items(), key=lambda item: item[1], reverse=True)]
    sorted_diff = [k for k, v in sorted(diff_dict.items(), key=lambda item: item[1], reverse=True)]
    sorted_nsurp = [k for k, v in sorted(nsurp_dict.items(), key=lambda item: item[1], reverse=True)]
    sorted_dsurp = [k for k, v in sorted(dsurp_dict.items(), key=lambda item: item[1], reverse=True)]

    if top_perc != 0.:
        if len(index_list) > top_perc:
            sorted_new = sorted_new[:int(len(sorted_new) * (top_perc/100))]
            sorted_uniq = sorted_uniq[:int(len(sorted_uniq) * (top_perc/100))]
            sorted_diff = sorted_diff[:int(len(sorted_diff) * (top_perc/100))]
            sorted_nsurp = sorted_nsurp[:int(len(sorted_nsurp) * (top_perc/100))]
            sorted_dsurp = sorted_dsurp[:int(len(sorted_dsurp) * (top_perc/100))]
        
    if bottom_perc != 0.:
        if len(index_list) > bottom_perc:
            sorted_new = sorted_new[-int(len(sorted_new) * (bottom_perc/100)):]
            sorted_uniq = sorted_uniq[-int(len(sorted_uniq) * (bottom_perc/100)):]
            sorted_diff = sorted_diff[-int(len(sorted_diff) * (bottom_perc/100)):]
            sorted_nsurp = sorted_nsurp[-int(len(sorted_nsurp) * (bottom_perc/100)):]
            sorted_dsurp = sorted_dsurp[-int(len(sorted_dsurp) * (bottom_perc/100)):]

    return sorted_new, sorted_uniq, sorted_diff, sorted_nsurp, sorted_dsurp

def rbo(list1, list2, p=0.8):
   
    # tail recursive helper function
   def helper(ret, i, d):
       l1 = set(list1[:i]) if i < len(list1) else set(list1)
       l2 = set(list2[:i]) if i < len(list2) else set(list2)
       a_d = len(l1.intersection(l2))/i
       term = math.pow(p, i) * a_d
       if d == i:
           return ret + term
       return helper(ret + term, i + 1, d)
   k = max(len(list1), len(list2))
   x_k = len(set(list1).intersection(set(list2)))
   summation = helper(0, 1, k)
    
   return ((float(x_k)/k) * math.pow(p, k)) + ((1-p)/p * summation)

def kendall_tau(list1, list2):

    #https://stackoverflow.com/questions/60941997/how-to-compute-the-distance-between-2-ranked-lists
    tau, p_value = stats.kendalltau(list1, list2)
    return tau, p_value

def nb_common(list1, list2):
    return len(set(list1) & set(list2))