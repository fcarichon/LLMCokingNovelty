import ast
from collections import Counter
import re
import ast
import json
from tqdm import tqdm
from math import log
import heapq
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.tokenize import word_tokenize
import nltk
from collections import OrderedDict
from Divergences import Jensen_Shannon
from utils import pmi_to_dict

class Surprise():

    def __init__(self, pmi_new):
        self.pmi_new = pmi_new
        self.JS = Jensen_Shannon()
        
    def get_common_vectors(self, dict_old, dict_new, epsilon):
        """ Input : nested dictionaries for each PMI collocations
        Ouput : list of tuples vectors for each words """
        variables_1 = dict_old['variables']
        variables_2 = dict_new['variables']
        inter_list = list(set(variables_1) & set(variables_2)) #list(set(variables_1 + variables_2))

        vectors = {}
        for entry in inter_list:
            vec_1 = [dict_old.get(entry, {}).get(key, epsilon) for key in inter_list]
            vec_2 = [dict_new.get(entry, {}).get(key, epsilon) for key in inter_list]
            vectors[entry] = (vec_1, vec_2)
    
        return vectors
    
    def new_surprise(self, pmi_known, thr_surp=0.0104):
        """ On compare la distribution avec en sans new_Q -- on compare l'apparition de nouvelles collocations -- PMI augmente drastiquement selon un threshold
        By setting threshold to 0, as long as there is a new tuple we will consider it """
        
        # Find tuples in list_1 but not in list_2 and exceed the threshold
        temp_known = [t[0] for t in pmi_known] ##  Useful to have only list of tuple not associated with their probbilities
        unique_tuples = [t for t in self.pmi_new if t[0] not in temp_known and t[1] > 0.]
        
        #count_unique = len(unique_tuples)
        surprise_rate = len(unique_tuples) / (len(self.pmi_new)+1)
        
        new_suprise = 0
        if surprise_rate > thr_surp:
            new_suprise = 1
        
        return surprise_rate, new_suprise

    def uniq_surprise(self, dict_known, eps= 0.000001, thr_surp=0.):
        """ On compare la distribution avec en sans new_Q -- on compare la divergence JSD moyenne de ces deux distributions"""
        dict_new = pmi_to_dict(self.pmi_new)
        vecotr_tuples = self.get_common_vectors(dict_known, dict_new, epsilon = eps)
        
        key_list = vecotr_tuples.keys()
        surprise_dists = []
        
        for entry in key_list:
            tuple_known = vecotr_tuples[entry][0]
            tuple_new = vecotr_tuples[entry][1]
            
            #### We want only positive PMI score -- no negative values for not going nan or inf values
            tuple_known = [max(0., val) for val in tuple_known]
            tuple_new = [max(0., val) for val in tuple_new]   
            if sum(tuple_known) != 0 and sum(tuple_new) != 0:   #### We can't compare to non existing vectors neither
                surprise_dists.append(self.JS.JSDiv(tuple_known, tuple_new))
    
        surprise_score = sum(surprise_dists) / (len(surprise_dists)+1)
        dist_surprise = 0
        if surprise_score > thr_surp:
            dist_surprise = 1
            
        return surprise_score, dist_surprise