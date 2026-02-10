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
import numpy as np

class Newness():
    """
    Estimate the ratio of new terms in the distribution and the ratio of disappearing terms
    known_P is the known distribution (Knowledge or Expectation Base) and new_Q should be the novel distribrution (Determine if the document is new or not)
    The two option are mathematicallyequivalent if you set equivalent threshold -- to choose based on your ease of interpretation
    """
    def __init__(self, known_P, new_Q, lambda_=0.9):

        self.known_P = known_P
        self.new_Q = new_Q
        self.lambda_ = lambda_
        
        JS = Jensen_Shannon()
        self.JS = JS
        self.JSD_vector = JS.linear_JSD(known_P, new_Q)
        self.nb_elements = len(self.JSD_vector)

    def divergent_terms(self, thr_div=0.0129, thr_new=0.0014):

        """
        JSD == 0 if and only if pi = qi, but we want to make sure the distribution gap between this two are large enough
        To interpret as if the new term make the divergence greater than threshold, then it is a significant cointributing, we just need to know in appearing or disappearing
        """
        count_appear = 0
        count_disappear = 0

        for i in range(self.nb_elements):
            if self.JSD_vector[i] > thr_div:
                if self.new_Q[i] > self.known_P[i]:
                    count_appear += 1
                if self.known_P[i] > self.new_Q[i]:
                    count_disappear += 1

        appear_ratio = count_appear / self.nb_elements
        disappear_ratio = count_disappear / self.nb_elements
        newness = self.lambda_ * appear_ratio + (1-self.lambda_) * disappear_ratio
        novelty = 0
        if newness > thr_new:
            novelty = 1
        
        return newness, novelty

    def probable_terms(self, thr_prob=2, cte = 1e-10, thr_new=0.5):
        """
        To interpret as if the new term has a probability of appearing thr_new times greater than old doc, it is a new significant term. Reverse is true for disappearing
        """
        count_appear = 0
        count_disappear = 0
        for i in range(self.nb_elements):
            if self.JSD_vector[i] != 0:
                # To interpret as if the new term has a probability of appearing thr_new times greater than old doc, it is a new significant term. Reverse is true for disappearing
                if self.new_Q[i] / (self.known_P[i]+cte) > thr_prob:  
                    count_appear += 1
                if self.known_P[i] / (self.new_Q[i]+cte) > thr_prob:
                    count_disappear += 1
        
        appear_ratio = count_appear / self.nb_elements
        disappear_ratio = count_disappear / self.nb_elements
        newness = self.lambda_ * appear_ratio + (1 - self.lambda_) * disappear_ratio
        novelty = 0
        if newness > thr_new:
            novelty = 1
        
        return newness, novelty

    def new_extremes(self, thr_div=0.1, thr_new=0.5, pareto=0.8):
    
        arr_known = np.array(self.known_P)
        arr_new = np.array(self.new_Q)

        #Filtering 0 values from Q because you want values that appear in Q or not in its tail but not non existing variables
        filtered_known = np.array([x for x, y in zip(arr_known, arr_new) if y != 0.])
        filtered_new = np.array([y for y in arr_new if y != 0.])
        
        sorted_indices = np.argsort(filtered_known)
        
        num_values_to_keep = int(pareto * len(filtered_known))
        appear_indices = sorted_indices[:num_values_to_keep]
        
        inv_num_values = len(filtered_known) - num_values_to_keep
        disappear_indices = sorted_indices[inv_num_values:]
        
        appear_know = filtered_known[appear_indices]
        appear_new = filtered_new[appear_indices]
        
        JSD_appear = self.JS.linear_JSD(appear_know, appear_new)
        count_appear = 0
        if len(appear_new) > 0:
            for i in range(len(JSD_appear)):
                if JSD_appear[i] > thr_div:
                    if self.new_Q[i] > self.known_P[i]:
                        count_appear += 1
            appear_ratio = count_appear / len(JSD_appear)
        else:
            appear_ratio = 0.
                
        disappear_know = filtered_known[disappear_indices]
        disappear_new = filtered_new[disappear_indices]
        
        count_disappear = 0
        if len(disappear_new) > 0:
            JSD_disappear = self.JS.linear_JSD(disappear_know, disappear_new)
            for i in range(len(JSD_disappear)):
                if self.JSD_vector[i] > thr_div:
                    if self.known_P[i] > self.new_Q[i]:
                        count_disappear += 1
            
            disappear_ratio = count_disappear / len(JSD_disappear)
        else:
            disappear_ratio = 0.

        newness = self.lambda_ * appear_ratio + (1 - self.lambda_) * disappear_ratio
        
        novelty = 0
        if newness > thr_new:
            novelty = 1
                
        return newness, novelty

    def new_ranking(self, thr_new=0.5, rank_thr=3):

        # Convert lists to numpy arrays
        arr_known = np.array(self.known_P)
        arr_new = np.array(self.new_Q)

        #Filtering 0 values from Q because you want modified ranked but not non existing variables
        filtered_known = np.array([x for x, y in zip(arr_known, arr_new) if y != 0.])
        filtered_new = np.array([y for y in arr_new if y != 0.])
        
        # Get the rank positions in descending order (rank 1 for the highest value)
        ranks_known = np.argsort(-filtered_known) + 1
        ranks_new = np.argsort(-filtered_new) + 1
    
        # Calculate the differences in ranks
        rank_diffs = np.abs(ranks_known - ranks_new)
        
        new_score = 0
        for elem in rank_diffs:
            if elem >= rank_thr:
                new_score += 1
            
        newness = new_score / len(rank_diffs) ## With that we consider appearing and disappearing as the same phenomenon
        novelty = 0
        if newness > thr_new:
            novelty = 1
            
        return newness, novelty

class Uniqueness():
    """
        We estimate the distance between an new distribution and the overall generall distribution
    """
    def __init__(self, known_P):
        self.known_P = known_P
        #self.new_Q = new_Q
        self.JS = Jensen_Shannon()
        
    def dist_to_proto(self, new_Q, thr_uniq=0.05):
        
        novel_uniq = 0
        uniqueness_ = self.JS.JSDiv(self.known_P, new_Q)
        if uniqueness_ >= thr_uniq:
            novel_uniq = 1
            
        return uniqueness_, novel_uniq

    def proto_dist_shift(self, new_P, thr_uniqp=0.05):
        
        #new_P = self.known_P + self.new_Q
        uniqueness = self.JS.JSDiv(self.known_P, new_P)
        novel_uniq = 0
        if uniqueness >= thr_uniqp:
            novel_uniq = 1

        return uniqueness, novel_uniq

class Difference():
    """
        We estimate the ratio of point that are in close vicinity of the point. 
        list_know_P : represent the list of all distribution vectors for each individual documents
    """
    def __init__(self, list_know_P, new_Q, N=5):

        self.list_know_P = list_know_P
        self.new_Q = new_Q
        self.JS = Jensen_Shannon()
        self.N = N
        #self.neighbor_dist = self.dist_estimate()
        
    def dist_estimate(self):
        """
        Here we take the N closest neighbours of each points and we estimate the average distance to each points to its closests neighbors.
        Then we return the average for the whole dataset to know what is the average distance a point is close to its neighbors

            Stop at a sample of points -- prevent the code here to run forever???
        """
        avg_dists = []
        for i in range(len(self.list_know_P)):
            P_i = self.list_know_P[i]
            #list_execpt = self.list_know_P[:i] + self.list_know_P[i+1:] ## We compare the dist to all elements except himself
            list_execpt = np.delete(self.list_know_P, i, axis=0)
            
            all_dists = []
            for P_j in list_execpt:
                all_dists.append(self.JS.JSDiv(P_i, P_j))
            
            if len(all_dists) > self.N:
                all_dists = heapq.nsmallest(self.N, all_dists)
            
            avg_dist_i = sum(all_dists) / len(all_dists)
            avg_dists.append(avg_dist_i)
            
        avg_final = sum(avg_dists) / len(avg_dists)
    
        return avg_final
    
    def ratio_to_all(self, neighbor_dist, thr_diff=0.95):
        count_diff = 0
        for P_i in self.list_know_P:
            distance = self.JS.JSDiv(P_i, self.new_Q)
            if distance >= neighbor_dist:
                count_diff += 1
        
        #Proportion of points where the distance is superior to the average distance to normal neighbor -- the higher the more different
        difference = count_diff / len(self.list_know_P)
        novel_diff = 0
        if difference > thr_diff:
            novel_diff = 1

        return difference, novel_diff

    def ratio_to_neighbors(self, neighbor_dist, thr_diff=0.85):
        """
        Computes the ratio of nearest neighbors where the distance to new_Q exceeds neighbor_dist.
        """
        count_diff = 0
        num_known_P = self.list_know_P.shape[0]
        all_dists = []
        list_know_P = self.list_know_P#.tocsr()  # Ensure CSR format for efficiency
        new_Q = self.new_Q

        # Compute distances to all points
        for i in range(num_known_P):
            P_i = list_know_P[i].flatten()
            #P_i = list_know_P[i].toarray().flatten()
            all_dists.append(self.JS.JSDiv_Edgar(P_i, self.new_Q))

        # Identify the closest N neighbors
        closest_dists = heapq.nsmallest(self.N, all_dists)

        # Count neighbors with distances exceeding the threshold
        count_diff = sum(1 for dist in closest_dists if dist > neighbor_dist)

        # Compute the proportion of neighbors exceeding the threshold
        difference = count_diff / len(closest_dists)
        novel_diff = int(difference > thr_diff)

        return difference, novel_diff

        
        