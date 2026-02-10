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
import statistics
from scipy.spatial import distance

def get_metadata(recette_var, KB_length, index_list):
    
    nb_ingr, nb_new_ingr, ratio_new_ingr, clean_text_length, lexical_div, clean_nb_uniq_tokens, ratio_length = [], [], [], [], [], [], []
    for index in index_list:
        nb_ingr.append(recette_var[index]['nb_ingredients'])
        nb_new_ingr.append(recette_var[index]['nb_new_ingredients'])
        ratio_new_ingr.append(recette_var[index]['nb_new_ingredients'] / (recette_var[index]['nb_ingredients']+1))
        clean_text_length.append(recette_var[index]['clean_text_length'])
        lexical_div.append(recette_var[index]['clean_nb_uniq_tokens']/recette_var[index]['clean_text_length'])
        clean_nb_uniq_tokens.append(recette_var[index]['clean_nb_uniq_tokens'])
        ratio_length.append(recette_var[index]['clean_text_length']/KB_length)

    return uniq_, nb_ingr, nb_new_ingr, ratio_new_ingr, clean_text_length, clean_nb_uniq_tokens, lexical_div, ratio_length

def get_novel_scores(recette_var, index_list):
    
    scores = {'newness_div': [], 'newness_prob': [], 'new_extremes': [], 'new_rank': [], 'uniq_dist': [], 'uniq_proto': [],
        'diff_global': [], 'diff_local': [], 'new_surprise': [], 'dist_surprise': []}
    for index in index_list:
        item = recette_var[index]
        for key in scores:
            if key in item:
                scores[key].append(item[key])

    return tuple(scores[key] for key in scores)

def get_KB_size(recette_dict):
    
    indexes = [item for item in recette_dict.keys() if item != 'AllIngredients']
    length = []
    for index in indexes:
        recette = recette_dict[index]['recipe_clean']
        recette_list = recette.split(' ')
        length.append(len(recette_list))
    avg_len = sum(length)/len(length)
    
    return avg_len

def inglehart_dist(df): 

    distances = pd.DataFrame(index=df['Country'], columns=df['Country'])
    # Calculate Euclidean distances
    for i in range(len(df)):
        for j in range(len(df)):
            distances.iloc[i, j] = distance.euclidean((df.iloc[i]['TradAgg'], df.iloc[i]['SurvSAgg']), (df.iloc[j]['TradAgg'], df.iloc[j]['SurvSAgg']))

    # Convert distances to numeric
    distances = distances.apply(pd.to_numeric)
    return distances

def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of Earth in kilometers. Use 3956 for miles.
    r = 6371.0
    
    # Calculate the result
    return c * r

def geograph_dist(df):
    ## Calculate the Haversine distances
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                lat1, lon1 = df.iloc[i][['latitude', 'longitude']]
                lat2, lon2 = df.iloc[j][['latitude', 'longitude']]
                distances.iloc[i, j] = haversine(lat1, lon1, lat2, lon2)
            else:
                distances.iloc[i, j] = 0

    return distances

def get_distance(recette_var, index_list, KB_df, type='ling'):
    
    """ FOR LINGUISTIC AND RELIGIOUS DISTANCE 
        type='ling' or 'reli', 'ingle', 'geo'
    """
    distance = []
    country_list_j = list(set(KB_df['country_j']))
    for index in index_list:
        country_index = recette_var[index]['country']
        if country_index == 'congo':
            country_index = 'united states'

        if type == 'ling' or type == 'reli':
            if country_index in country_list_j: 
                if type == 'ling':
                    dist = KB_df[KB_df['country_j'] == country_index]['Lang Dist '].iloc[0]
                    distance.append(dist)
                else:
                    dist = KB_df[KB_df['country_j'] == country_index]['religious_distance'].iloc[0]
                    distance.append(dist)
        if type == 'ingle' or type == 'geo':
            if country_index in list(df_dist.index): 
                distance.append(df_dist[country_index])
            
    return distance