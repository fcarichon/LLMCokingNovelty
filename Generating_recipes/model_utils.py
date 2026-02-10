import json
import os
from os import walk
from tqdm import tqdm
#import statistics
import numpy as np

def getting_recipes(recipes):

    recipe_list = []
    for key in list(recipes.keys()):
        if key != "AllIngredients":                          #That is for the reference base case
            recipe_list.append(recipes[key]['recipe_clean'])
    
    #full_KB_recipe = ' '.join(recipe_list)
    full_KB_recipe = str(recipe_list)
    return full_KB_recipe

def get_template_for_recipes(file_names, other_file_names):
    """
    For each file name in other_file_names, find the corresponding template file
    in file_names by inserting '_template' before '.json'.
    
    Returns a dictionary where:
      - Key: file name from other_file_names.
      - Value: matching template file from file_names if found, else None.
    """
    mapping = {}
    for recipe in other_file_names:
        # Construct the expected template file name.
        expected_template = recipe.replace(".json", "_template.json")
        # Map the recipe to the expected template if it exists in file_names.
        mapping[recipe] = expected_template if expected_template in file_names else None
    return mapping
