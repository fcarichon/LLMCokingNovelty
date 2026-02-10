import os
from os import walk

current_path = os.getcwd()
home_path = base_path = os.path.join(*current_path.split(os.sep)[:5])


path = {
    'cache_dir': f"/{home_path}/",
    'save_dir': f'./Gen_recipes/',                                               ## To save your generated recipes
    'template_path': f'./Templates/',                                ## To access you template repository
    'recipe_dir': f'./GlobalFusion/'                                 ## PAth to GlobalFusion
}

hp_balanced = {
    'temp_value': 0,           # Controls randomness (lower = more deterministic, higher = more diverse)
    'top_p_value':0.92,        # Nucleus sampling (keeps only top 92% of probability mass for fluency)
    'top_k_value':50,          # Limits sampling to top 50 most likely words (reduces weird token choices)
     'repet_value':1.4,        # Penalizes repetition to avoid looping outputs
     'gen_len':800,
     'sample_bin':True         # Enables sampling for natural sentence variation
     }

hp_strict = {
    'temp_value': 0,           # Controls randomness (lower = more deterministic, higher = more diverse)
    'top_p_value':0.92,        # Nucleus sampling (keeps only top 92% of probability mass for fluency)
    'top_k_value':20,          # Limits sampling to top 50 most likely words (reduces weird token choices)
     'repet_value':1.2,        # Penalizes repetition to avoid looping outputs
     'gen_len':800,
     'sample_bin':True         # Enables sampling for natural sentence variation
     }

hp_divergent = {
    'temp_value': 0,           # Controls randomness (lower = more deterministic, higher = more diverse)
    'top_p_value':0.92,        # Nucleus sampling (keeps only top 92% of probability mass for fluency)
    'top_k_value':100,         # Limits sampling to top 50 most likely words (reduces weird token choices)
     'repet_value':1.8,        # Penalizes repetition to avoid looping outputs
     'gen_len':800,
     'sample_bin':True         # Enables sampling for natural sentence variation
     }