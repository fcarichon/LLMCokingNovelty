
######### CONFIG FOR LLM Evaluation here so everything is score_true since I just want to specifically execute for LLMs only
path = {
    "root": '.',
    "gf_path": f"{ROOT_DIR}/GlobalFusion_withLLMs/",
    "save_path": f"{ROOT_DIR}/LLM_Scored/",
    "model_gen_path":"Generated_answers_llama3-instruct_run_0config_1/",  ## Replace by your config/run name used Gen_recipes.py
    'cache_dir': f".",
    "score_train": 'store_false',
    "score_valid": 'store_false',
    "score_test": 'store_false',        ##### IF you run python script.py --score_test then score_test == False, if you run only python_scirpt.py then score_test == True
    "score_LLMS": 'store_false',
    "score_embeddings": 'store_false'
}

other_hp = {
    "model_name": 'llama3-instruct',
    "add_middle_layers": 'store_false',   ### If you want to autmoatically add the middle layers in the list of analyzed_layers. If you want to set it manually the change to store_true
    "analyzed_layers": [0,1,-2,-1]            ## With two middle layers is enough
}

# Default thrsholds for going from Novelty scores to binary scores
# Estimated as average in Train datasets -- See file thrshold estimate for details
thrs = {"newness_div": 0.00092,
        "newness_prob":14.64,
        "new_extremes":0.0014,
        "novelty_new":0.0014,
        "newness_rank":10,
        "uniq_dist":0.527,
        "uniq_proto":0.00358,
        "diff_global":0.897,
        "diff_local":0.614,
        "neighbors":3,
        "new_surprise":0.0104,
        "dist_surprise":0.00256
       }

# Default option to estimate novelty and surprise
types = {"newness_div": True,
         "newness_prob": False,
         "new_extremes": False,
         "new_rank":False,
         "uniq_dist":True,
         "uniq_proto":False,
         "diff_global":False,
         "diff_local":True,
         "new_surprise":True,
         "dist_surprise":True}