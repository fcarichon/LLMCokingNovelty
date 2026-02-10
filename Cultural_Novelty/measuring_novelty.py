import os
from os import walk
import json
import re
from tqdm import tqdm
import sys
import score_config as config
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

from model_registry import MODEL_REGISTRY

sys.path.append("./Novelty_Scoring/")
from utils import data_analysis, pmi, pmi_to_dict, docs_distribution, new_distribution, get_info, get_new_ingr, text_cleaning
from utils_embeddings import logit_lens, add_falcon_compat_aliases
from Scoring import compute_scores

import argparse

#####################################################################################################################
######### At the time of coding you needed this configuraiotn for Microsoft Phi-4  -- Uncomment only for this model
#from transformers.cache_utils import DynamicCache
#def _get_usable_length(self, seq_length, layer_idx=None):
    # Match the old behaviour: just return the full cached length
#    return self.get_seq_length()

# Monkeypatch DynamicCache to be compatible with older remote code
#if not hasattr(DynamicCache, "get_usable_length"):
#    DynamicCache.get_usable_length = _get_usable_length
#####################################################################################################################

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "True", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False","f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating Novelty and Surprise scoring for all variation recipes")
    parser.add_argument("--root", default=config.path["root"], help="ROOT Directory where model is located")
    parser.add_argument("--gf_path", default=config.path["gf_path"], help="File path to folder of json files for each recipes")
    parser.add_argument("--model_gen_path", default=config.path["model_gen_path"], help="File path to folder of json files for each recipes")
    parser.add_argument("--save_path", default=config.path["save_path"], help="File path to folder of json files for each recipes")
    parser.add_argument("--cache_dir", default=config.path["cache_dir"], help="File path to folder of json files for each recipes")
    parser.add_argument("--score_train", action=config.path["score_train"], help="If you want to estimante novelty score for train variations")
    parser.add_argument("--score_valid", action=config.path["score_valid"], help="If you want to estimante novelty score for valid variations")
    parser.add_argument("--score_test", action=config.path["score_test"], help="If you want to estimante novelty score for test variations")
    parser.add_argument("--score_llms", action=config.path["score_LLMS"], help="If you want to estimante novelty score for LLMs variations")
    parser.add_argument("--score_embeddings", action=config.path["score_embeddings"], help="If you want to estimante novelty score for Embedding spaces for each variations")
    parser.add_argument("--model_name", default=config.other_hp["model_name"], help="ROOT Directory where model is located")
    parser.add_argument("--add_middle_layers", action=config.other_hp["add_middle_layers"], help="If you want to estimante novelty score for test variations")
    parser.add_argument('--layers', nargs='+', type=int, default=config.other_hp["analyzed_layers"],help="List of layers to analyze")
    parser.add_argument('--gen_type', default='same', help='if same -- new generation mode with same country from LLM and GlobalFusion, if dist, gen LLMs based on sampling countries on cult_dist')
    parser.add_argument('--nb_gpus', default=2, type=int, help='tensor_parallel_size in vLLMs for large models only')
    args = parser.parse_args() 

    ### Making sure that embeddings analyzer correspond to the model that generated the recipes
    assert args.model_name in args.model_gen_path, f"'{args.model_name}' not found in '{args.model_gen_path}' - make sure to analyze results with the same model as for generation"

    ############# Loading Files + Making sure that we processed only non processed ones
    data_path = args.gf_path + args.model_gen_path
    filenames = list(set(next(walk(data_path), (None,None,[]))[2]))

    save_path_model = args.save_path + args.model_gen_path
    if not os.path.exists(save_path_model):
        os.makedirs(save_path_model)
        already_saved_files = []
    else:
        already_saved_files = list(set(next(walk(save_path_model), (None,None,[]))[2]))
    
    #Filtering already scored recipes -- to continue moving forward if server shut down code
    if len(already_saved_files) > 0:
        already_saved_prefixes = set(f.split('_full')[0] for f in already_saved_files if '_full' in f)
        filtered_filenames = [f for f in filenames if f.split('_full')[0] not in already_saved_prefixes]
    else:
        filtered_filenames = filenames
    
    if args.score_embeddings:
        big_model_list = ['llama3-big', 'gemma-2-big', 'gemma-3','falcon-big', 'apertus', 'bloom','microsoft', 'qwen_2', 'qwen_3']

        try:
            model_name = MODEL_REGISTRY[args.model]
        except KeyError:
            raise ValueError(f"Unknown model: {args.model}")

        max_memory = {f"cuda:{i}": "20GiB" for i in range(args.nb_gpus)}  # adjust for your GPUs

        ### REGARDER COMMENT FIXER CERTAINs HP ICI - NOTAMMENT TEMPERATURE A 0 etc. -- REPRENDRE CONFIG DANS RECIPE GENERATION
        if args.model_name in big_model_list:
            dtype_m = 'float16'
            if args.model_name in ['gemma-2-big', 'gemma-3']:
                dtype_m = "bfloat16"#torch.bfloat16
                #Compared to vllm, the device_map=auto account automatically for the number of gpus for big models
                model = AutoModelForCausalLM.from_pretrained(model_name, return_dict_in_generate=True, output_hidden_states=True, torch_dtype=dtype_m, device_map="auto", trust_remote_code=True, cache_dir=args.cache_dir)
            elif args.model_name == 'microsoft':
                model = AutoModelForCausalLM.from_pretrained(model_name, return_dict_in_generate=True, output_hidden_states=True, dtype=dtype_m, revision="main", device_map="auto", trust_remote_code=True, cache_dir=args.cache_dir, low_cpu_mem_usage=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, return_dict_in_generate=True, output_hidden_states=True, torch_dtype=dtype_m, revision="main", device_map="auto", trust_remote_code=True, cache_dir=args.cache_dir, low_cpu_mem_usage=True)
                if args.model_name == 'falcon-big':
                    model.model = model.transformer
                    model = add_falcon_compat_aliases(model)
        else:
            dtype_m = torch.float16
            if args.model_name == 'orion':
                model = AutoModelForCausalLM.from_pretrained(model_name, return_dict_in_generate=True, output_hidden_states=True, torch_dtype=dtype_m, device_map="auto", trust_remote_code=True, cache_dir=args.cache_dir)
            else:
                #Compared to vllm, the device_map=auto account automatically for the number of gpus for big models
                model = AutoModelForCausalLM.from_pretrained(model_name, return_dict_in_generate=True, output_hidden_states=True, torch_dtype=dtype_m, device_map="auto", trust_remote_code=True, cache_dir=args.cache_dir)

        if args.model_name == 'orion':
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
        
        #adding middle layers to the model analysis
        if args.add_middle_layers:
            config_model = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            if args.model_name == 'gemma-3':
                num_layers = config_model.text_config.num_hidden_layers
            else:
                num_layers = config_model.num_hidden_layers
            
            ##Modify here depending on how many layers you want to analyze
            mid1 = num_layers // 2 - 1
            middle_two = [mid1]    #Less layers because it takes way too much time otherwise -- takes already 3 days per models here
            #Inserting middle_two at the proper positions in layers
            max_pos = max([x for x in args.layers if x >= 0], default=None)   # largest positive
            min_neg = min([x for x in args.layers if x < 0], default=None)    # smallest negative
            insert_idx = args.layers.index(max_pos) + 1 if max_pos is not None else len(args.layers)
            layers_ = args.layers[:insert_idx] + middle_two + args.layers[insert_idx:]
        else:
            layers_ = args.layers

    for k in tqdm(range(len(filtered_filenames))):
        recette = filtered_filenames[k]
        file_path = data_path + recette
        with open(file_path) as json_file:
            recipe_dict = json.load(json_file)

        ## We set to 0 the distance here for each recipe -- difference needs to estimate distance between all points. 
        #This serves as optim to not calculate for each varaitions but only once since it is the same distance for all KB
        neighboroud_distance  = 0. 
        
        #### Skipping errors if there are no reference recipes in the dataset
        KB_recettes_, _ = data_analysis(recipe_dict)

        if len(KB_recettes_) <= 0:
            continue
        
        #### NOW WE HAVE n_layers + the original KB_recettes
        if args.score_embeddings:
            KB_embeddings, KB_prob_sentences = logit_lens(KB_recettes_, model, tokenizer, preserved_layers=layers_, model_name=args.model_name)
            ## Getting the reference based intermediate layers decoded sentence to compare where LLMs dereive ## List of recipes then list of [n_layers, seq_len, hidden_dim]
            #Pairing all KB_sentences from same layers in the same list:

            KB_grouped_by_layer = list(map(list, zip(*KB_prob_sentences)))  #sent_grouped_by_layer[0] → [sent_1_lay_1, sent_2_lay_1, sent_3_lay_1,...]

            #Mandatory cleaning steps for homogenesing with pre-processing done to the orgiinal recipes
            clean_KB_layers = []
            for i in range(len(KB_grouped_by_layer)):
                clean_KB_layers.append(text_cleaning(KB_grouped_by_layer[i]))
            #final_recipes - combine with the original processed sentences
            final_KB_recipes = [KB_recettes_] + clean_KB_layers
        else:
            final_KB_recipes = [KB_recettes_]
        
        #For verification purpose that the code works properly only
        checked_list = [final_KB_recipes]

        #We initiate the list as empty list if you don't go on the scoring loop, their length can be counted as 0 to allow keep proper index tracking
        train_recettes, valid_recettes, test_recettes, LLM_recettes, recette_variations = [], [], [], [], []
        
        ## Setting to False if the train, valid, or test are empty -- prevent errors on assert len
        do_score_train = args.score_train if recipe_dict['Train_Variations'] else False
        do_score_valid = args.score_valid if recipe_dict['Valid_Variations'] else False
        do_score_test = args.score_test if recipe_dict['Test_Variations'] else False
        do_score_llms = args.score_llms if recipe_dict['LLM_gen'] else False

        if do_score_train:
            train_recettes, train_indexes  = data_analysis(recipe_dict, ref=False, col_name='Train_Variations')
            if args.score_embeddings:
                train_embeddings, train_prob_sentences = logit_lens(train_recettes, model, tokenizer, preserved_layers=layers_, model_name=args.model_name)
                train_grouped_by_layer = list(map(list, zip(*train_prob_sentences)))
                
                #Mandatory cleaning steps for homogenesing with pre-processing done to the orgiinal recipes
                clean_train_layers = []
                for i in range(len(train_grouped_by_layer)):
                    clean_train_layers.append(text_cleaning(train_grouped_by_layer[i]))
                selected_train_recettes = [train_recettes] + clean_train_layers
            else:
                selected_train_recettes = [train_recettes]
            checked_list.append(selected_train_recettes)

        if do_score_valid:
            valid_recettes, valid_indexes  = data_analysis(recipe_dict, ref=False, col_name='Valid_Variations')
            if args.score_embeddings:
                valid_embeddings, valid_prob_sentences = logit_lens(valid_recettes, model, tokenizer, preserved_layers=layers_, model_name=args.model_name)
                valid_grouped_by_layer = list(map(list, zip(*valid_prob_sentences)))
                clean_valid_layers = []
                for i in range(len(valid_grouped_by_layer)):
                    clean_valid_layers.append(text_cleaning(valid_grouped_by_layer[i]))
                selected_valid_recettes = [valid_recettes] + clean_valid_layers
            else:
                selected_valid_recettes = [valid_recettes]
            checked_list.append(selected_valid_recettes)

        if do_score_test:
            test_recettes, test_indexes  = data_analysis(recipe_dict, ref=False, col_name='Test_Variations')
            if args.score_embeddings:
                test_embeddings, test_prob_sentences = logit_lens(test_recettes, model, tokenizer, preserved_layers=layers_, model_name=args.model_name)
                test_grouped_by_layer = list(map(list, zip(*test_prob_sentences)))
                clean_test_layers = []
                for i in range(len(test_grouped_by_layer)):
                    clean_test_layers.append(text_cleaning(test_grouped_by_layer[i]))
                selected_test_recettes = [test_recettes] + clean_test_layers
            else:
                selected_test_recettes = [test_recettes]
            checked_list.append(selected_test_recettes)
        
        if do_score_llms:
            if args.gen_type == 'dist':
                LLM_recettes, _ = data_analysis(recipe_dict, ref=False, col_name='LLM_gen') #output = list
            elif args.gen_type == 'same':
                LLM_recettes, _ = data_analysis(recipe_dict, ref=False, col_name='LLM_gen', gen_type='same') #output = list
            
            #Cleaning LLM recipes here -- files from json original like KB comes clean due to the pre-processing when creating the Global fusion dataset : column 'recipe_clean'
            LLM_recettes = text_cleaning(LLM_recettes)
            if args.score_embeddings:
                LLM_embeddings, LLM_prob_sentences = logit_lens(LLM_recettes, model, tokenizer, preserved_layers=layers_, model_name=args.model_name)
                LLM_grouped_by_layer = list(map(list, zip(*LLM_prob_sentences)))
                #Cleaning the list of recipes per layer
                clean_LLM_layers = []
                for i in range(len(LLM_grouped_by_layer)):
                    clean_LLM_layers.append(text_cleaning(LLM_grouped_by_layer[i]))
                selected_LLM_recettes = [LLM_recettes] + clean_LLM_layers
            else:
                selected_LLM_recettes = [LLM_recettes]
            checked_list.append(selected_LLM_recettes)
        
        assert all(len(l) == len(checked_list[0]) for l in checked_list), "Not all lists have the same length"

        for r in range(len(final_KB_recipes)):  ### here len(final_KB_recipes) == layers_
            # For saving purpose
            if r == (len(final_KB_recipes) -1):
                layer_name = 'last'
            elif r == 0:
                layer_name = 'original'
            else:
                layer_name = layers_[r]
            
            #Update the recette variation DB depending on which data we want to analyze and measure
            KB_recettes = final_KB_recipes[r]
            if do_score_train:
                recette_variations = selected_train_recettes[r]
                assert len(train_recettes) == len(selected_train_recettes[r]), "You have a matching problem for list of recipes"
            if do_score_valid:
                recette_variations = recette_variations + selected_valid_recettes[r]
                assert len(valid_recettes) == len(selected_valid_recettes[r]), "You have a matching problem for list of recipes"
            if do_score_test:
                recette_variations = recette_variations + selected_test_recettes[r]
                assert len(test_recettes) == len(selected_test_recettes[r]), "You have a matching problem for list of recipes"
            if do_score_llms:
                recette_variations = recette_variations + selected_LLM_recettes[r]
                assert len(LLM_recettes) == len(selected_LLM_recettes[r]), "You have a matching problem for list of recipes"

            KB_texts = ' '.join(KB_recettes).split()
            EB_PMI = pmi(KB_texts)
            dict_know_pmi = pmi_to_dict(EB_PMI)

            KB_matrix, KB_dist, Count_matrix = docs_distribution(KB_recettes, recette_variations)
            KB_size = list(range(KB_matrix.shape[0]))

            ### Before computing novelty scores - initializing information for metadata
            KB_country = recipe_dict['Country']
            KB_ingr_list = recipe_dict['Reference_Base']['AllIngredients']

            ########### Train recipes
            if do_score_train:
                country_list_train, ingr_list_train, train_raw_len, train_clean_len, train_raw_uniq, train_clean_uniq = get_info(train_indexes, recipe_dict, KB_country, index_name='Train_Variations')
                new_ingr_train = get_new_ingr(ingr_list_train, KB_ingr_list)
                for i in range(len(selected_train_recettes[r])):
                    select_variation = KB_size + [len(KB_size)+i]
                    NewKB_dist, variation_dist = new_distribution(Count_matrix, select_variation)

                    KB_updated = [selected_train_recettes[r][i]]
                    updated_text = ' '.join(KB_updated).split()
                    New_EB_PMI = pmi(updated_text)

                    results_train, neighboroud_distance = compute_scores(KB_matrix, KB_dist, NewKB_dist, variation_dist, EB_PMI, dict_know_pmi, New_EB_PMI, neighboroud_distance, config.types, config.thrs)
                    
                    current_index = train_indexes[i]
                    recipe_dict['Train_Variations'][current_index].setdefault(f'layer_{layer_name}', {})
                    for key, value in results_train.items():
                        recipe_dict['Train_Variations'][current_index][f'layer_{layer_name}'][key] = value
                    
                    #Saving results in the recipe json
                    recipe_dict['Train_Variations'][current_index][f'layer_{layer_name}']['recipe_text_cleaned'] = selected_train_recettes[r][i]
                    recipe_dict['Train_Variations'][current_index][f'layer_{layer_name}']['nb_ingredients'] = len(ingr_list_train[i])
                    recipe_dict['Train_Variations'][current_index][f'layer_{layer_name}']['nb_new_ingredients'] = new_ingr_train[i]
                    recipe_dict['Train_Variations'][current_index][f'layer_{layer_name}']['text_length'] = train_raw_len[i]
                    recipe_dict['Train_Variations'][current_index][f'layer_{layer_name}']['clean_text_length'] = train_clean_len[i]
                    recipe_dict['Train_Variations'][current_index][f'layer_{layer_name}']['nb_uniq_tokens'] = train_raw_uniq[i]
                    recipe_dict['Train_Variations'][current_index][f'layer_{layer_name}']['clean_nb_uniq_tokens'] = train_clean_uniq[i]

            ######## Validation
            if do_score_valid:
                country_list_valid, ingr_list_valid, valid_raw_len, valid_clean_len, valid_raw_uniq, valid_clean_uniq = get_info(valid_indexes, recipe_dict, KB_country, index_name='Valid_Variations')
                new_ingr_valid = get_new_ingr(ingr_list_valid, KB_ingr_list)
                for i in range(len(selected_valid_recettes[r])):
                    # We let the train_recettes here because it's len is 0 if we don't include train + the assert l.158 verify the lengths are equivalent
                    select_variation = KB_size + [(len(KB_size)+len(train_recettes))+i]  
                    NewKB_dist, variation_dist = new_distribution(Count_matrix, select_variation)
            
                    KB_updated = [selected_valid_recettes[r][i]] ## We don't use valid_recettes here though because we want the proper variation
                    updated_text = ' '.join(KB_updated).split()
                    New_EB_PMI = pmi(updated_text)
                    
                    #Computing novelty scores
                    results_valid, neighboroud_distance = compute_scores(KB_matrix, KB_dist, NewKB_dist, variation_dist, EB_PMI, dict_know_pmi, New_EB_PMI, neighboroud_distance, config.types, config.thrs)

                    current_index = valid_indexes[i]
                    recipe_dict['Valid_Variations'][current_index].setdefault(f'layer_{layer_name}', {})
                    for key, value in results_valid.items():
                        recipe_dict['Valid_Variations'][current_index][f'layer_{layer_name}'][key] = value

                    recipe_dict['Valid_Variations'][current_index][f'layer_{layer_name}']['recipe_text_cleaned'] = selected_valid_recettes[r][i]
                    recipe_dict['Valid_Variations'][current_index][f'layer_{layer_name}']['nb_ingredients'] = len(ingr_list_valid[i])
                    recipe_dict['Valid_Variations'][current_index][f'layer_{layer_name}']['nb_new_ingredients'] = new_ingr_valid[i]
                    recipe_dict['Valid_Variations'][current_index][f'layer_{layer_name}']['text_length'] = valid_raw_len[i]
                    recipe_dict['Valid_Variations'][current_index][f'layer_{layer_name}']['clean_text_length'] = valid_clean_len[i]
                    recipe_dict['Valid_Variations'][current_index][f'layer_{layer_name}']['nb_uniq_tokens'] = valid_raw_uniq[i]
                    recipe_dict['Valid_Variations'][current_index][f'layer_{layer_name}']['clean_nb_uniq_tokens'] = valid_clean_uniq[i]
            
            ############## Test
            if do_score_test:
                country_list_test, ingr_list_test, test_raw_len, test_clean_len, test_raw_uniq, test_clean_uniq = get_info(test_indexes, recipe_dict, KB_country, index_name='Test_Variations')
                new_ingr_test = get_new_ingr(ingr_list_test, KB_ingr_list)
                for i in range(len(selected_test_recettes[r])):
                    select_variation = KB_size + [(len(KB_size)+len(train_recettes)+len(valid_recettes))+i]
                    NewKB_dist, variation_dist = new_distribution(Count_matrix, select_variation)
            
                    KB_updated = [selected_test_recettes[r][i]]
                    updated_text = ' '.join(KB_updated).split()
                    New_EB_PMI = pmi(updated_text)
            
                    results_test, neighboroud_distance = compute_scores(KB_matrix, KB_dist, NewKB_dist, variation_dist, EB_PMI, dict_know_pmi, New_EB_PMI, neighboroud_distance, config.types, config.thrs)

                    current_index = test_indexes[i]
                    recipe_dict['Test_Variations'][current_index].setdefault(f'layer_{layer_name}', {})
                    for key, value in results_test.items():
                        recipe_dict['Test_Variations'][current_index][f'layer_{layer_name}'][key] = value
                    
                    recipe_dict['Test_Variations'][current_index][f'layer_{layer_name}']['recipe_text_cleaned'] = selected_test_recettes[r][i]
                    recipe_dict['Test_Variations'][current_index][f'layer_{layer_name}']['nb_ingredients'] = len(ingr_list_test[i])
                    recipe_dict['Test_Variations'][current_index][f'layer_{layer_name}']['nb_new_ingredients'] = new_ingr_test[i]
                    recipe_dict['Test_Variations'][current_index][f'layer_{layer_name}']['text_length'] = test_raw_len[i]
                    recipe_dict['Test_Variations'][current_index][f'layer_{layer_name}']['clean_text_length'] = test_clean_len[i]
                    recipe_dict['Test_Variations'][current_index][f'layer_{layer_name}']['nb_uniq_tokens'] = test_raw_uniq[i]
                    recipe_dict['Test_Variations'][current_index][f'layer_{layer_name}']['clean_nb_uniq_tokens'] = test_clean_uniq[i]

            ############## Generated by the models
            if do_score_llms:
                llm_dict = recipe_dict['LLM_gen']
                if 'same_country_novelty' in llm_dict.keys():
                    llm_same = llm_dict['same_country_novelty']
                    same_key_list = list(llm_same.keys())
                    
                    ## For the same country generated
                    for i in range(len(same_key_list)):
                        select_variation = KB_size + [(len(KB_size)+len(train_recettes)+len(valid_recettes))+len(test_recettes)+i]

                        NewKB_dist, variation_dist = new_distribution(Count_matrix, select_variation)
                        KB_updated = [selected_LLM_recettes[r][i]]
                        updated_text = ' '.join(KB_updated).split()
                        New_EB_PMI = pmi(updated_text)

                        #Computing novelty scores
                        results_, neighboroud_distance = compute_scores(KB_matrix, KB_dist, NewKB_dist, variation_dist, EB_PMI, dict_know_pmi, New_EB_PMI, neighboroud_distance, config.types, config.thrs)
                        
                        llm_same[same_key_list[i]].setdefault(f'layer_{layer_name}', {})
                        for key_result, value in results_.items():
                            llm_same[same_key_list[i]][f'layer_{layer_name}'][key_result] = value

                        #Also saving the text processed for the analysis
                        llm_same[same_key_list[i]][f'layer_{layer_name}']['recipe_text_cleaned'] = selected_LLM_recettes[r][i]

                ## ## here we have N variations
                llm_varia = llm_dict['variation_novelty']
                key_countries = list(llm_varia.keys())  

                for key_c in key_countries:
                    key_list_novel = list(llm_varia[key_c].keys())
                    for i in range(len(key_list_novel)):
                        #if key_list_novel[i] != 'country': ##because for the first key_country which is country_origin we have a country key
                        select_variation = KB_size + [(len(KB_size)+len(train_recettes)+len(valid_recettes))+len(test_recettes)+len(same_key_list)+i]

                        NewKB_dist, variation_dist = new_distribution(Count_matrix, select_variation)
                        KB_updated = [selected_LLM_recettes[r][i]]
                        updated_text = ' '.join(KB_updated).split()
                        New_EB_PMI = pmi(updated_text)

                        #Computing novelty scores
                        results_, neighboroud_distance = compute_scores(KB_matrix, KB_dist, NewKB_dist, variation_dist, EB_PMI, dict_know_pmi, New_EB_PMI, neighboroud_distance, config.types, config.thrs)
                        
                        llm_varia[key_c][key_list_novel[i]].setdefault(f'layer_{layer_name}', {})
                        
                        for key_result, value in results_.items():
                            llm_varia[key_c][key_list_novel[i]][f'layer_{layer_name}'][key_result] = value

                        #Also saving the text processed for the analysis
                        llm_varia[key_c][key_list_novel[i]][f'layer_{layer_name}']['recipe_text_cleaned'] = selected_LLM_recettes[r][i]


        #Saving final results
        short_ = recette.removesuffix(".json")
        file_name = save_path_model + f"{short_}_withscore.json"

        with open(file_name, "w") as outfile:
            json.dump(recipe_dict, outfile)

        del recipe_dict, llm_dict