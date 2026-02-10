from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import json
import os
from os import walk
import argparse
import sys
from vllm import LLM, SamplingParams
import re


from model_utils import getting_recipes, get_template_for_recipes
from model_registry import MODEL_REGISTRY
import config_genLLMs as config_genLLMs

#Building block for LLMs models
def build_model(args, model_name, in_big, tokenizer):
    
    if in_big:
        dtype_m = torch.float16 #'float16'
        if model_name == 'google/gemma-3-27b-it' or model_name == 'google/gemma-2-27b-it':
            dtype_m = "bfloat16"
        model = LLM(model=model_name, task="generate", revision='main', max_model_len=args.model_token_len, max_num_batched_tokens=args.model_token_len, tokenizer=model_name, download_dir=args.cache_dir, trust_remote_code=True, dtype=dtype_m, tensor_parallel_size=args.nb_gpus)
    elif model_name == 'Qwen3-30B-A3B-Instruct-2507':
        dtype_m = torch.float16
        #Set max_len_token to 512 for Qwen3 to reduce cache usage and prevent OOM         # a bit more conservative
        model = LLM(model=model_name, task="generate", revision='main', max_model_len=args.model_token_len, max_num_batched_tokens=512, tokenizer=model_name, download_dir=args.cache_dir, trust_remote_code=True, dtype=dtype_m, tensor_parallel_size=args.nb_gpus, gpu_memory_utilization=0.85,enforce_eager=True)
    else:
        model = LLM(model=model_name, task="generate", revision='main', max_model_len=args.model_token_len, tokenizer=model_name, download_dir=args.cache_dir, trust_remote_code=True, dtype='float16')
    
    return model

def main():
    #Defining the home path
    #Getting home_path to have the path to actually go to the recipes

    #############################################################################################
    ##############################   PARSER ARGUMENTS   #########################################
    parser = argparse.ArgumentParser(description="Recipe Generator for GlobalFusion")
    parser.add_argument("--model", default='gemma-2-small', help="LLMs model for genetaion. Accepted values : llama3-base | llama3-instruct | gemma-2 | bloom | deepseek")
    parser.add_argument("--run", default=0, type=int, help="indicate the run number")
    parser.add_argument("--config", default=1, type=int, help="indicate the config number to use -- 3 poissibilities 1 for balanced crea 2 for strict model 3 for divergent generation")
    parser.add_argument("--cache_dir", default=config_genLLMs.path['cache_dir'], help="path to model cache directory")
    parser.add_argument("--save_path", default=config_genLLMs.path['save_dir'], help="path to general directory")
    parser.add_argument('--recipe_dir', default=config_genLLMs.path['recipe_dir'], help="path to the original recipe directory")
    parser.add_argument('--gen_type', default='same', help='if same -- new generation mode with same country from LLM and GlobalFusion, if dist, gen LLMs based on sampling countries on cult_dist')
    parser.add_argument('--model_token_len', default=1024, type=int, help="input token length for vLLLM models")
    parser.add_argument('--nb_gpus', default=2, type=int, help='tensor_parallel_size in vLLMs for large models only')
    #### ADD POTENTIAL OPTION TO CHOOSE WHICH DISTANCE TO COMPUTE WHEN RELEASE -- IT'S FINE FOR NOW

    args = parser.parse_args() 

    ###########################################################################################################################
    ##############################################   Loading models    ########################################################
    #Exclue qwen 3 from big list because the model needs another exception
    big_model_list = ['llama3-big', 'gemma-2-big', 'gemma-3','falcon-big', 'apertus', 'bloom','microsoft', 'qwen_2']
    in_big = args.model in big_model_list

    try:
        model_name = MODEL_REGISTRY[args.model]
    except KeyError:
        raise ValueError(f"Unknown model: {args.model}")

    #Loading tokenizer
    if args.model == 'orion':
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)

    model = build_model(args, model_name, in_big, tokenizer)

    ###########################################################################################################################
    ###########################################################################################################################
    # Loading the configs
    config_ = f'config_{args.config}'
    if config_ == 'config_1':
        configs = config_genLLMs.hp_balanced
    elif config_ == 'config_2':
        configs = config_genLLMs.hp_strict
    elif config_ == 'config_3':
        configs = config_genLLMs.hp_divergent
    else:
        print('Please provide a valid entry for configurations -- ')
        raise KeyError

    ###########################################################################################################################
    ###########################################################################################################################
    # Loading templates, recipes, and saving path
    run = f'_run_{args.run}'

    model_configs = args.model + run + config_
    save_path = args.save_path + f'Generated_answers_{model_configs}/'
    
    #Creating the directory if the saving path does not already exists
    save_directory = os.path.dirname(save_path)
    if save_directory and not os.path.exists(save_directory):
        os.makedirs(save_directory)
        already_saved_files = []
        print(f"Created directory: {save_directory}")
    else:
        already_saved_files = list(set(next(walk(save_path), (None,None,[]))[2]))

    # Loading the proper generation templates for our recipes -- different options are possible between with or without KB and with instruction only or full reccipe generation
    templates_path = config_genLLMs.path['template_path']

    ###########################################################################################################################
    ######################################              Laoding the filenames            ######################################
    #The List evolve depending on the number of already processed files
    filenames_recipes = list(set(next(walk(args.recipe_dir), (None,None,[]))[2]))
    filenames_templates = list(set(next(walk(templates_path), (None,None,[]))[2]))
    if len(already_saved_files) > 0:
        common_bases = set(re.sub(r"(_recipe_\d+).*", r"\1", name) for name in already_saved_files)
        filenames_recipes = [name for name in filenames_recipes if not any(name.startswith(base) for base in common_bases)]
        filenames_templates = [name for name in filenames_templates if not any(name.startswith(base) for base in common_bases)]

    filenames = get_template_for_recipes(filenames_templates, filenames_recipes)
    print('Number of files to process : ', len(filenames))

    ###########################################################################################################################
    ##############################################    Generating recipes   ####################################################
    for recipe, template in filenames.items():
        print(recipe)
        #Saving dictionary for the final results 
        gen_new_noveldict = {}
        #Loading the dictionary with the recipe
        recipe_pathfile = args.recipe_dir + '/' + recipe
        with open(recipe_pathfile) as json_file:
            recipe_dict = json.load(json_file)
        
        # Loading the novelty templates for the recipe
        template_pathfile = templates_path + '/' + template
        with open(template_pathfile) as json_file:
            template_dict = json.load(json_file)

        #Measuring average recipe length -- reference base for generating the traditional and novel ones from same country 
        country = recipe_dict['Country']
        country_templates = template_dict['general_novelty']['country_origin']
        recipe_name = recipe_dict['Recipe_Name']

        ##Exceptions forcountry due to dataset issues on this specific example
        if country == 'congo':
            country = 'united states'
            
        #Making sure that we are generating the proper variations
        assert country_templates.lower() == country.lower()

        #Loading the recipes of the KB for the KB Templates since we need to provide them as input to the model too
        if args.template == 'KB_templates':
            reicpe_list = recipe_dict['Reference_Base']
            recipe_KB = getting_recipes(reicpe_list)

    
        gen_new_noveldict['country_origin'] = {}
        gen_new_noveldict['country_origin']['country'] = country
        
        ###########################################################################
        ### Generation for same country novelty

        #Getting the batched templates: 
        gennovel_dict = {k: v for k, v in template_dict['general_novelty'].items() if k != "country_origin"}
        key_list = list(gennovel_dict.keys())
        prompt_list = []
        for key in key_list:
            #start_time = time.time()
            if args.template == 'KB_templates':
                prompt_list.append(recipe_KB + gennovel_dict[key])
            else: 
                prompt_list.append(gennovel_dict[key])
        
        #Generation
        sampling_params = SamplingParams(max_tokens=configs['gen_len'], temperature=configs['temp_value'], top_p=configs['top_p_value'], 
                                            top_k=configs['top_k_value'], repetition_penalty=configs['repet_value'])
        outputs = model.generate(prompt_list, sampling_params=sampling_params, use_tqdm=False)
        #Saving outputs
        for i, key in enumerate(key_list):
            gen_new_noveldict['country_origin'].setdefault(key, {})['prompt'] = outputs[i].prompt
            gen_new_noveldict['country_origin'][key]['answer'] = outputs[i].outputs[0].text
        
        ###########################################################################
        ### For all the variation countries 
        varia_dict = template_dict['variation_novelty']
        country_keys = list(varia_dict.keys()) ## Format = variation_country
        
        if len(country_keys) > 0:  ## Apparently I have some recipes with no variation -- Jab Chae recipe... Weird...
            prompt_list = []
            for country in country_keys:
                gen_new_noveldict.setdefault(country, {})
                genvaria_dict = {k: v for k, v in varia_dict[country].items() if k not in ["country_origin", "variation_country"]}
                key_list = list(genvaria_dict.keys())
                
                for key in key_list:
                    if args.template == 'KB_templates':
                        prompt_list.append(recipe_KB + genvaria_dict[key])
                    else: 
                        prompt_list.append(genvaria_dict[key])

            ### Generation
            sampling_params = SamplingParams(max_tokens=configs['gen_len'], temperature=configs['temp_value'], top_p=configs['top_p_value'], 
                                                top_k=configs['top_k_value'], repetition_penalty=configs['repet_value'])
            outputs = model.generate(prompt_list, sampling_params=sampling_params, use_tqdm=False)
           
            ###########################################################################
            ### Reorganizing the save of answers due to the paralellization
            ### Here the length of the output is perfectly divided by / len(country_keys)
            chunk_size = int(len(outputs) / len(country_keys))
            for i, country in enumerate(country_keys):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                genvaria_dict = {k: v for k, v in varia_dict[country].items() if k not in ["country_origin", "variation_country"]}
                key_list = list(genvaria_dict.keys())

                output_country = outputs[start_idx:end_idx]
                assert len(output_country) == len(key_list)

                for j, varia_key in enumerate(key_list):
                    if varia_key not in gen_new_noveldict[country]:
                        gen_new_noveldict[country][varia_key] = {}
                    gen_new_noveldict[country][varia_key]['prompt'] = output_country[j].prompt
                    gen_new_noveldict[country][varia_key]['answer'] = output_country[j].outputs[0].text

        ###########################################################################################################
        ###########################################################################################################
        #SAVING FILES
        # Split the base name and extension
        base_name, ext = os.path.splitext(recipe)
        new_filename = f"{base_name}_{model_configs}_gen{ext}"
        file_path_save = f"{save_path}{new_filename}"

        # Save the nested dictionary to a JSON file
        with open(file_path_save, "w") as json_file:
            json.dump(gen_new_noveldict, json_file, indent=4)

if __name__ == "__main__":
    main()