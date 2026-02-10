import json
import pandas as pd
from scipy.spatial import distance
import os
from os import walk
from tqdm import tqdm
import argparse
import random
from data_utils import template_gen

#############################################################################################
##############################   PARSER ARGUMENTS   #########################################
###################   Default Values are the value used in the paper ########################
parser = argparse.ArgumentParser(description="Templates Generator for GlobalFusion")
parser.add_argument("--templates", default='reg_templates', help="Type of templates to generate - possible values: reg_templates or KB_templates")
parser.add_argument('--instruct_only', default=False, help="True then the model generate only the instruction for the recipe other wise it generates the raw answer python script.py --instruct_only False")
parser.add_argument('--dir_templates', default=f'.', help='Path to directory of novelty templates')
parser.add_argument('--templates_file', default=f'./Novelty_Templates.json', help='Template file name')
parser.add_argument('--dir_recipes', default=f'./GlobalFusion/', help='Path to directory containing GlobalFusion dataset')
parser.add_argument('--path_countries', default=f'./countries.csv', help='Path to directory containing all distances')
parser.add_argument('--save_dir', default=f'./Templates/', help='Path to directory for saving the generated templates')
parser.add_argument("--country_sampling", default="same", help="If same, sample countries from the list of existing countries in the original files, if dist then it samples countries from the different cultural distances")
parser.add_argument("--sampling_method", default="uniq", help="if uniq, keep one instance of every country, if random select radom size sample keeping distribution")
parser.add_argument("--sample_size", type=int, default=12, help="Number of coutries to sample to generate templates")
args = parser.parse_args() 

#############################################################################################
############################## Loading Templates and recipes ################################
#Defining the saving path for the diffrent files
save_path = args.save_dir

#save_dir = os.path.dirname(save_path)
if save_path and not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Created directory: {save_path}")

#Novelty templates
novelty_file = args.dir_templates + args.templates_file
with open(novelty_file) as json_file:
    dict_template = json.load(json_file)

#List of recipes names
filenames = list(set(next(walk(args.dir_recipes), (None,None,[]))[2]))

#############################################################################################
################### Generating the templates for all recipes ################################
#Getitng the first table to have correspondance between codes, country names and nationaltyas in the original dataset constitution
country_nat = pd.read_csv(args.path_countries)

#Loading the class for generating the templates
gen_temp = template_gen(dict_template, country_nat)

##ITerating over GlobalFusion recipes -- generating templates for each paired countries in GF
for k in tqdm(range(len(filenames))):
    filename = filenames[k]
    file_path = args.dir_recipes + filename
    with open(file_path) as json_file:
        recipe_dict = json.load(json_file)

    country_orig = recipe_dict['Country']
    recipe_name = recipe_dict['Recipe_Name']

    ##Exceptions forcountry due to GlobalFusion issues on this specific example
    if country_orig == 'congo':
        country_orig = 'united states'
    
    #Generating the templates for novelty variations
    novelty_templates = gen_temp.novelty_templates(recipe_name, country_orig, temp_name=args.templates)

    #Mixing / Fusing all dictionnaries together to create the final json for the recipes
    saving_dict = {'Recipe_Name': recipe_name, 'country_orig': country_orig, 'general_novelty': novelty_templates}

    #Taking the list of total countries in the list of training / valid / test recipes
    countries = []
    train_indexes = list(recipe_dict["Train_Variations"].keys())
    for index in train_indexes:
        countries.append(recipe_dict["Train_Variations"][index]['country'])

    valid_indexes = list(recipe_dict["Valid_Variations"].keys())
    for index in valid_indexes:
        countries.append(recipe_dict["Valid_Variations"][index]['country'])

    test_indexes = list(recipe_dict["Test_Variations"].keys())
    for index in test_indexes:
        countries.append(recipe_dict["Test_Variations"][index]['country'])

    #Filtering none values
    countries = [item for item in countries if isinstance(item, str) and item]
    countries = [item for item in countries if item != country_orig]

    #Here we can sample at random or keeping one instance of each country
    if args.sampling_method == "uniq":
        sampled_countries = list(set(countries))
    else:
        sampled_countries = []
        for i in range(args.sample_size -1):
            sampled_countries.append(random.choice(countries))

    saving_dict['variation_novelty'] = {}
    for country_v in sampled_countries:
        temp_dict_template = gen_temp.variations_templates(recipe_name, country_orig, country_v, temp_name=args.templates)
        saving_dict['variation_novelty'][f'variation_{country_v}'] = temp_dict_template

    
    #############################################################################################
    ################### Saving the templates for all recipes ################################
    # Split the base name and extension
    base_name, ext = os.path.splitext(filename)

    # Add "_template" to the base name
    new_filename = f"{base_name}_template{ext}"
    file_path_save = f"{save_path}{new_filename}"
    
    # Save the nested dictionary to a JSON file
    with open(file_path_save, "w") as json_file:
        json.dump(saving_dict, json_file, indent=4)