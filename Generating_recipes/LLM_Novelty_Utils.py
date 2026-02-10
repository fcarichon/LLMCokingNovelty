import re
from os import walk
from langdetect import detect
import spacy
nlp = spacy.load("en_core_web_sm")
import re



def file_matching(orig_temp, gen_recipe, id_=False):

    #Function to identify the original filename of my recipes to concatenate all versions 
    #clean LLMs generated recipes with variations and KB from original file
    match = re.match(r'^(.+?_[^_]+?_(\d+))(?=(_|\.|$))', gen_recipe)
    recipe_name = match.group(1) if match else gen_recipe

    for i, orig_recipe in enumerate(orig_temp):
        if recipe_name in orig_recipe:
            if id_:
                print(i)
            return recipe_name, orig_recipe
    
    raise KeyError('No file matching', print(gen_recipe))


def analyze_text_ingredient(text):

    # Getting title position
    text_lower = text.lower()
    ingredient_matches = [m.start() for m in re.finditer(r'\bingredient\w*\b', text_lower)]
    first_ingredient = ingredient_matches[0] if len(ingredient_matches) > 0 else None
    if first_ingredient is not None:
        return True, first_ingredient
    else:
        return False, first_ingredient
    
def analyze_text_instruction(text, pos_ingre):

    # Getting title position
    text_lower = text.lower()
    instruction_matches = [m.start() for m in re.finditer(r'\binstruction\w*\b', text_lower)]
    first_instruction = instruction_matches[0] if len(instruction_matches) > 0 else None
    
    if first_instruction is not None:
        if first_instruction > pos_ingre:                    #We select recipes only if they are the proper ordering because otherwise it is noise
            return True
        else:
            return False
    else:
        return False 
    

def split_text_sequentially(text):
    """
    Splits the text sequentially:
    1. First, at 'first_keyword'.
    2. Then, at the first occurrence of any word in 'second_keywords', but only in the 'after' part of the first split.
    :param text: The input text to be split.
    :return: A dictionary with the split results.
    """
    results = {}
    text = text.lower()
    match = re.search(r'\bingred\w*\b', text, re.IGNORECASE)
    if match:
        split_index = match.start()
        before = text[:split_index].strip()
        after = text[split_index:].strip()
        results['Title'] = before
    else:
        results["Title"] = None
        return results  # If the first keyword is not found, return immediately
    
    ###########################################################################################
    # Step 2: Split the 'after' part at the first occurrence of any word in `second_keywords`
    ##########################################################################################
    second_keywords = ["instruction", "method", "description","instructions", "methods", "descriptions"]
    second_pattern = r'(' + '|'.join(map(re.escape, second_keywords)) + r')'
    match_second = re.search(second_pattern, after, re.IGNORECASE)
    
    if match_second:
        split_index = match_second.start()
        before_second = after[:split_index].strip()
        after_second = after[split_index:].strip()
        results["Ingredients"] = before_second
        results["Instructions"] = after_second
    else:
        results["Ingredients"] = None  # If no second keyword is found

    return results

### CHERCHER LA FONCTION DANS LE VIEUX CODE -- CLEANING DES RECETTES EN MÊME TEMPS -- QUE TU FASSES PAS DEUX FOIS LE TRAVAIL POUR RIEn
def text_cleaning(recipe, lemma="True", authrorized_pos= ['PROPN', 'PRON', 'ADJ', 'ADV', 'NOUN', 'NUM', 'VERB']):
    recipe_doc = nlp(str(recipe))
    clean_list = []
    for token in recipe_doc:
        if token.pos_ in authrorized_pos:
            if not token.is_stop:
                if lemma:
                    clean_list.append(token.lemma_)
                else:
                    clean_list.append(token)

    #### Re,sub -- \n et \t
                
    clean_recipe = ' '.join(clean_list)
    return clean_recipe

def clean_ingr(ingr_str):
    # Preprocess: split, strip, and clean bullets/tabs -- Return clean list of ingredients
    lines = [line.strip().lstrip("*+\t ") for line in ingr_str.split("\n") if line.strip()]
    cleaned_ingredients = []

    units = {
        "cup", "cups", "tablespoon", "tablespoons", "teaspoon", "teaspoons",
        "clove", "cloves", "gram", "grams", "ounce", "ounces", "oz", "ml", "l",
        "pound", "pounds", "tbsp", "tsp", "kg", "g", "pinch"
    }

    for line in lines:
        if line.lower().startswith("ingr"):
            continue

        # If parentheses exist with comma-separated values, extract them
        match = re.search(r'\(([^)]+)\)', line)
        if match:
            items = [item.strip() for item in match.group(1).split(',')]
            for item in items:
                full_item = item  # Use just the item; omit outer context to avoid clumsiness
                doc = nlp(full_item)
                tokens = [
                    token.text for token in doc
                    if not token.like_num
                    and token.text.lower() not in units
                    and not token.is_stop
                    and token.is_alpha or "-" in token.text
                ]
                cleaned = " ".join(tokens)
                if cleaned:
                    cleaned_ingredients.append(cleaned)
            # Remove the entire parenthetical for main line cleanup
            line = re.sub(r'\([^)]*\)', '', line).strip()

        # Clean the rest of the line (outside the parentheses)
        doc = nlp(line)
        tokens = [
            token.text for token in doc
            if not token.like_num
            and token.text.lower() not in units
            and not token.is_stop
            and token.is_alpha or "-" in token.text
        ]
        cleaned_line = " ".join(tokens)
        if cleaned_line:
            cleaned_ingredients.append(cleaned_line)

    return cleaned_ingredients

def clean_title(raw_title):
    #For the title just return what is after the colon : if there are some, otherwise keep the raw one
    if ":" in raw_title:
        parts = raw_title.split(":", 1)
        return parts[1].strip()
    return raw_title.strip()