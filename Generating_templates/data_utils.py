import json
from os import walk
import re

class template_gen():

    def __init__(self, dict_templ, df_country_nat):
        self.dict_templ = dict_templ
        self.df_country_nat = df_country_nat

    def get_nationality(self, country_name):
        """ Function to get nationality based on country name
            INPUT : Specific country / nationality pairing file present in Cultural_Datasets/countries.csv + name of the country
            Output : Equivalent nationalty name
        """
        if country_name == 'Slovak Republic':
            country_name = 'Slovakia'
        if country_name == 'Bosnia And Herzegovina':
            country_name = 'Bosnia and Herzegovina'
        if country_name == "Cote D’Ivoire":
            country_name = "Cote d’Ivoire"
        nationality = self.df_country_nat.loc[self.df_country_nat['Name'] == country_name, 'Nationality'].values
        return nationality[0] if len(nationality) > 0 else None

    def novelty_templates(self, recipe_name, country_origin, temp_name='reg_templates'):
        """
        Generate the 3 traiditionals + 18*3=54 novelty KW + 4*2=8 with novelty definitions templates
        """
        dict_ = {}
        country_origin = country_origin.title()
        nationality_origin = self.get_nationality(country_origin)
        
        dict_['country_origin'] = country_origin

        tradi_ = self.dict_templ['templates'][temp_name]['standard']
        tradi_kw0 = list(self.dict_templ['keywords_trad'].keys()) 
        tradi_kw = list(self.dict_templ['keywords_trad'].values())
        for i, keyword in enumerate(tradi_kw):
            dict_[f'basic_{tradi_kw0[i]}'] = tradi_['basic'].replace("<KW_T>", keyword).replace("<RECIPE_NAME>", recipe_name)
            dict_[f'country_{tradi_kw0[i]}'] = tradi_['basic_country'].replace("<KW_T>", keyword).replace("<NATIONALITY_ORIG>", nationality_origin).replace("<RECIPE_NAME>", recipe_name)
            dict_[f'known_country_{tradi_kw0[i]}'] = tradi_['with_knowledge'].replace("<KW_T>", keyword).replace("<COUNTRY_ORIG>", country_origin).replace("<RECIPE_NAME>", recipe_name)

        novel_ = self.dict_templ['templates'][temp_name]['novelty']
        novel_kw0 = list(self.dict_templ['keywords_new'].keys()) 
        novel_kw1 = list(self.dict_templ['keywords_new'].values())
        #Since with have 18 different novelty keywords that we will test with Romain, I get 3*18=54 templates
        for i, keyword in enumerate(novel_kw1):
            dict_[f'basic_novel_{novel_kw0[i]}'] = novel_['basic'].replace("<KW1>", keyword).replace("<RECIPE_NAME>", recipe_name)
            dict_[f'country_novel_{novel_kw0[i]}'] = novel_['basic_country'].replace("<KW1>", keyword).replace("<RECIPE_NAME>", recipe_name).replace("<NATIONALITY_ORIG>", nationality_origin)
            dict_[f'know_country_novel_{novel_kw0[i]}'] = novel_['with_knowledge'].replace("<KW1>", keyword).replace("<RECIPE_NAME>", recipe_name).replace("<COUNTRY_ORIG>", country_origin)
        
        novelty_def = self.dict_templ['templates'][temp_name]['novelty with definition']
        def_keywords_list = list(self.dict_templ['novelty_definitions'].keys())

        for kw in def_keywords_list:
            definition = self.dict_templ['novelty_definitions'][kw]
            kw1 = self.dict_templ['keywords_new'][kw]
            dict_[f'basic_novel_defined_{kw}'] = novelty_def['basic_definition'].replace("<DEFINITION>", definition).replace("<KW0>", kw).replace("<KW1>", kw1).replace("<RECIPE_NAME>", recipe_name)
            dict_[f'know_country_novel_defined_{kw}'] = novelty_def['with_knowledge_definition'].replace("<DEFINITION>", definition).replace("<COUNTRY_ORIG>", country_origin).replace("<KW0>", kw).replace("<KW1>", kw1).replace("<RECIPE_NAME>", recipe_name)

        return dict_


    def cultural_templates(self, recipe_name, country_origin, country_close, country_mid, country_far, temp_name='reg_templates'):

        dict_ = {}
        country_origin = country_origin.title()

        country_close = country_close.title()
        nationality_close = self.get_nationality(country_close)
        country_mid = country_mid.title()
        nationality_mid = self.get_nationality(country_mid)
        country_far = country_far.title()
        nationality_far = self.get_nationality(country_far)

        dict_['country_close'] = country_close
        dict_['country_mid'] = country_mid
        dict_['country_far'] = country_far
        dict_['country_origin'] = country_origin

        cultural_ = self.dict_templ['templates'][temp_name]['cultural']
        #I want to test generation for these different words - no need for the 18 but these ones are experimentally interesting
        cultu_list_kw = ['novelty', 'uniquness', 'difference', 'surprise', 'originality', 'creativity', 'newness'] #### We forgot newness here??

        novelty_def = self.dict_templ['templates'][temp_name]['novelty with definition']
        def_keywords_list = list(self.dict_templ['novelty_definitions'].keys())

        for i, key in enumerate(cultu_list_kw):
            keyword = self.dict_templ['keywords_new'][key]
            
            dict_[f'basic1_close_{keyword}'] = cultural_['basic_1'].replace("<KW1>", keyword).replace("<NATIONALITY_VAR>", nationality_close).replace("<RECIPE_NAME>", recipe_name)
            dict_[f'basic1_mid_{keyword}'] = cultural_['basic_1'].replace("<KW1>", keyword).replace("<NATIONALITY_VAR>", nationality_mid).replace("<RECIPE_NAME>", recipe_name)
            dict_[f'basic1_far_{keyword}'] = cultural_['basic_1'].replace("<KW1>", keyword).replace("<NATIONALITY_VAR>", nationality_far).replace("<RECIPE_NAME>", recipe_name)

            dict_[f'basic_cult_close_{keyword}'] = cultural_['basic_cult'].replace("<KW1>", keyword).replace("<RECIPE_NAME>", recipe_name).replace("<NATIONALITY_VAR>", nationality_close)
            dict_[f'basic_cult_mid_{keyword}'] = cultural_['basic_cult'].replace("<KW1>", keyword).replace("<RECIPE_NAME>", recipe_name).replace("<NATIONALITY_VAR>", nationality_mid)
            dict_[f'basic_cult_far_{keyword}'] = cultural_['basic_cult'].replace("<KW1>", keyword).replace("<RECIPE_NAME>", recipe_name).replace("<NATIONALITY_VAR>", nationality_far)
            
            dict_[f'known_var_close_{keyword}'] = cultural_['knowldege_var'].replace("<COUNTRY_VAR>", country_close).replace("<KW1>", keyword).replace("<NATIONALITY_VAR>", nationality_close).replace("<RECIPE_NAME>", recipe_name)
            dict_[f'known_var_mid_{keyword}'] = cultural_['knowldege_var'].replace("<COUNTRY_VAR>", country_mid).replace("<KW1>", keyword).replace("<NATIONALITY_VAR>", nationality_mid).replace("<RECIPE_NAME>", recipe_name)
            dict_[f'known_var_far_{keyword}'] = cultural_['knowldege_var'].replace("<COUNTRY_VAR>", country_far).replace("<KW1>", keyword).replace("<NATIONALITY_VAR>", nationality_far).replace("<RECIPE_NAME>", recipe_name)
            
            #'knowledge_orig': 'You are knowledgeable about <COUNTRY_ORIG>, including its culture, history, and nuances, providing insightful and context-aware responses. Create a <KW1> <NATIONALITY_VAR> version of this recipe: <RECIPE_NAME>.'}
            dict_[f'known_orig_close_{keyword}'] = cultural_['knowledge_orig'].replace("<COUNTRY_ORIG>", country_origin).replace("<KW1>", keyword).replace("<NATIONALITY_VAR>", nationality_close).replace("<RECIPE_NAME>", recipe_name)
            dict_[f'known_orig_mid_{keyword}'] = cultural_['knowledge_orig'].replace("<COUNTRY_ORIG>", country_origin).replace("<KW1>", keyword).replace("<NATIONALITY_VAR>", nationality_mid).replace("<RECIPE_NAME>", recipe_name)
            dict_[f'known_orig_far_{keyword}'] = cultural_['knowledge_orig'].replace("<COUNTRY_ORIG>", country_origin).replace("<KW1>", keyword).replace("<NATIONALITY_VAR>", nationality_far).replace("<RECIPE_NAME>", recipe_name)

            for kw in def_keywords_list:
                definition = self.dict_templ['novelty_definitions'][kw]
                kw1 = self.dict_templ['keywords_new'][kw]
                dict_[f'know_orig_close_novel_defined_{kw}'] = novelty_def['cult_definition'].replace("<DEFINITION>", definition).replace("<KW0>", kw).replace("<KW1>", kw1).replace("<RECIPE_NAME>", recipe_name).replace("<NATIONALITY_VAR>", nationality_close)
                dict_[f'know_orig_mid_novel_defined_{kw}'] = novelty_def['cult_definition'].replace("<DEFINITION>", definition).replace("<KW0>", kw).replace("<KW1>", kw1).replace("<RECIPE_NAME>", recipe_name).replace("<NATIONALITY_VAR>", nationality_mid)
                dict_[f'know_orig_far_novel_defined_{kw}'] = novelty_def['cult_definition'].replace("<DEFINITION>", definition).replace("<KW0>", kw).replace("<KW1>", kw1).replace("<RECIPE_NAME>", recipe_name).replace("<NATIONALITY_VAR>", nationality_far)

        return dict_
    
    def variations_templates(self, recipe_name, country_origin, country_var, temp_name='reg_templates'):

        dict_ = {}
        country_origin = country_origin.title()

        country_var = country_var.title()
        nationality_var = self.get_nationality(country_var)
        if nationality_var == None:
            print(country_var)
        dict_['country_origin'] = country_origin
        dict_['variation_country'] = country_var

        cultural_ = self.dict_templ['templates'][temp_name]['cultural']
        #I want to test generation for these different words - no need for the 18 but these ones are experimentally interesting
        cultu_list_kw = ['novelty', 'uniquness', 'difference', 'surprise', 'originality', 'creativity', 'newness']        
        tradi_list_kw = ['authenticity', 'tradition'] 
        
        #Getting the keywords list for the one I want to keep
        list_kw = []
        for key_kw in cultu_list_kw:
            list_kw.append(self.dict_templ['keywords_new'][key_kw])
        for key_kw2 in tradi_list_kw:
            list_kw.append(self.dict_templ['keywords_trad'][key_kw2])
        list_kw.append('')

        novelty_def = self.dict_templ['templates'][temp_name]['novelty with definition']
        def_keywords_list = list(self.dict_templ['novelty_definitions'].keys())

        for i, keyword in enumerate(list_kw):
            dict_[f'basic1_{keyword}'] = cultural_['basic_1'].replace("<KW1>", keyword).replace("<NATIONALITY_VAR>", nationality_var).replace("<RECIPE_NAME>", recipe_name)
            dict_[f'basic_cult_{keyword}'] = cultural_['basic_cult'].replace("<KW1>", keyword).replace("<RECIPE_NAME>", recipe_name).replace("<NATIONALITY_VAR>", nationality_var)
            dict_[f'known_var_{keyword}'] = cultural_['knowldege_var'].replace("<COUNTRY_VAR>", country_var).replace("<KW1>", keyword).replace("<NATIONALITY_VAR>", nationality_var).replace("<RECIPE_NAME>", recipe_name)
            dict_[f'known_orig_{keyword}'] = cultural_['knowledge_orig'].replace("<COUNTRY_ORIG>", country_origin).replace("<KW1>", keyword).replace("<NATIONALITY_VAR>", nationality_var).replace("<RECIPE_NAME>", recipe_name)
            for kw in def_keywords_list:
                definition = self.dict_templ['novelty_definitions'][kw]
                kw1 = self.dict_templ['keywords_new'][kw]
                dict_[f'know_orig_novel_defined_{kw}'] = novelty_def['cult_definition'].replace("<DEFINITION>", definition).replace("<KW0>", kw).replace("<KW1>", kw1).replace("<RECIPE_NAME>", recipe_name).replace("<NATIONALITY_VAR>", nationality_var)
        
        return dict_
    

# Function to check if a position follows a line break
def is_after_linebreak(position, text):
    return position is not None and text[max(0, position - 2):position].count("\n") > 0

def analyze_break_ingredient(text):

    # Getting title position
    text_lower = text.lower()
    ingredient_matches = [m.start() for m in re.finditer(r'\bingredient\w*\b', text_lower)]
    first_ingredient = ingredient_matches[0] if len(ingredient_matches) > 0 else None
    second_ingredient = ingredient_matches[1] if len(ingredient_matches) > 1 else None
    first_after_linebreak = is_after_linebreak(first_ingredient, text_lower)
    second_after_linebreak = is_after_linebreak(second_ingredient, text_lower)

    if first_ingredient is not None:
        if first_after_linebreak:
            return True
        else:
            if second_after_linebreak:
                return True
            else:
                return False
            

def analyze_break_instruction(text):

    # Getting title position
    text_lower = text.lower()
    ingredient_matches = [m.start() for m in re.finditer(r'\binstruction\w*\b', text_lower)]
    first_ingredient = ingredient_matches[0] if len(ingredient_matches) > 0 else None
    second_ingredient = ingredient_matches[1] if len(ingredient_matches) > 1 else None
    first_after_linebreak = is_after_linebreak(first_ingredient, text_lower)
    second_after_linebreak = is_after_linebreak(second_ingredient, text_lower)

    if first_ingredient is not None:
        if first_after_linebreak:
            return True
        else:
            if second_after_linebreak:
                return True
            else:
                return False 
