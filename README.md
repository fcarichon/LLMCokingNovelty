## Cultural LLMS for cooking recipes:
Paper : Insert name and link here
Access to GlobalFusion Dataset : https://drive.google.com/file/d/1kYUw1BIym8E55gmloYLDFlkTJm71VxKi/view?usp=sharing

## Detail of each file
LLMS Recipe Generation:
	Gen_dataset.py : Generating the templates for all recipes (Smaples by countries or distance)
	data_utils.py : Include sampling method by distances and class for template gen based on template instructions

	Gen_recipes_LLMs.py : Main   (to merge with Apertus configuration)
	config_genLLMs.py : Configurations for LLM & paths
	model_utils.py : function to get 

Combining LLMs Generated recipes with Paired GlobalFusion Recipes:
	GlobalFusion_withLLMS_Generation.py : Main
	LLM_Novelty_Utils.py : Pocessing LLMs Generated recipes to match GlobalFuion format {Title, Ingredient, Instructions}

Measure Novelty:
	measuring_novelty.py : Main
	score_config.py : Hyperparameters for scoring novelty for recipes
	utils_embeddings.py : Calling logit_lens toapply metrics on internal model layers
	Scoring.py : Running metrics (Surprise, Newness, etc.)
	utils.py : utils for measuring novelty

Statistical Analyses:
	MeatStatistics.ipynb : Measuring the ingredients, title, and country measures from the paper
	Correlations_CultDist.ipynb : Measuring correaltion with Cultural Distances
	Stats_utils.py : Utils for measuring distances
	Correlations_HumanLLMs.ipynb : Reporting results for human vs LLMs t-test + per layer analyses
	Correlations_NoveltyTypes.ipynb : Reporting results by keyword types (Traidtional vs creative ones)
    
## Instructions for running the whole code: 
	1. Dowload GlobalFusion Dataset                                
	2. Run Gen_dataset.py to generate templates for all recipes
	3. Run Gen_recipes_LLMs.py to generate the recipes for a given LLM
	4. Run GlobalFusion_withLLMS_Generation.py to pair the generated LLMs in one file with GF           
	5. Run measuring_novelty.py to generate the novelty scores for a Given LLM
	6 Run the various scripts in Statistical Analyses to replicate analyses from the paper
