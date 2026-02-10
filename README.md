## Cultural LLMs for Cooking Recipes

**Paper:** Insert name and link here  
**Access to GlobalFusion Dataset:**  
https://drive.google.com/file/d/1kYUw1BIym8E55gmloYLDFlkTJm71VxKi/view?usp=sharing

## Repository Structure
### LLMs Recipe Generation
- **Gen_dataset.py**  
  Generating the templates for all recipes
- **data_utils.py**  
  Includes sampling methods by distance and classes for template generation
- **Gen_recipes_LLMs.py**  
  Main script
- **config_genLLMs.py**  
  Configurations for LLMs and paths
- **model_utils.py**  
  Model utility functions

### Combining LLM-Generated Recipes with Paired GlobalFusion Recipes
- **GlobalFusion_withLLMS_Generation.py**  
  Main script
- **LLM_Novelty_Utils.py**  
  Processing LLM-generated recipes to match the GlobalFusion format  
  (Title, Ingredients, Instructions)

### Measuring Novelty
- **measuring_novelty.py**  
  Main script
- **score_config.py**  
  Hyperparameters for novelty scoring
- **utils_embeddings.py**  
  Calling logit lens to apply metrics on internal model layers
- **Scoring.py**  
  Running metrics (Surprise, Newness, etc.)
- **utils.py**  
  Utility functions for novelty measurement

### Statistical Analyses
- **MeatStatistics.ipynb**  
  Ingredient, title, and country statistics from the paper
- **Correlations_CultDist.ipynb**  
  Correlation with cultural distances
- **Stats_utils.py**  
  Utilities for distance measurements
- **Correlations_HumanLLMs.ipynb**  
  Human vs LLMs t-tests and per-layer analyses
- **Correlations_NoveltyTypes.ipynb**  
  Results by novelty keyword types (Traditional vs Creative)

----
    
## Instructions for running the whole code: 
	1. Dowload GlobalFusion Dataset                                
	2. Run Gen_dataset.py to generate templates for all recipes
	3. Run Gen_recipes_LLMs.py to generate the recipes for a given LLM
	4. Run GlobalFusion_withLLMS_Generation.py to pair the generated LLMs in one file with GF           
	5. Run measuring_novelty.py to generate the novelty scores for a Given LLM
	6 Run the various scripts in Statistical Analyses to replicate analyses from the paper
