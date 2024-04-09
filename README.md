# AdaMedGraph
Official code for AdaMedGraph and Personalized Progression Modelling and Prediction in Parkinson's Disease with a Novel Multi-Modal Graph Approach 
## This project is divided into four main parts:

### Data Processing:

Description: This folder encompasses all the necessary code for processing the PPMI and PDBP data.
Steps:
Initially, we processed the motor, non-motor, demographic, and ledd data, adapting code from https://github.com/kseverso/Discovery-of-PD-States-using-ML
Next, the processed modalities were combined. PD and HC were selected using "data_merge_hc_pd_separation.ipynb".
Feature engineering was performed using either "data_processing_simple.ipynb" or "data_processing_imputation_hyper.ipynb".

### Model:

Description: This folder contains our AdaMedGraph model. Additionally, "ada_med_graph_pdbp.py" is included to handle multi-dataset situations. You can train and tune the models using "train.py" or define your own training code.

### ML Comparison:

Description: This section includes our comparison models. You have the flexibility to create and train your own models.
Usage: Feel free to develop your code and train your models within this section.

### Analysis:

Description: This part contains our sub-analysis code.
Usage: Utilize the provided code for sub-analyzing the data as per your requirements.
