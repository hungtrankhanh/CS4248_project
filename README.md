# CS4248_project

## Environment setup

- python version = 3.7

- pip install -r requirements.txt

## Dataset
Training corpora are from Task1-Marking_Task in NUS-MOOC-Transacts-Corpus

## Run program
Preprocessing and feature engineering over the corpora takes long time (nearly 2 hours), so we save feature engineering vectors into files, then we only need to load these files for next studying such as : hyper-parameter tuning, ablation testing, model perfomance verifying ...
### tf_idf feature: 
#### a. load raw files from dataset and process feature engineering : 
	- python main.py --feature tf_idf --dataset unprocessed
#### b. load feature engineering dataset (processed in step a):
	- python main.py --feature tf_idf --dataset processed

### word2vec feature: 
#### a. load raw files from dataset and process feature engineering :
	- python main.py --feature word2vec --dataset unprocessed
#### b. load feature engineering dataset (processed in step a):
	- python main.py --feature word2vec --dataset processed

## Model performance
### tf_idf feature: 
	- f1 score on validation = 0.88
	- accuracy score on validation = 0.81
	- precision score on validation = 0.85
	- recall score on validation = 0.92
	- f2 score on validation = 0.91
### word2vec feature: 
	- f1 score on validation = 0.88
	- accuracy score on validation = 0.82
	- precision score on validation = 0.85
	- recall score on validation = 0.92
	- f2 score on validation = 0.90
