# CS4248_project - Group 04 : Instructor Intervention Prediction In MOOC

## 1. Environment setup

- python version = 3.6.12

- pip install -r requirements.txt

## 2. Dataset
Training corpora are from Task1-Marking_Task in NUS-MOOC-Transacts-Corpus

## 3. Run program
Preprocessing and feature engineering over the corpora takes long time (nearly 2 hours), so we save feature engineering vectors into files, then we only need to load these files for next studying such as : hyper-parameter tuning, ablation testing, model perfomance verifying ...
### 3.1 tf_idf feature: 
#### a. load raw files from dataset and process feature engineering : 
	- python main.py --feature tf_idf --dataset unprocessed
#### b. load feature engineering dataset (processed in step a):
	- python main.py --feature tf_idf --dataset processed

### 3.2 word2vec feature: 
#### a. load raw files from dataset and process feature engineering :
	- python main.py --feature word2vec --dataset unprocessed
#### b. load feature engineering dataset (processed in step a):
	- python main.py --feature word2vec --dataset processed

## 4. Model performance
### 4.1 tf_idf feature: 
	- f1 score on test data = 0.88
	- accuracy score on test data = 0.81
	- precision score on test data = 0.85
	- recall score on test data = 0.92
	- f2 score on test data = 0.91

### 4.2 word2vec feature: 
	- f1 score on test data = 0.88
	- accuracy score on test data = 0.82
	- precision score on test data = 0.85
	- recall score on test data = 0.92
	- f2 score on test data = 0.90

