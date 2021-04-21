# CS4248_project

##Environment setup

- python version = 3.7

- pip install -r requirements.txt

# run program
## tf_idf feature: 
### a. load raw files from dataset and featuring : 
	- python main.py --feature tf_idf --dataset raw
### b. load featured dataset (processed in step a):
	- python main.py --feature tf_idf --dataset processed

## word2vec feature: 
### a. load raw files from dataset and featuring :
	- python main.py --feature word2vec --dataset raw
### b. load featured dataset (processed in step a):
	- python main.py --feature word2vec --dataset processed

