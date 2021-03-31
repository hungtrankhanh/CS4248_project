import pandas as pd

import vectorizer


path_prefix = "/Users/ansundaram/OneDrive - PayPal/Personal/nus/CS4248/project/"

data_path = path_prefix + "NUS-MOOC-Transacts-Corpus/data/nus-mooc-transacts-corpus-pswd-protected/Task1-Marking_Task/_organalysis-003.lecture.5.csv"

train_data = pd.read_csv(data_path)['Input.posts']


feature_vectors = vectorizer.get_feature_vectors(train_data, None, "tfidf")
print(feature_vectors[0].shape)
print(feature_vectors[0][0])

feature_vectors = vectorizer.get_feature_vectors(train_data, None, "word2vec")
print(feature_vectors[0].shape)
print(feature_vectors[0][0])