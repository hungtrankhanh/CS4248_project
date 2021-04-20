import os
import pandas as pd
import re
import string
from textblob import TextBlob
import json
import nltk
from nltk.corpus import stopwords
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

from vectorizer import *
raw_file_train= 'datasets/train'
raw_file_test = 'datasets/test'

featured_file_word2vec = 'processed_datasets/word2vec/'
featured_file_tf_idf = 'processed_datasets/tf_idf/'


def load_datasets(args):
    if args.dataset == "raw":
        print("load dataset: load raw files")
        dir_path = raw_file_train

        list_of_dataframes = []
        for filename in os.listdir(dir_path):
            print("filename : ", filename)
            if ".csv" in filename:
                list_of_dataframes.append(pd.read_csv(os.path.join(dir_path, filename)))

        csv_data = pd.concat(list_of_dataframes)
        train_data = sub_threads(csv_data)

        dir_path = raw_file_test
        list_of_dataframes = []
        for filename in os.listdir(dir_path):
            print("filename : ", filename)
            if ".csv" in filename:
                list_of_dataframes.append(pd.read_csv(os.path.join(dir_path, filename)))

        csv_data = pd.concat(list_of_dataframes)
        test_data = sub_threads(csv_data)

        if args.feature == "word2vec":
            print("load_dataset: word2vec featuring")
            x_train, y_train_label, x_test, y_test_label = add_word2vec_vectors_to_features(train_data, test_data)
            save_processed_datasets(x_train, y_train_label, x_test, y_test_label, featured_file_word2vec)
        elif args.feature == "tf_idf":
            print("load_dataset: tf_idf featuring")
            x_train, y_train_label, x_test, y_test_label = add_tfidf_vectors_to_features(train_data, test_data)
            save_processed_datasets(x_train, y_train_label, x_test, y_test_label, featured_file_tf_idf)
    else:
        if args.feature == "word2vec":
            print("load_dataset: word2vec featuring")
            x_train, y_train_label, x_test, y_test_label = load_processed_datasets(featured_file_word2vec)
        elif args.feature == "tf_idf":
            print("load_dataset: tf_idf featuring")
            x_train, y_train_label, x_test, y_test_label = load_processed_datasets(featured_file_tf_idf)

    print("load_dataset : x_train = ", x_train.shape)
    print("load_dataset : y_train_label = ",y_train_label.shape)
    return x_train, y_train_label, x_test, y_test_label

def save_processed_datasets(X_train_data, y_train_label, X_test_data, y_test_label, path):
    X_data_len = len(X_train_data)
    N = X_data_len // 3
    train_data_list = X_train_data.tolist()
    with open(path + 'train_data_1.txt', 'w') as outfile:
        json.dump(train_data_list[0:N], outfile)
    with open(path + 'train_data_2.txt', 'w') as outfile:
        json.dump(train_data_list[N:2*N], outfile)
    with open(path + 'train_data_3.txt', 'w') as outfile:
        json.dump(train_data_list[2*N:X_data_len], outfile)

    train_label_list = y_train_label.tolist()
    with open(path + 'train_label.txt', 'w') as outfile:
        json.dump(train_label_list, outfile)

    test_data_list = X_test_data.tolist()
    with open(path + 'test_data.txt', 'w') as outfile:
        json.dump(test_data_list, outfile)

    test_label_list = y_test_label.tolist()
    with open(path + 'test_label.txt', 'w') as outfile:
        json.dump(test_label_list, outfile)

def load_processed_datasets(path):

    with open(path + 'train_data_1.txt') as json_file:
        data = json.load(json_file)
        X_train_data = np.array(data)
    with open(path + 'train_data_2.txt') as json_file:
        data = json.load(json_file)
        X_train_data = np.concatenate((X_train_data, np.array(data)), axis=0)
    with open(path + 'train_data_3.txt') as json_file:
        data = json.load(json_file)
        X_train_data = np.concatenate((X_train_data, np.array(data)), axis=0)

    with open(path + 'train_label.txt') as json_file:
        data = json.load(json_file)
        y_train_label = np.array(data)

    with open(path + 'test_data.txt') as json_file:
        data = json.load(json_file)
        X_test_data = np.array(data)

    with open(path + 'test_label.txt') as json_file:
        data = json.load(json_file)
        y_test_label = np.array(data)

    return X_train_data, y_train_label, X_test_data, y_test_label


def sub_threads(data_df):
    data_df2 = data_df[pd.notnull(data_df['Input.posts'])]  # remove empty posts
    data_df2['Input.posts'] = data_df2['Input.posts'].apply(
        lambda x: BeautifulSoup(x, features="html.parser").get_text())
    data_df2_list = data_df2.to_dict('records')

    data_df3_list = []
    for data_df2_dict in data_df2_list:
        posts = re.sub(r'POST[\s]+#[\d]+[\s]+by[\s]+User[\s]+#[\d]+', '____', data_df2_dict['Input.posts'])
        posts = re.split(r'____', posts)
        post_num = 1
        post_string = None
        for p in posts:
            tmp = p.strip()
            if len(tmp) > 0:
                item_dict = data_df2_dict.copy()
                last_upvotes = re.findall("Upvotes:[\s]+([-+]?\d+)", tmp)
                if post_string is None:
                    post_string = tmp
                else:
                    post_string = post_string + "____" + tmp

                item_dict['Input.posts'] = post_string
                ans_key = 'Answer.' + str(post_num)
                ans_val = 0

                if ans_key in item_dict:
                    ans_val = np.nan_to_num(item_dict[ans_key])
                item_dict["Label"] = 0

                if ans_val > 0:
                    item_dict["Label"] = 1
                item_dict["Num_Of_Posts"] = post_num

                if len(last_upvotes) > 0:
                    item_dict["Last_Upvotes"] = int(last_upvotes[0])
                else:
                    item_dict["Last_Upvotes"] = 0

                data_df3_list.append(item_dict)
                post_num += 1

    data_df3 = pd.DataFrame(data_df3_list)
    return data_df3


def lema_count(token, text):
    # token = token.casefold()
    return len(re.findall(r"{}".format(token), text)) / 100.0


def extract_percentage(text):
    if text == '':
        return 0
    tmp = re.split(r'%', text)
    rate = tmp[0].strip()
    return int(rate) / 100.0


def add_word2vec_vectors_to_features(x_train_df, x_test_df):
    # Word Vector Features
    train_wordvecs = np.vstack(x_train_df['Input.posts'].apply(lambda x: word2vec_avg(x)))
    test_wordvecs = np.vstack(x_test_df['Input.posts'].apply(lambda x: word2vec_avg(x)))


    train_features_df = engineer_features(x_train_df)
    test_features_df = engineer_features(x_test_df)

    train_result = train_features_df.to_numpy()
    test_result = test_features_df.to_numpy()

    print(train_result.shape)
    print(test_result.shape)
    print(train_wordvecs.shape)
    print(test_wordvecs.shape)

    train_result = np.concatenate((train_result, train_wordvecs), axis=1)
    test_result = np.concatenate((test_result, test_wordvecs), axis=1)
    print(train_result.shape)
    # result = preprocess(data=result)
    return train_result, x_train_df['Label'].to_numpy(), test_result, x_test_df['Label'].to_numpy()


def add_tfidf_vectors_to_features(x_train_df, x_test_df):
    # Word Vector Features
    train_tfidf, test_tfidf = get_tfidf_vectors(x_train_df['Input.posts'], x_test_df['Input.posts'])

    train_features_df = engineer_features(x_train_df)
    test_features_df = engineer_features(x_test_df)

    train_result = train_features_df.to_numpy()
    test_result = test_features_df.to_numpy()

    print(train_result.shape)
    print(test_result.shape)
    print(train_tfidf.shape)
    print(test_tfidf.shape)

    train_result = np.concatenate((train_result, train_tfidf), axis=1)
    test_result = np.concatenate((test_result, test_tfidf), axis=1)
    print(train_result.shape)
    # result = preprocess(data=result)
    return train_result, x_train_df['Label'].to_numpy(), test_result, x_test_df['Label'].to_numpy()


def engineer_features(x_train_df):
    """
    Engineer features of training data
    :param x_train: DataFrame of training data with column names following column names of input csv files
    :return: sparse matrix of engineered features
    """
    features_df = pd.DataFrame(index=x_train_df.index)

    # == Thread only features ==
    # 1. If thread was started by an anonymous user
    # TODO but not sure
    # 2. If thread was marked as approved or deleted
    features_df['f2a'] = x_train_df['ApprovalTime'].apply(lambda x: 1 if x != '' else 0)
    features_df['f2b'] = x_train_df['RejectionTime'].apply(lambda x: 1 if x != '' else 0)
    # 3. Forum ID
    # TODO but not sure
    # 4. Start time of thread
    # 5. Time of last post in thread
    # TODO
    # 6. Total number of posts in thread
    # TODO
    features_df['f6'] = x_train_df['Num_Of_Posts']
    # 7. If thread title contains the words "lecture[s]"
    features_df['f7'] = x_train_df['Input.threadtitle'].apply(lambda x: lema_count("lecture", x))
    # 8. If thread title contains the words "assignment[s]", "quiz[zes]", "grade[s]", "project[s]", "exam[s]"
    features_df['f8_1'] = x_train_df['Input.threadtitle'].apply(lambda x: lema_count("assignment", x))
    features_df['f8_2'] = x_train_df['Input.threadtitle'].apply(lambda x: lema_count("quiz", x))
    features_df['f8_3'] = x_train_df['Input.threadtitle'].apply(lambda x: lema_count("grade", x))
    features_df['f8_4'] = x_train_df['Input.threadtitle'].apply(lambda x: lema_count("project", x))
    features_df['f8_5'] = x_train_df['Input.threadtitle'].apply(lambda x: lema_count("exam", x))
    features_df['f8_6'] = x_train_df['Input.threadtitle'].apply(lambda x: lema_count("reading", x))
    features_df['f8_7'] = x_train_df['Input.threadtitle'].apply(lambda x: lema_count("exercise", x))

    # == Aggregated post features (presense of keywords) ==
    # 9. Total number of votes for each individual post
    # TODO
    features_df['f9_1'] = x_train_df['Last_Upvotes']
    # 10. Mean and variance of posting times of individual posts in thread
    # TODO
    # 11. Mean of time difference between between the posting times of individual posts and the closest course landmark. A course landmark is the deadline of an assignment, exam or project.
    # TODO
    features_df['f11_2'] = x_train_df['LifetimeApprovalRate'].apply(lambda x: extract_percentage(x))  # percentage
    features_df['f11_3'] = x_train_df['Last30DaysApprovalRate'].apply(lambda x: extract_percentage(x))  # percentage
    features_df['f11_4'] = x_train_df['Last7DaysApprovalRate'].apply(lambda x: extract_percentage(x))  # percentage
    # 12. Count of occurrences of assessment related words "grade[s]", "exam[s]", "assignment[s]", "quiz[zes]", "reading[s]", "project[s]"

    features_df['f12_1'] = x_train_df['Input.posts'].apply(lambda x: lema_count("assignment", x))
    features_df['f12_2'] = x_train_df['Input.posts'].apply(lambda x: lema_count("quiz", x))
    features_df['f12_3'] = x_train_df['Input.posts'].apply(lambda x: lema_count("grade", x))
    features_df['f12_4'] = x_train_df['Input.posts'].apply(lambda x: lema_count("project", x))
    features_df['f12_5'] = x_train_df['Input.posts'].apply(lambda x: lema_count("exam", x))
    features_df['f12_6'] = x_train_df['Input.posts'].apply(lambda x: lema_count("reading", x))
    features_df['f12_7'] = x_train_df['Input.posts'].apply(lambda x: lema_count("exercise", x))

    # 13. Count of occurences of words indicating technical problems "problem[s]", "error[s]"
    features_df['f13_1'] = x_train_df['Input.posts'].apply(lambda x: lema_count("problem", x))
    features_df['f13_2'] = x_train_df['Input.posts'].apply(lambda x: lema_count("error", x))
    features_df['f13_3'] = x_train_df['Input.posts'].apply(lambda x: lema_count("fail", x))

    # 14. Count of occurences of thread conclusive words "thank you", "thank[s]", "appreciate"
    features_df['f14_1'] = x_train_df['Input.posts'].apply(lambda x: lema_count("thank", x))
    features_df['f14_2'] = x_train_df['Input.posts'].apply(lambda x: lema_count("appreciate", x))

    # 15. Count of occurences of words like "request", "submit", "suggest"
    # list15 = ['request', 'submit', 'suggest']
    # features_df['f15'] = x_train_df['Input.posts'].apply(lambda x: sum([x.count(w) for w in list15]))

    features_df['f15_1'] = x_train_df['Input.posts'].apply(lambda x: lema_count("request", x))
    features_df['f15_2'] = x_train_df['Input.posts'].apply(lambda x: lema_count("submit", x))
    features_df['f15_3'] = x_train_df['Input.posts'].apply(lambda x: lema_count("suggest", x))

    # == POS Tags of thread title and posts (labelled a and b respectively) ==
    # 16. word count
    features_df['16a'] = x_train_df['Input.threadtitle'].apply(lambda x: len(x.split()) / 100.0)
    features_df['16b'] = x_train_df['Input.posts'].apply(lambda x: len(x.split()) / 100.0)
    # 17. character count
    features_df['17a'] = x_train_df['Input.threadtitle'].apply(len)
    features_df['17b'] = x_train_df['Input.posts'].apply(len)

    features_df['17a'] = features_df['17a'] / 100.0
    features_df['17b'] = features_df['17b'] / 100.0
    # 18. word density
    features_df['18a'] = features_df['17a'] / (features_df['16a'] + 1)
    features_df['18b'] = features_df['17b'] / (features_df['16b'] + 1)
    # 19. punctuation count
    features_df['19a'] = x_train_df['Input.threadtitle'].apply(
        lambda x: len("".join(p for p in x if p in string.punctuation)) / 10.0)
    features_df['19b'] = x_train_df['Input.posts'].apply(
        lambda x: len("".join(p for p in x if p in string.punctuation)) / 10.0)
    # 20. stopword count
    stop_words = list(set(stopwords.words('english')))
    features_df['20a'] = x_train_df['Input.threadtitle'].apply(
        lambda x: len([s for s in x.split() if x.lower in stop_words]) / 10.0)
    features_df['20b'] = x_train_df['Input.posts'].apply(
        lambda x: len([s for s in x.split() if x.lower in stop_words]) / 10.0)
    # 21. text polarity
    features_df['21a'] = x_train_df['Input.threadtitle'].apply(lambda x: TextBlob(x).sentiment.polarity)
    features_df['21b'] = x_train_df['Input.posts'].apply(lambda x: TextBlob(x).sentiment.polarity)
    # 22. text subjectivity
    features_df['22a'] = x_train_df['Input.threadtitle'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    features_df['22b'] = x_train_df['Input.posts'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    # 23. noun count
    features_df['23a'] = x_train_df['Input.threadtitle'].apply(
        lambda x: count_partofspeech_tag(text=x, tag='noun') / 10.0)
    features_df['23b'] = x_train_df['Input.posts'].apply(lambda x: count_partofspeech_tag(text=x, tag='noun') / 10.0)
    # 24. verb count
    features_df['24a'] = x_train_df['Input.threadtitle'].apply(
        lambda x: count_partofspeech_tag(text=x, tag='verb') / 10.0)
    features_df['24b'] = x_train_df['Input.posts'].apply(lambda x: count_partofspeech_tag(text=x, tag='verb') / 10.0)
    # 25. adjective count
    features_df['25a'] = x_train_df['Input.threadtitle'].apply(
        lambda x: count_partofspeech_tag(text=x, tag='adj') / 10.0)
    features_df['25b'] = x_train_df['Input.posts'].apply(lambda x: count_partofspeech_tag(text=x, tag='adj') / 10.0)
    # 26. adverb count
    features_df['26a'] = x_train_df['Input.threadtitle'].apply(
        lambda x: count_partofspeech_tag(text=x, tag='adv') / 10.0)
    features_df['26b'] = x_train_df['Input.posts'].apply(lambda x: count_partofspeech_tag(text=x, tag='adv') / 10.0)
    # 27. pronoun count
    features_df['27a'] = x_train_df['Input.threadtitle'].apply(
        lambda x: count_partofspeech_tag(text=x, tag='pronoun') / 10.0)
    features_df['27b'] = x_train_df['Input.posts'].apply(lambda x: count_partofspeech_tag(text=x, tag='pronoun') / 10.0)

    # == Post Emission Features == TODO but not sure
    # 28. (pi, hi) = count of occurrences of question words or question marks in pi if the state is hi; 0 otherwise.
    # 29. (pi, hi) = count of occurrences of thank words (thank you or thanks) in pi if the state is hi; 0 otherwise.
    # 30. (pi, hi) = count of occurrences of greeting words (e.g. hi, hello, good morning, welcome etc ) in pi if the state is hi; 0 otherwise.
    # 31. (pi, hi) = count of occurrences of assessment related words (e.g. grade, exam, assignment, quiz, reading, project etc.) in pi if the state is hi; 0 otherwise.
    # 32. (p, ihi) = count of occurrences of request, submit or suggest in pi if the state is hi; 0 otherwise.
    # 33. (pi, hi) = log(course duration/t(pi)) if the state is hi; 0 otherwise. Here t(pi) is the difference between the posting time of pi and the closest course landmark (assignment or project deadline or exam).
    # 34. (pi, pi−1, hi) = difference between posting times of pi and pi−1 normalized by course duration if the state is hi; 0 otherwise.

    # == Transition Features == TODO but not sure
    # 35. (hi−1, hi) = 1 if previous state is hi−1 and current state is hi; 0 otherwise.
    # 36. (hi−1, hi, pi, pi−1) = cosine similarity between pi−1 and pi if previous state is hi−1 and current state is hi; 0 otherwise.
    # 37. (hi−1, hi, pi, pi−1) = length of pi if previous state is hi−1, pi−1 has non-zero question words and current state is hi; 0 otherwise.
    # 38. (hn, r) = 1 if last post’s state is hn and intervention decision is r; 0 otherwise.
    # 39. (hn, r, pn) = 1 if last post’s state is hn, pn has non-zero question words and intervention decision is r; 0 otherwise.
    # 40. (hn, r, pn) = log(course duration/t(pn)) if last post’s state is hn and intervention decision is r; 0 otherwise. Here t(pn) is the difference between the posting time of pn and the closest course landmark (assignment or project deadline or exam).

    # result = sp.sparse.csr_matrix(features_df.values)
    return features_df


def count_partofspeech_tag(text, tag):
    pos_dict = {'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
                'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
                'adj':  ['JJ', 'JJR', 'JJS'],
                'adv': ['RB', 'RBR', 'RBS', 'WRB'],
                'pronoun': ['PRP', 'PRP$', 'WP', 'WP$']}
    count = 0
    for x in TextBlob(text).tags:
        if list(x)[1] in pos_dict[tag]:
            count += 1
    return count