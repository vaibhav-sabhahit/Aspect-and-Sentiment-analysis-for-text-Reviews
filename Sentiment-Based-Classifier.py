# NLTK
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.corpus import words

#Spacy
import spacy
nlp = spacy.load("en_core_web_sm")


import numpy as np
import pandas as pd
import sklearn
import os
import sys
import re


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')


class Classifier:
    """The Classifier"""


    def normalize_accent(self, string):
        string = string.replace('á', 'a')
        string = string.replace('â', 'a')
        string = string.replace('é', 'e')
        string = string.replace('è', 'e')
        string = string.replace('ê', 'e')
        string = string.replace('ë', 'e')
        string = string.replace('î', 'i')
        string = string.replace('ï', 'i')
        string = string.replace('ö', 'o')
        string = string.replace('ô', 'o')
        string = string.replace('ò', 'o')
        string = string.replace('ó', 'o')
        string = string.replace('ù', 'u')
        string = string.replace('û', 'u')
        string = string.replace('ü', 'u')
        string = string.replace('ç', 'c')
        return string



    def raw_to_tokens(self, raw_string):
        string = raw_string.lower()
        string = self.normalize_accent(string)
        spacy_tokens = self.spacy_nlp(string)
        string_tokens = [token.orth_ for token in spacy_tokens if not token.is_punct] 
        clean_string = " ".join(string_tokens)
        clean_string = clean_string.replace("n't", " not")
        clean_string = clean_string.replace("'ve", " have")
        clean_string = clean_string.replace("'re", "  are")
        clean_string = clean_string.replace("it's", "it is")
        return clean_string



    def create_features(self, type='train'):
        if(type=='train'):
            self.train_df['wordcnt'] = self.train_df['review'].apply(lambda x: len(str(x).split()))
            # Number of capital letters
            self.train_df['capitalcnt'] = self.train_df['review'].apply(lambda x: len([c for c in str(x) if c.isupper()]))
            #Number of special characters
            self.train_df['specialcnt'] = self.train_df['review'].str.findall(r'[^a-zA-Z0-9 ]').str.len()
            #Number of unique words
            self.train_df['uniquecnt'] = self.train_df['review'].apply(lambda x: len(set(str(x).split())))
            #Numbers of digits
            self.train_df['digitcnt'] = self.train_df['review'].apply(lambda x: sum(c.isdigit() for c in x))
            #Number of characters
            self.train_df['charcnt'] = self.train_df['review'].apply(lambda x: len(str(x)))
            #Number of stopwords
            self.train_df['stopcnt'] = self.train_df['review'].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords.words('english')]))
            ######################### generating sentiment columns using nltk library #########################
            self.train_df['review_new'] = self.train_df['review'].apply(lambda x: self.raw_to_tokens(x))
            ######################### mapping sentiment to numbers ######################### 
            self.train_df['sentiment_new'] = self.train_df['sentiment'].map({'positive': 1, 'negative': 2, 'neutral': 3})
        else:
            self.test_df['wordcnt'] = self.test_df['review'].apply(lambda x: len(str(x).split()))
            # Number of capital letters
            self.test_df['capitalcnt'] = self.test_df['review'].apply(lambda x: len([c for c in str(x) if c.isupper()]))
            #Number of special characters
            self.test_df['specialcnt'] = self.test_df['review'].str.findall(r'[^a-zA-Z0-9 ]').str.len()
            #Number of unique words
            self.test_df['uniquecnt'] = self.test_df['review'].apply(lambda x: len(set(str(x).split())))
            #Numbers of digits
            self.test_df['digitcnt'] = self.test_df['review'].apply(lambda x: sum(c.isdigit() for c in x))
            #Number of characters
            self.test_df['charcnt'] = self.test_df['review'].apply(lambda x: len(str(x)))
            #Number of stopwords
            self.test_df['stopcnt'] = self.test_df['review'].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords.words('english')]))
            ######################### generating sentiment columns using nltk library #########################
            self.test_df['review_new'] = self.test_df['review'].apply(lambda x: self.raw_to_tokens(x))
            ######################### mapping sentiment to numbers ######################### 
            self.test_df['sentiment_new'] = self.test_df['sentiment'].map({'positive': 1, 'negative': 2, 'neutral': 3})
            



    def get_sentiment(self, type='train'): 
    # function used to get the polarity columns - positive, negative, neutral and compound
        sentiments = pd.DataFrame()
        s = StandardScaler()
        if(type=='train'): 
            self.sid = SentimentIntensityAnalyzer() # to get the sentiment using vader_lexicon
            for title in self.train_df.review_new:
                pol = self.sid.polarity_scores(title)   
                sentiments = sentiments.append(pol, ignore_index=True)
            strain = pd.DataFrame(s.fit_transform(sentiments),columns = sentiments.columns)
            #concatenate the 4 columns obtained to original dataframe
            self.train_df = pd.concat([self.train_df, strain], axis=1)
            #round off the 4 columns to 6 decimal points
            self.train_df['compound'] = self.train_df['compound'].apply(lambda x: round(x,6))
            self.train_df['neg'] = self.train_df['neg'].apply(lambda x: round(x,6))
            self.train_df['neu'] = self.train_df['neu'].apply(lambda x: round(x,6))
            self.train_df['pos'] = self.train_df['pos'].apply(lambda x: round(x,6))

        else:
            for title in self.test_df.review_new:
                pol = self.sid.polarity_scores(title)   
                sentiments = sentiments.append(pol, ignore_index=True)
            stest = pd.DataFrame(s.fit_transform(sentiments),columns = sentiments.columns)
            #concatenate the 4 columns obtained to original dataframe
            self.test_df = pd.concat([self.test_df, stest], axis=1)
            #round off the 4 columns to 6 decimal points
            self.test_df['compound'] = self.test_df['compound'].apply(lambda x: round(x,6))
            self.test_df['neg'] = self.test_df['neg'].apply(lambda x: round(x,6))
            self.test_df['neu'] = self.test_df['neu'].apply(lambda x: round(x,6))
            self.test_df['pos'] = self.test_df['pos'].apply(lambda x: round(x,6))



    def get_word_features(self, type='train'):
    # function to obtain the features - not, ok, so, just, bad - value is 1 if review_new column contains the word, 0 otherwise
        if(type=='train'):
            self.train_df['not'] = self.train_df['review_new'].apply(lambda x:  1 if 'not' in nltk.word_tokenize(str(x)) else 0)
            self.train_df['ok'] = self.train_df['review_new'].apply(lambda x:  1 if 'ok' in nltk.word_tokenize(str(x)) else 0)
            self.train_df['so'] = self.train_df['review_new'].apply(lambda x:  1 if 'so' in nltk.word_tokenize(str(x)) else 0)
            self.train_df['just'] = self.train_df['review_new'].apply(lambda x:  1 if 'just' in nltk.word_tokenize(str(x)) else 0)
            self.train_df['bad'] = self.train_df['review_new'].apply(lambda x:  1 if 'bad' in nltk.word_tokenize(str(x)) else 0)
        else:
            self.test_df['not'] = self.test_df['review_new'].apply(lambda x:  1 if 'not' in nltk.word_tokenize(str(x)) else 0)
            self.test_df['ok'] = self.test_df['review_new'].apply(lambda x:  1 if 'ok' in nltk.word_tokenize(str(x)) else 0)
            self.test_df['so'] = self.test_df['review_new'].apply(lambda x:  1 if 'so' in nltk.word_tokenize(str(x)) else 0)
            self.test_df['just'] = self.test_df['review_new'].apply(lambda x:  1 if 'just' in nltk.word_tokenize(str(x)) else 0)
            self.test_df['bad'] = self.test_df['review_new'].apply(lambda x:  1 if 'bad' in nltk.word_tokenize(str(x)) else 0)

    def get_tfidf(self, type):
        if(type=='train'):
            #appends all review_new strings to vec_train
            vec_train=[]
            for string in self.train_df['review_new']:
                vec_train.append(string)
            #creating tfidf vectorizer
            self.tfidf = TfidfVectorizer(ngram_range=(1, 1))
            #getting the tfidf features for train dataset
            self.train_tfidf = self.tfidf.fit_transform(vec_train)
        else:
            #appends all review_new strings to vec_test
            vec_test=[]
            for string in self.test_df['review_new']:
                vec_test.append(string)
            #getting the tfidf features for test dataset
            self.test_tfidf = self.tfidf.transform(vec_test)


    def get_input(self, type):
    # function to obtain the input array
        if(type=='train'):
            #converting sparse matrix containing tfidf features to array
            self.train_tfidf1 = self.train_tfidf.toarray()
            self.train_all = np.concatenate((self.train_df_arr, self.train_tfidf1), axis=1)
        else:
            #converting sparse matrix containing tfidf features to array
            self.test_tfidf1 = self.test_tfidf.toarray()
            self.test_all = np.concatenate((self.test_df_arr, self.test_tfidf1), axis=1)



    def get_output(self, type):
    # function to obtain the output variables
        if(type=='train'):
            self.train_y = self.train_df.sentiment_new.values
        else:
            self.test_y = self.test_df.sentiment_new.values

    
    def get_model(self):
        #function to fit the machine learning model on train data
        base_learners = [ ('lr', LogisticRegression(random_state = 20, max_iter = 2000)), ('svc', LinearSVC(random_state = 146))]
        self.clf = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression(random_state = 146))
        self.clf.fit(self.train_all, self.train_y)
        


    def get_predictions(self):
    # function to get predictions
        test_pred = list(self.clf.predict(self.test_all))
        self.output = []
        for x in test_pred:
            if(x==1):
                self.output.append('positive')
            elif(x==2):
                self.output.append('negative')
            else:
                self.output.append('neutral')



    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        self.spacy_nlp = spacy.load('en_core_web_sm')
        #read train file
        self.train_df = pd.read_csv(trainfile, sep='\t', header=None)
        #renaming the columns
        self.train_df.columns = ['sentiment', 'aspect', 'term', 'location', 'review']
        self.create_features('train')
        self.get_sentiment('train')
        self.get_word_features('train')
        self.get_tfidf('train') 
        #keeping only the columns required for fitting the model
        train_new = self.train_df[['wordcnt', 'capitalcnt', 'specialcnt', 'uniquecnt', 'digitcnt', 'charcnt', 'stopcnt', 'compound', 'neg', 'neu', 'pos', 'not', 'ok', 'so', 'bad', 'just']] #'compound1', 'neg1', 'neu1', 'pos1']]
        self.train_df_arr = train_new.to_numpy()
        self.get_input('train')
        self.get_output('train')



    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        #read test file
        self.test_df = pd.read_csv(datafile, sep='\t', header=None)
        #renaming the columns
        self.test_df.columns = ['sentiment', 'aspect', 'term', 'location', 'review']
        self.create_features('test')
        self.get_sentiment('test')
        self.get_word_features('test')
        self.get_tfidf('test') 
        #keeping only the columns required for calculating the results
        test_new = self.test_df[['wordcnt', 'capitalcnt', 'specialcnt', 'uniquecnt', 'digitcnt', 'charcnt', 'stopcnt', 'compound', 'neg', 'neu', 'pos', 'not', 'ok', 'so', 'bad', 'just']] #'compound1', 'neg1', 'neu1', 'pos1']]
        self.test_df_arr = test_new.to_numpy()
        self.get_input('test')
        self.get_output('test')
        self.get_model()
        self.get_predictions()
        return self.output # returns the ouput in list format