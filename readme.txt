ASPECT BASED SENTIMENT ANALYSIS


Prerequisites
Install below packages:
•	scipy
•	Pandas
•	Spacy(en_core_web_sm)
•	Numpy 
•	Sklearn
•	nltk(download vader_lexicon, stopwords, punkt)
•	re
•	warnings

Steps of the model
Documents -> Preprocessing -> Feature engineering -> Machine Learning model -> Calculate accuracy

************************************************************************************************************************************************************

PREPROCESSING

review_new - In this part, we created "raw_to_tokens" and "normalize_accent" function to clean review text. 
First we normalized accents then move on to removing the punctuations present in the text. 
We then correct some basic forms of text such as "I've" to "I have" in order to ensure all the wordsare captured properly. 
We then join back the tokenized values to a string which gives us the "review_new" column
(we tried several other preprocessing steps such as removing stopwords, stemming, lemmatsation, getting the adjectived & adverbs alone by pos_tagging.  
However, this was reducing the accuracy and taking away the context of the wordsFor example, lemmatisation converts the word 'excellent' to 'excel')

************************************************************************************************************************************************************

FEATURE ENGINEERING

We created below features which went a long way in contributing towards the accuracy we finally obtained. 

1)Number of words - More words could imply a more critical explanation of what they did not like. Positive reviews were usually seen to have lesser words.
2)Number of capital letters - Excess of capital letters could imply a message in angry tone. 

The following features gives us a numeric representation of the properties of the text which we use to find correlation and patterns. 
3) Number of special characters
4) Number of unique words
5)Number of digits
6) Number of characters
7)Number of stopwords

The features 1-7 above were created after referring solutions to similar classification problems such as 'Quora Insincere questions classification' 
for example. These solutions showed that these features are extremely helpful in predicting the polarity. Also, they have high correlation with the
output column.

These features gives us the sentiment approximation as calculated by the nltk vader_lexicon sentimentintensityanalyzer. This calculates the polarity
based on the review_new column. We tried calculating the polarity of review column with different combinations of pre-processing steps, and kept the
one which gave the best correlation with the output column.
8) Polarity of Negative
9)  Polarity of Neutral
10) Polarity of Positive
11) Polarity of Compound

These are binary columns to help check if these words were present in the given review or not. These values were hardcoded after lot of
research. To get these features, we calculated the frequency of appearance of these words in positive, negative and neutral reviews seperately (using
nltk.FreqDist). Then, we generated columns for most frequent words. These columns will have a 1 if the review contains that particular word and 0 
otherwise. We chose the word columns to keep after analyzing the correlation matrix and the performance with multiple classifiers. Also, we confirmed
this by plotting the wordclouds and finding the words which are most frequently appearing as positive, negative or neutral words.
 
12)word "Not"
13)word "Ok"
14)word "So"
15)"Bad"
16)"just"

17) TFIDF vectorization
In the tfidf we tried several combinations bigram and unigram such as (1,1) , (1,2), (2,2) and found the optimal parameter to be the default values of (1,1). that is having unigram as both upper and lower bound. 
In order to convert the pre-processed review column(review_new) we fit transform the column using tidf vectorization.
We also perform the same vectorization process to the term column. The newly generated vectors are now used as features in the classification model. 
SVM classification
For the classification purpose we have used SVM classifier. After performing a hyper parameter grid search we found the optimal parameters to be XX.

************************************************************************************************************************************************************
Functions and purpose

1)normalize_accent(self, string)- Remove any accent present in the review column 
2)raw_to_tokens(self, raw_string)- Preprocess the text, normalizing the accent and removing punctiations. Also converts shorthands to full forms
3)create_features(self, type='train')- Create all the additional features that is mentioned in the feature engineering section
4)get_sentiment(self, type='train')- Generate the polarity approximations for the given text input
5)get_word_features(self, type='train')- Generate the binary column for presence/absense of words such as 'but' in the input text.
6)get_tfidf(self, type)- Generate the TFIDF vector for the input text
7)get_input(self, type)- Concatenate the features generated and the TFIDF vector 
8)get_model()- Return the classifier model 

***************************************************************************************************************************************
**********************************
ACCURACY of VARIOUS MODELS

Some other models we have tested
Model	Accuracy on dev data
Conv1D model with embedding layer and dense layers	77%
LSTM model with embedding layer and dense(3)	78%
Logistic regression	79%
Neural network with only dense layers	79%
Doc2Vec model followed by dense layers	 79%
LinearSVC model 82%
VotingClassifier 80%

LinearSVC is the model which gave us the best accuracy. However, Stacking Classifier is a more robust model. Since the final evaluation is on another 
dataset, we decided to keep Stacking Classifier as the best models. We performed hyperparameter tuning 


************************************************************************************************************************************************************
  
ACCURACY ON DEVDATA
Accuracy- On dev data
After training SVM classifier on the train data, we predict using devdata and get accuracy of 81.12%.


**********************************************************************************************************************************************************

