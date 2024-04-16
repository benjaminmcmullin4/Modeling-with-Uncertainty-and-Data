"""Volume 3: Naive Bayes Classifiers."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages into spam or ham.
    '''
    # __init__ for consistency
    def __init__(self):
        self.ham_probs = {}
        self.spam_probs = {}

    # Problem 1
    def fit(self, X, y):
        '''
        Compute the values P(C=Ham), P(C=Spam), and P(x_i|C) to fit the model.

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        # Extract all words from the input data
        words = X.str.split().explode()

        # Initialize dictionaries to store probabilities for each word in spam and ham
        self.spam_probs = {w: 0 for w in words}
        self.ham_probs = {w: 0 for w in words}

        # Split the data into spam and ham classes
        is_spam = X[y == 'spam']
        is_ham = X[y == 'ham']

        # Calculate the probabilities of spam and ham
        self.prob_spam = len(is_spam) / (len(is_spam) + len(is_ham))
        self.prob_ham = len(is_ham) / (len(is_spam) + len(is_ham))

        # Get the total number of words in spam and ham
        self.spam_size = len(is_spam.str.split().explode())
        self.ham_size = len(is_ham.str.split().explode())

        # Count the occurrences of each word in spam and ham
        word_count_spam = is_spam.str.split().explode().value_counts()
        word_count_ham = is_ham.str.split().explode().value_counts()

        # Calculate probabilities for each word using Laplace add-one smoothing
        for w in words:

            # If the word is in the spam or ham counts, use the formula. Else, use the formula with 0.
            if w in word_count_spam.index:
                self.spam_probs[w] = (word_count_spam.loc[w] + 1) / (self.spam_size + 2)
            else:
                self.spam_probs[w] = 1 / (self.spam_size + 2)
            
            if w in word_count_ham.index:
                self.ham_probs[w] = (word_count_ham.loc[w] + 1) / (self.ham_size + 2)
            else:
                self.ham_probs[w] = 1 / (self.ham_size + 2)

        # Return the trained NaiveBayesFilter model
        return self


    # Problem 2
    def predict_proba(self, X):
        '''
        Find ln(P(C=k,x)) for each x in X and for each class.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam.
                Column 0 is ham, column 1 is spam.
        '''
        # Initialize the array to store log probabilities for each class (ham and spam)
        log_prob_array  = np.zeros((len(X), 2))
        
        # Iterate over each message in the input data
        for i, row in enumerate(X):

            # Split the message into individual words
            row = row.split()
            
            # Set placeholders for the log probabilities
            ham_placeholder = 0
            spam_placeholder = 0
            
           # Calculate the log probabilities for each word in the message
            for w in row:

               # Use try/except block to handle the case where a word is not present in the trained model
                try:

                    # Update log probabilities based on the presence of the word in ham and spam
                    ham_placeholder += np.log(self.ham_probs[w]) if w in self.ham_probs else np.log(1/2)
                    spam_placeholder += np.log(self.spam_probs[w]) if w in self.spam_probs else np.log(1/2)
                
                except RuntimeWarning:
                    print(f'Runtime warning for {w}')
            
             # Add the final log probabilities to the array
            log_prob_array [i][0] = ham_placeholder + np.log(self.prob_ham)
            log_prob_array [i][1] = spam_placeholder + np.log(self.prob_spam)
            
        # Return the computed log probabilities array
        return log_prob_array 

    # Problem 3
    def predict(self, X):
        '''
        Predict the labels of each row in X, using self.predict_proba().
        The label will be a string that is either 'spam' or 'ham'.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # Predict the class labels based on the argmax of log probabilities from predict_proba
        # If the argmax is 0, label as 'ham', otherwise label as 'spam'. Store the result in an array with dtype object.
        return np.array(['ham' if x == 0 else 'spam' for x in np.argmax(self.predict_proba(X), axis = 1)], dtype = object)


def prob4():
    """
    Create a train-test split and use it to train a NaiveBayesFilter.
    Predict the labels of the test set.
    
    Compute and return the following two values as a tuple:
     - What proportion of the spam messages in the test set were correctly identified by the classifier?
     - What proportion of the ham messages were incorrectly identified?
    """
    df = pd.read_csv('sms_spam_collection.csv')
    X = df.Message
    y = df.Label
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Initialize and train the NaiveBayesFilter classifier
    NB = NaiveBayesFilter()
    NB.fit(X_train, y_train)

    # Make predictions on the test set
    predict = NB.predict(X_test)

    # Calculate the proportion of correctly classified spam messages
    proportion_spam = np.sum(predict[y_test == 'spam'] == 'spam') / np.sum(y_test == 'spam')

    # Calculate the proportion of ham messages incorrectly identified as spam
    proportion_ham = np.sum(predict[y_test == 'ham'] == 'spam') / np.sum(y_test == 'ham')

    # Return the proportions
    return proportion_spam, proportion_ham


# Problem 5
class PoissonBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    This classifier assumes that words are distributed like
    Poisson random variables.
    '''
    # __init__ for consistency
    def __init__(self):
        self.ham_rates = {}
        self.spam_rates = {}

    def fit(self, X, y):
        '''
        Compute the values P(C=Ham), P(C=Spam), and r_{i,k} to fit the model.

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        # Extract all unique words from the input data
        words = X.str.split().explode().unique()

        # Initialize dictionaries to store probabilities for each word in spam and ham
        self.spam_rates = {w: 0 for w in words}
        self.ham_rates = {w: 0 for w in words}

        # Split the data into spam and ham classes
        is_spam = X[y == 'spam']
        is_ham = X[y == 'ham']

        # Calculate the probabilities of spam and ham
        self.prob_spam = len(is_spam) / (len(is_spam) + len(is_ham))
        self.prob_ham = len(is_ham) / (len(is_spam) + len(is_ham))

        # Get the total number of words in spam and ham
        self.spam_size = len(is_spam.str.split().explode())
        self.ham_size = len(is_ham.str.split().explode())

        # Count the occurrences of each word in spam and ham
        word_count_spam = is_spam.str.split().explode().value_counts()
        word_count_ham = is_ham.str.split().explode().value_counts()

        # Save value counts for later use in prediction
        self.spam_value_counts = word_count_spam
        self.ham_value_counts = word_count_ham

        # Calculate probabilities for each word using Laplace add-one smoothing
        for w in words:
            # If the word is in the spam or ham counts, use the formula. Else, use the formula with 0.
            if w in word_count_spam.index:
                self.spam_rates[w] = (word_count_spam.loc[w] + 1)/(self.spam_size + 2)
            else:
                self.spam_rates[w] = 1 / (self.spam_size + 2)
            
            if w in word_count_ham.index:
                self.ham_rates[w] = (word_count_ham.loc[w] + 1)/(self.ham_size + 2)
            else:
                self.ham_rates[w] = 1 / (self.ham_size + 2)

        # Return the trained NaiveBayesFilter model
        return self


    def predict_proba(self, X):
        '''
        Find ln(P(C=k,x)) for each x in X and for each class.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam.
                Column 0 is ham, column 1 is spam.
        '''
        # Initialize the output array
        output = np.zeros((len(X), 2))

        # Iterate through each message
        for i, row in enumerate(X):
            # Split the row
            row = row.split()
            message_words, message_word_counts = np.unique(row, return_counts=True)
            
            # Set placeholders for probabilities
            ham_place = np.log(self.prob_ham)
            spam_place = np.log(self.prob_spam)
            
            # Get the probabilities of each value
            for word in message_words:
                # Get the index of the word in the message
                word_count_in_message = message_word_counts[np.where(message_words == word)[0][0]]
                
                # Update the probabilities
                try:
                    ham_place += stats.poisson.logpmf(word_count_in_message, self.ham_rates[word] * len(row)) if word in self.ham_rates else \
                                                                                                            stats.poisson.logpmf(word_count_in_message, 1/(self.ham_size + 2) * len(row))
                    spam_place += stats.poisson.logpmf(word_count_in_message, self.spam_rates[word] * len(row)) if word in self.spam_rates else \
                                                                                                            stats.poisson.logpmf(word_count_in_message, 1/(self.spam_size + 2) * len(row))
                # Handle the case where the word is not in the trained model
                except RuntimeWarning:
                    print(f'Runtime warning for {word}')
            
            # Add the probabilities to the output array 
            output[i][0] = ham_place
            output[i][1] = spam_place

        # Return the output array
        return output

    def predict(self, X):
        '''
        Predict the labels of each row in X, using self.predict_proba().
        The label will be a string that is either 'spam' or 'ham'.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # Predict the class labels based on the argmax of log probabilities from predict_proba
        return np.array(['ham' if x==0 else 'spam' for x in np.argmax(self.predict_proba(X), axis=1)], dtype=object)

def prob6():
    """
    Create a train-test split and use it to train a PoissonBayesFilter.
    Predict the labels of the test set.
    
    Compute and return the following two values as a tuple:
     - What proportion of the spam messages in the test set were correctly identified by the classifier?
     - What proportion of the ham messages were incorrectly identified?
    """
    df = pd.read_csv('sms_spam_collection.csv')
    X = df.Message
    y = df.Label
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Initialize and train the Poisson Bayes classifier
    PB = PoissonBayesFilter()
    PB.fit(X_train, y_train)

    # Test the performance of the classifier
    predict = PB.predict(X_test)

    # Calculate the proportion of correctly classified spam messages
    proportion_spam = np.sum(predict[y_test == 'spam'] == 'spam') / np.sum(y_test == 'spam')

    # Calculate the proportion of ham messages incorrectly identified as spam
    proportion_ham = np.sum(predict[y_test == 'ham'] == 'spam') / np.sum(y_test == 'ham')

    return proportion_spam, proportion_ham

    
# Problem 7
def sklearn_naive_bayes(X_train, y_train, X_test):
    '''
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    '''
    # Create a CountVectorizer to convert the training data into a document-term matrix (DTM)
    vectorizer = CountVectorizer()
    train_counts = vectorizer.fit_transform(X_train)

    # Initialize and fit a Multinomial Naive Bayes classifier using the transformed training data
    clf = MultinomialNB()
    clf = clf.fit(train_counts, y_train)

    # Transform the test data using the same CountVectorizer
    test_counts = vectorizer.transform(X_test)

    # Use the trained classifier to predict labels for the test data
    labels = clf.predict(test_counts)

    # Return the predicted labels for the test data
    return labels



if __name__ == "__main__":
    # file_path = 'sms_spam_collection.csv'

    # df = pd.read_csv(file_path)
    # X = df.Message
    # y = df.Label

    # NB = NaiveBayesFilter()
    # NB.fit(X,y)
    # print(NB.predict_proba(X[800:805]))
    # print(NB.predict(X[800:805]))
    # print(NB.ham_probs['out'])
    # print(NB.spam_probs['out'])

    # print(prob4())

    # PB = PoissonBayesFilter()
    # PB.fit(X[:300], y[:300])
    # print(PB.predict_proba(X[800:805]))
    # print(PB.predict(X[800:805]))
    # print(PB.ham_rates['in'])
    # print(PB.spam_rates['in'])

    # print(prob6())

    # print(sklearn_naive_bayes(X[:300], y[:300], X[800:805]))

    pass