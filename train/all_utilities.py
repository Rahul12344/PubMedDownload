import numpy as np
import pandas as pd
import logging
import argparse
import shap
from shap import DeepExplainer
import seaborn as sn
from sklearn.metrics import f1_score
from collections import defaultdict
import math
import statistics

import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import StratifiedKFold
import nltk
from nltk.corpus import stopwords 

from keras import models
from keras import initializers
from keras import regularizers
#from keras.optimizers import adam_v2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold


from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from tensorflow import keras
import keras_tuner as kt

import os
import random
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.compat.v1.keras.backend import get_session 
from sklearn.metrics import ConfusionMatrixDisplay


shap.initjs()
#tf.compat.v1.disable_v2_behavior()

stop_words = set(stopwords.words('english'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NGRAM_RANGE = (1,1)#(2, 3)

TOP_K = 20000

TOKEN_MODE = 'word'

MIN_DOCUMENT_FREQUENCY = 2

SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 28

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class HyperModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def mlp_model(self, hp):
        op_units, op_activation = 1, 'sigmoid'
        model = models.Sequential()
        
        dropout_rate = hp.Choice('dropout_rate', values=[0.4, 0.5, 0.6, 0.7])
        
        model.add(Dropout(rate=dropout_rate, input_shape=self.input_shape))
        
        layers = hp.Int('layers', 2, 6, 1, default=2)
        units=hp.Int('units', 8, 64, 4, default=32)
        activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu')
        
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        for _ in range(layers-1):
            model.add(Dense(units=units, activation=activation))
            model.add(Dropout(rate=dropout_rate))
            model.add(Flatten())

        model.add(Dense(units=units, activation=op_activation))
        
        # Compile model with learning parameters.
        loss = 'binary_crossentropy'
        #optimizer = adam_v2.Adam(learning_rate=learning_rate)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
        
        return model
        

def load_bacteria_data(data_path):
    path_to_data = os.path.join(data_path, 'Garud')
    texts = []
    abstracts = []
    path = os.path.join(path_to_data, 'bacteria')
    for fname in sorted(os.listdir(path)):
        if fname.endswith('.txt'):
            with open(os.path.join(path, fname)) as f:
                filtered_sentence = []
                lower_cased_ = f.read().lower()
                lower_cased = ''.join([i for i in lower_cased_ if not i.isdigit()])
                for w in lower_cased.split(' '): 
                    if w not in stop_words: 
                        filtered_sentence.append(w) 
                texts.append(' '.join(filtered_sentence))
                abstracts.append(fname)
    indices = list(range(0, len(texts)))
    
    abstracts_ = []
    texts_ = []
    for i in range(len(texts)):
        abstracts_.append(abstracts[i])
        texts_.append(texts[i])
    return abstracts_, texts_

def load_vip_data(data_path):
    path_to_data = os.path.join(data_path, 'Garud')
    texts = []
    abstracts = []
    path = os.path.join(path_to_data, 'train/dataset/new_vips')
    for fname in sorted(os.listdir(path)):
        if fname.endswith('.txt'):
            with open(os.path.join(path, fname)) as f:
                filtered_sentence = []
                lower_cased_ = f.read().lower()
                lower_cased = ''.join([i for i in lower_cased_ if not i.isdigit()])
                for w in lower_cased.split(' '): 
                    if w not in stop_words: 
                        filtered_sentence.append(w) 
                texts.append(' '.join(filtered_sentence))
                abstracts.append(fname)
    indices = list(range(0, len(texts)))
    
    abstracts_ = []
    texts_ = []
    for i in range(len(texts)):
        abstracts_.append(abstracts[i])
        texts_.append(texts[i])
    return abstracts_, texts_

def write_data(data_path, classes, test_abstracts, iteration_num):
    file_path = open("outputs/updated_bacteria_predictionT_{}.txt".format(iteration_num), "w")
    for i in range(len(test_abstracts)):
        file_path.write("%s %f\n" % (test_abstracts[i], classes[i]))
    file_path.close()       
    
def write_data_vip(data_path, classes, test_abstracts, iteration_num):
    file_path = open("outputs/vip_prediction_{}.txt".format(iteration_num), "w")
    for i in range(len(test_abstracts)):
        file_path.write("%s %f\n" % (test_abstracts[i], classes[i]))
    file_path.close()      


def load_data(data_path, positive_percentage=0.3, seed=123244, dataset_size=450, training_size=0.6, testing_size=0.2, validation_size=0.2, data="VIP"):
    training_pos_size = training_size * float(dataset_size) * positive_percentage
    training_neg_size = training_size * float(dataset_size) * (1-positive_percentage)
    
    testing_pos_size = testing_size * float(dataset_size) * positive_percentage
    testing_neg_size = testing_size * float(dataset_size) * (1-positive_percentage)
    
    validation_pos_size = validation_size * float(dataset_size) * positive_percentage
    validation_neg_size = validation_size * float(dataset_size) * (1-positive_percentage)
    
    path_to_data = os.path.join(data_path, 'Garud/train/dataset/')
    
    training_data = "training"
    validation_data = "validation"
    testing_data = "testing"
    
    if data == "malaria":
        training_data = "malaria_training"
        validation_data = "malaria_validation"
        testing_data = "malaria_testing"
    if data == "bacteria":
        training_data = "bacteria_train"
        validation_data = "bacteria_validation"
        testing_data = "bacteria_test"
    if data == "update_bacteria":
        training_data = "updated_bacteria_train"
        validation_data = "updated_bacteria_validation"
        testing_data = "updated_bacteria_test"
    
    #path_to_data = os.path.join(data_path, 'train')
    train_texts = []
    train_labels = []
    training_pos = len(os.listdir(os.path.join(path_to_data, training_data, "pos")))
    for category in ['pos', 'neg']:
        train_path = os.path.join(path_to_data, training_data, category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    filtered_sentence = []
                    #lower_cased = f.read().lower()
                    lower_cased_ = f.read().lower()
                    lower_cased = ''.join([i for i in lower_cased_ if not i.isdigit()])
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    train_texts.append(' '.join(filtered_sentence))
                train_labels.append(0 if category == 'neg' else 1)
                
    val_texts = []
    val_labels = []
    validation_pos = len(os.listdir(os.path.join(path_to_data, validation_data, "pos")))
    for category in ['pos', 'neg']:
        val_path = os.path.join(path_to_data, validation_data, category)
        for fname in sorted(os.listdir(val_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(val_path, fname)) as f:
                    filtered_sentence = []
                    #lower_cased = f.read().lower()
                    lower_cased_ = f.read().lower()
                    lower_cased = ''.join([i for i in lower_cased_ if not i.isdigit()])
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    val_texts.append(' '.join(filtered_sentence))
                val_labels.append(0 if category == 'neg' else 1)
                
    test_texts = []
    test_labels = []
    test_abstracts = []
    testing_pos = len(os.listdir(os.path.join(path_to_data, testing_data, "pos")))
    for category in ['pos', 'neg']:
        test_path = os.path.join(path_to_data, testing_data, category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    filtered_sentence = []
                    lower_cased_ = f.read().lower()
                    lower_cased = ''.join([i for i in lower_cased_ if not i.isdigit()])
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    test_texts.append(' '.join(filtered_sentence))
                test_labels.append(0 if category == 'neg' else 1)
                test_abstracts.append(fname)
                
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)
    
    num_pos = 0
    num_neg = 0 
    train_texts_ = []
    train_labels_ = []
    val_texts_ = []
    val_labels_ = []
    test_texts_ = []
    test_labels_ = []
    test_abstracts_ = []
    
    for i in range(len(train_labels)):
        if train_labels[i] == 1 and num_pos < math.floor(training_pos_size):
            train_texts_.append(train_texts[i])
            train_labels_.append(train_labels[i])
            num_pos += 1
        elif train_labels[i] == 0 and num_neg < math.ceil(training_neg_size):
            train_texts_.append(train_texts[i])
            train_labels_.append(train_labels[i])
            num_neg += 1
    
    num_pos = 0
    num_neg = 0 
    for i in range(len(val_labels)):
        if val_labels[i] == 1 and num_pos < math.floor(validation_pos_size):
            val_texts_.append(val_texts[i])
            val_labels_.append(val_labels[i])
            num_pos += 1
        elif val_labels[i] == 0 and num_neg < math.ceil(validation_neg_size):
            val_texts_.append(val_texts[i])
            val_labels_.append(val_labels[i])
            num_neg += 1
    
    num_pos = 0
    num_neg = 0    
    for i in range(len(test_labels)):
        if test_labels[i] == 1 and num_pos < math.floor(testing_pos_size):
            test_texts_.append(test_texts[i])
            test_labels_.append(test_labels[i])
            test_abstracts_.append(test_abstracts[i])
            num_pos += 1
        elif test_labels[i] == 0 and num_neg < math.ceil(testing_neg_size):
            test_texts_.append(test_texts[i])
            test_labels_.append(test_labels[i])
            num_neg += 1

    return ((train_texts_, np.array(train_labels_)),
            (val_texts_, np.array(val_labels_)),
            (test_texts_, np.array(test_labels_)))
    
def load_every_data(data_path, seed=123244):
    path_to_data = os.path.join(data_path, 'Garud/train/dataset')
    #path_to_data = os.path.join(data_path, 'train')
    train_texts = []
    train_labels = []
    training_pos = len(os.listdir(os.path.join(path_to_data, 'training', "pos")))
    for category in ['pos', 'neg']:
        train_path = os.path.join(path_to_data, 'training', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    filtered_sentence = []
                    lower_cased = f.read().lower()
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    train_texts.append(' '.join(filtered_sentence))
                train_labels.append(0 if category == 'neg' else 1)
    
    training_pos = len(os.listdir(os.path.join(path_to_data, 'malaria_training', "pos")))
    for category in ['pos', 'neg']:
        train_path = os.path.join(path_to_data, 'malaria_training', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    filtered_sentence = []
                    lower_cased = f.read().lower()
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    train_texts.append(' '.join(filtered_sentence))
                train_labels.append(0 if category == 'neg' else 1)
                
    val_texts = []
    val_labels = []
    validation_pos = len(os.listdir(os.path.join(path_to_data, 'validation', "pos")))
    for category in ['pos', 'neg']:
        val_path = os.path.join(path_to_data, 'validation', category)
        for fname in sorted(os.listdir(val_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(val_path, fname)) as f:
                    filtered_sentence = []
                    lower_cased = f.read().lower()
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    val_texts.append(' '.join(filtered_sentence))
                val_labels.append(0 if category == 'neg' else 1)
    
    validation_pos = len(os.listdir(os.path.join(path_to_data, 'malaria_validation', "pos")))
    for category in ['pos', 'neg']:
        val_path = os.path.join(path_to_data, 'malaria_validation', category)
        for fname in sorted(os.listdir(val_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(val_path, fname)) as f:
                    filtered_sentence = []
                    lower_cased = f.read().lower()
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    val_texts.append(' '.join(filtered_sentence))
                val_labels.append(0 if category == 'neg' else 1)
                
    test_texts = []
    test_labels = []
    test_abstracts = []
    testing_pos = len(os.listdir(os.path.join(path_to_data, 'testing', "pos")))
    for category in ['pos', 'neg']:
        test_path = os.path.join(path_to_data, 'testing', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    filtered_sentence = []
                    lower_cased = f.read().lower()
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    test_texts.append(' '.join(filtered_sentence))
                test_labels.append(0 if category == 'neg' else 1)
                test_abstracts.append(fname)
                
    testing_pos = len(os.listdir(os.path.join(path_to_data, 'malaria_testing', "pos")))
    for category in ['pos', 'neg']:
        test_path = os.path.join(path_to_data, 'malaria_testing', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    filtered_sentence = []
                    lower_cased = f.read().lower()
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    test_texts.append(' '.join(filtered_sentence))
                test_labels.append(0 if category == 'neg' else 1)
                test_abstracts.append(fname)
                
    temp = list(zip(train_texts, train_labels))
    random.seed(seed)
    random.shuffle(temp)
    train_texts, train_labels = zip(*temp)  
    

    return ((train_texts, np.array(train_labels)),
            (val_texts, np.array(val_labels)),
            (test_texts, np.array(test_labels)))
    
def load_all_data(data_path, seed=123):
    path_to_data = os.path.join(data_path, 'Garud/train/dataset')
    train_texts = []
    train_labels = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(path_to_data, 'unified_malaria_train', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    train_texts.append(f.read())
                train_labels.append(0 if category == 'neg' else 1)
    test_texts = []
    test_labels = []  
    test_abstracts = []  
    for category in ['pos', 'neg']:
        val_path = os.path.join(path_to_data, 'unified_malaria_test', category)
        for fname in sorted(os.listdir(val_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(val_path, fname)) as f:
                    test_texts.append(f.read())
                test_labels.append(0 if category == 'neg' else 1)
                test_abstracts.append(fname)
    val_texts = []
    val_labels = []          
    for category in ['pos', 'neg']:
        test_path = os.path.join(path_to_data, 'unified_malaria_validation', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    val_texts.append(f.read())
                val_labels.append(0 if category == 'neg' else 1)
         
    temp = list(zip(train_texts, train_labels))
    random.seed(seed)
    random.shuffle(temp)
    train_texts, train_labels = zip(*temp)       

    train_texts_ = []
    train_labels_ = []
    val_texts_ = []
    val_labels_ = []
    test_texts_ = []
    test_labels_ = []
    test_abstracts_ = []
    for i in range(len(train_labels)):
        if train_labels[i] == 1:
            train_texts_.append(train_texts[i])
            train_labels_.append(train_labels[i])
        elif train_labels[i] == 0:
            train_texts_.append(train_texts[i])
            train_labels_.append(train_labels[i])
    
    for i in range(len(val_labels)):
        val_texts_.append(val_texts[i])
        val_labels_.append(val_labels[i])
            
    for i in range(len(test_labels)):
        test_abstracts_.append(test_abstracts[i])
        test_texts_.append(test_texts[i])
        test_labels_.append(test_labels[i])

    return (test_abstracts_, (train_texts_, np.array(train_labels_)),
            (val_texts_, np.array(val_labels_)),
            (test_texts_, np.array(test_labels_)))
    

def get_num_words_per_sample(sample_texts):
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)

def plot_sample_length_distribution(sample_texts):
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()
    
def plot_frequency_distribution_of_ngrams(sample_texts,
                                          ngram_range=(4, 6),
                                          num_ngrams=50):
    kwargs = {
            'ngram_range': (1, 1),
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',  # Split text into word tokens.
    }
    vectorizer = CountVectorizer(**kwargs)

    vectorized_texts = vectorizer.fit_transform(sample_texts)

    all_ngrams = list(vectorizer.get_feature_names())
    num_ngrams = min(num_ngrams, len(all_ngrams))

    all_counts = vectorized_texts.sum(axis=0).tolist()[0]

    all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
        zip(all_counts, all_ngrams), reverse=True)])
    ngrams = list(all_ngrams)[:num_ngrams]
    counts = list(all_counts)[:num_ngrams]

    idx = np.arange(num_ngrams)
    plt.bar(idx, counts, width=0.8, color='b')
    plt.xlabel('N-grams')
    plt.ylabel('Frequencies')
    plt.title('Frequency distribution of n-grams')
    plt.xticks(idx, ngrams, rotation=45)
    plt.show()

def ngram_vectorize(train_texts, train_labels, val_texts, ngram_range=NGRAM_RANGE):
    kwargs = {
            'ngram_range': NGRAM_RANGE,
            'dtype': 'int32',
            'stop_words': "english",
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE, 
            'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    x_train = vectorizer.fit_transform(train_texts)

    x_val = vectorizer.transform(val_texts)

    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32').todense()
    x_val = selector.transform(x_val).astype('float32').todense()
    return x_train, x_val, vectorizer.get_feature_names(), vectorizer

def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.

    # Arguments
        num_classes: int, number of classes.

    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation

def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model

def hyperparam_tuning(data, ngram_range=NGRAM_RANGE):
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = data

    # Vectorize texts.
    x_train, x_val, name_train, vectorizer = ngram_vectorize(
        train_texts, train_labels, val_texts, ngram_range)
    
    x_train, x_test, name_test, vectorizer = ngram_vectorize(
        train_texts, train_labels, test_texts, ngram_range)
    
    input_shape = (x_train.shape[1:],)
    hypermodel = HyperModel(input_shape)
    
    y_train = np.asarray(train_labels).astype('float32').reshape((-1, 1))
    y_test = np.asarray(test_labels).astype('float32').reshape((-1, 1))
    
    tuner_bo = kt.BayesianOptimization(
            hypermodel.mlp_model,
            objective='mse',
            max_trials=10,
            seed=42,
            executions_per_trial=2
        )
    tuner_bo.search(x_train, y_train, epochs=215, validation_split=0.2, verbose=0)
    best_model = tuner_bo.get_best_models(num_models=1)[0]
    best_model.evaluate(x_test, y_test)

def train_ngram_model(data,
                      learning_rate=1e-4,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.4,
                      ngram_range=NGRAM_RANGE):
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = data

    # Verify that validation labels are in the same range as training labels.
    num_classes = 2#explore_data.get_num_classes(train_labels)
    unexpected_labels = [v for v in val_labels if v not in range(2)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))

    # Vectorize texts.
    x_train, x_val, name_train, vectorizer = ngram_vectorize(
        train_texts, train_labels, val_texts, ngram_range)
    
    x_train, x_test, name_test, vectorizer = ngram_vectorize(
        train_texts, train_labels, test_texts, ngram_range)

    
    # Create model instance.
    model = mlp_model(layers=layers,
                                  units=units,
                                  dropout_rate=dropout_rate,
                                  input_shape=x_train.shape[1:],
                                  num_classes=num_classes)

    # Compile model with learning parameters.
    loss = 'binary_crossentropy'
    #optimizer = adam_v2.Adam(learning_rate=learning_rate)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    #callbacks = [tf.keras.callbacks.EarlyStopping(
        #monitor='val_loss', patience=2)]

    # Train and validate model.
    classifier = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            #callbacks=callbacks,
            #validation_split=0.2,
            validation_data=(x_val, val_labels),
            verbose=2, 
            batch_size=batch_size)

    # Print results.
    history = classifier.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    #model.save('VIP_mlp_model.h5')
    
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, test_labels, batch_size=batch_size)
    print('test loss, test acc:', results)
    
    return model, history, history['val_acc'][-1], history['val_loss'][-1], results[1], results[0], classifier

def create_mal_bac_diff(bact_file, mal_file):
    bact_file = open(bact_file)
    mal_file = open(mal_file)
    diff_file = open("outputs/diff_file.txt", "w")
    
    bact_file_lines = bact_file.readlines()
    mal_file_lines = mal_file.readlines()
    
    for i in range(0, len(bact_file_lines)):
        b_line = bact_file_lines[i].split()
        print(float(b_line[1]))
        m_line = mal_file_lines[i].split()
        diff_file.write("{0},{1},{2}\n".format(b_line[0], round(float(b_line[1])), round(float(m_line[1]))))
    
    bact_file.close()
    mal_file.close()
    diff_file.close()
    
def create_diff(rahul_file, anshu_file):
    r_file = open(rahul_file)
    a_file = open(anshu_file)
    diff_file = open("outputs/diff.txt", "w")
    
    r_file_lines = r_file.readlines()
    a_file_lines = a_file.readlines()
    
    for i in range(1, len(r_file_lines)):
        r_line = r_file_lines[i].rstrip().split(',')
        a_line = a_file_lines[i].rstrip().split(',')
        diff_file.write("{0},{1},{2},{3},{4}\n".format(r_line[0], r_line[1], a_line[0], r_line[2], a_line[1]))
    
    r_file.close()
    a_file.close()
    diff_file.close()
    
def create_unified(data_path, positive_percentage=1.0, seed=123, dataset_size=1000):
    pos_size = float(dataset_size) * positive_percentage
    neg_size = dataset_size-pos_size
    
    path_to_data = os.path.join(data_path, 'Garud')
    train_texts = []
    train_labels = []
    training_pos = len(os.listdir(os.path.join(path_to_data, 'malaria_training', "pos")))
    for category in ['pos', 'neg']:
        train_path = os.path.join(path_to_data, 'malaria_training', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    filtered_sentence = []
                    lower_cased = f.read().lower()
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    sentence = ' '.join(filtered_sentence)
                    f = open("/Users/rahulnatarajan/Research/Garud/train/unified_malaria_train/{}/{}".format(category,fname), "w")
                    f.write(sentence)
                    f.close()
                
    val_texts = []
    val_labels = []
    validation_pos = len(os.listdir(os.path.join(path_to_data, 'malaria_validation', "pos")))
    for category in ['pos', 'neg']:
        val_path = os.path.join(path_to_data, 'malaria_validation', category)
        for fname in sorted(os.listdir(val_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(val_path, fname)) as f:
                    filtered_sentence = []
                    lower_cased = f.read().lower()
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    sentence = ' '.join(filtered_sentence)
                    f = open("/Users/rahulnatarajan/Research/Garud/train/unified_malaria_validation/{}/{}".format(category,fname), "w")
                    f.write(sentence)
                    f.close()
                
                
    test_texts = []
    test_labels = []
    testing_pos = len(os.listdir(os.path.join(path_to_data, 'malaria_testing', "pos")))
    for category in ['pos', 'neg']:
        test_path = os.path.join(path_to_data, 'malaria_testing', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    filtered_sentence = []
                    lower_cased = f.read().lower()
                    for w in lower_cased.split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    sentence = ' '.join(filtered_sentence)
                    f = open("/Users/rahulnatarajan/Research/Garud/train/unified_malaria_test/{}/{}".format(category,fname), "w")
                    f.write(sentence)
                    f.close()

def overlap_comparison():
    f1 = open("/Users/rahulnatarajan/Research/Garud/train/outputs/bacttestclassifcation.txt", "r")
    f2 = open("/Users/rahulnatarajan/Research/Garud/train/outputs/bacttestclassifcation_malaria.txt", "r")
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    file_path = open("unity.txt", "w")
    for i in range(len(lines1)):
        line1 = lines1[i].split()
        line2 = lines2[i].split()
        if line1[1] == '1.000000' and line2[1] == '1.000000':
            file_path.write("{}\n".format(line1[0]))
    file_path.close()
    return

def split_into_kfolds(input_x, input_y, k):
    size = len(input_x)
    k_fold_size = math.floor(size * float(1)/k)
    training_x_sets = []
    training_y_sets = []
    validation_x_sets = []
    validation_y_sets = []
    for i in range(k):
        num_in_k = 0
        training_x_sets.append(list())
        training_y_sets.append(list())
        validation_x_sets.append(list())
        validation_y_sets.append(list())
        while num_in_k < k_fold_size:
            position = random.randint(0, size-1)
            item_already_placed = False
            for j in range(0, i+1):
                if input_x[position] in validation_x_sets[j]:
                    item_already_placed = True
            if not item_already_placed:
                validation_x_sets[i].append(input_x[position])
                validation_y_sets[i].append(input_y[position])
                num_in_k += 1
        for index in range(size):
            if input_x[index] not in validation_x_sets[i]:
                training_x_sets[i].append(input_x[index])
                training_y_sets[i].append(input_y[index])
    
    training_y_sets_fixed = list()
    validation_y_sets_fixed = list()
    for train_y in training_y_sets:
        training_y_sets_fixed.append(np.array(train_y))
    for val_y in validation_y_sets:
        validation_y_sets_fixed.append(np.array(val_y))
    
    return training_x_sets, training_y_sets_fixed, validation_x_sets, validation_y_sets_fixed

def comp_predictions_for_each_abstract():
    classifications = defaultdict(list)
    for i in range(10):
        file_path = open("outputs/updated_bacteria_prediction_{}.txt".format(i), "r")
        lines = file_path.readlines()
        for line in lines:
            prediction = line.split()
            classifications[prediction[0]].append(prediction[1])
        file_path.close()   
    return classifications

def comp_predictions_for_each_abstract_2():
    classifications = defaultdict(list)
    for i in range(10):
        file_path = open("outputs/vip_prediction_{}.txt".format(i), "r")
        lines = file_path.readlines()
        for line in lines:
            prediction = line.split()
            classifications[prediction[0]].append(prediction[1])
        file_path.close()   
    return classifications

def diff_predictions(classifications):
    file_path = open("outputs/updated_bacteria_prediction_diffT.csv", "w")
    file_path.write("abstract,1,2,3,4,5,6,7,8,9,10\n")
    for key in classifications.keys():
        file_path.write("{},{}\n".format(key, ','.join(classifications[key])))
        
def diff_predictions_2(classifications):
    file_path = open("outputs/vip_pred_diff.csv", "w")
    file_path.write("abstract,1,2,3,4,5,6,7,8,9,10\n")
    for key in classifications.keys():
        file_path.write("{},{}\n".format(key, ','.join(classifications[key])))      

def create_comparison(data="VIP", shap_enabled=False, box_plot=False, confusion_matrix=False, cycles=10, comp=True, dataset_size=1500, comparison_type="VIP"):
    parsers = argparse.ArgumentParser(description='Training handler')
    parsers.add_argument('--pos_size', type=float, default=1.0, help='Percentage of positives to be included in training/validation/testing')
    training_size = 0.6
    testing_size = 0.2
    validation_size = 0.2
    
    args = parsers.parse_args()

    logger.info('Logging results for number of epochs: {0}'.format(215))
        
    box_plot_dict = defaultdict(list)    
    for iterations in range(0,cycles,1): 
        seed=123244
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data("/Users/rahulnatarajan/Research/", positive_percentage=0.3, seed=seed, dataset_size=dataset_size, training_size=training_size,testing_size=testing_size,validation_size=validation_size,data=data)
        print("training on ", len(X_train), " validating on ", len(X_val))
        model, history, acc, loss, res_acc, res_loss, classifier = train_ngram_model(((X_train, y_train), (X_val, y_val), (X_test, y_test)), epochs=215, ngram_range=NGRAM_RANGE)

        x_train, x_test, names, vectorizer = ngram_vectorize(
                    X_train, y_train, X_test, ngram_range=NGRAM_RANGE)
        y_pred_keras = model.predict(x_test).ravel()

        X_train_ = np.array(x_train)
        tf.convert_to_tensor(X_train_.flatten())
        background = X_train_#[np.random.choice(X_train_.shape[0], 500, replace=False)]
        e = shap.DeepExplainer(model, background)
        
        X_test_ = np.array(x_test)
        X_labels = np.array(names)
        tf.convert_to_tensor(X_test_.flatten())
        tf.convert_to_tensor(X_labels.flatten())
        
        shap_values = e.shap_values(background)
        feature_order = np.flip(np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)))

        shap_val_sum = np.flip(np.sort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)))
        
        if box_plot:
            for i in range(0,len(shap_val_sum)):
                if X_labels[feature_order[i]] not in box_plot_dict:
                    box_plot_dict[X_labels[feature_order[i]]] = []
                box_plot_dict[X_labels[feature_order[i]]].append(shap_val_sum[i])
            
        seed +=1
        if shap_enabled:
            shap.summary_plot(shap_values, background, X_labels)
        

        if confusion_matrix:
            cf_matrix = confusion_matrix(y_test, y_pred_keras.round(), normalize=None)
            
            group_names = ['True Neg','False Pos','False Neg','True Pos']
            group_counts = ["{0:0.0f}".format(value) for value in
                            cf_matrix.flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in
                                cf_matrix.flatten()/np.sum(cf_matrix)]
            labels = ["{}\n{}\n{}".format(v1,v2,v3) for v1, v2, v3 in
                    zip(group_names,group_counts,group_percentages)]
            labels = np.asarray(labels).reshape(2,2)

            sn.heatmap(cf_matrix, annot=labels, fmt='')
                    
            plt.show()
            
            f1 = f1_score(y_test, y_pred_keras.round(), average='macro')
            print(f1)
        
        
        if comp:
            transformer = TfidfTransformer()
            loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vectorizer.vocabulary_)
            print("loaded")
            
            abstracts, data = [], []
            if comparison_type == "VIP": 
                abstracts, data = load_vip_data('/Users/rahulnatarajan/Research')
                print("loaded")
            elif comparison_type == "bacteria": 
                abstracts, data = load_bacteria_data('/Users/rahulnatarajan/Research')  
                print("loaded")
            testing = [1] * len(data)
            texts = transformer.fit_transform(loaded_vec.fit_transform(np.array(data)))
            texts = texts.astype('float32').todense()

            test_vals = np.squeeze(np.asarray(texts))
            predictions = model.predict(texts).ravel()
            predictions = np.round(predictions, 0)
            if comparison_type == "VIP": 
                write_data_vip('/Users/rahulnatarajan/Research', predictions, abstracts, iterations)
            elif comparison_type == "bacteria": 
                write_data('/Users/rahulnatarajan/Research', predictions, abstracts, iterations)
        seed+=1
    
    if comp: 
        if comparison_type == "VIP":       
            classifications = comp_predictions_for_each_abstract_2()
            diff_predictions_2(classifications)
        elif comparison_type == "bacteria": 
            classifications = comp_predictions_for_each_abstract()
            diff_predictions(classifications)
    
    data = []
    keys = box_plot_dict.keys()
    if box_plot: 
        #print(box_plot_dict)    
        median_sorted = sorted(box_plot_dict.items(), key=lambda k: statistics.median(k[1]), reverse=True)
        #print(median_sorted) 
        #print(len(median_sorted))   
        #for key in keys:
        #    data.append(box_plot_dict[key])
            
        keys = [item[0] for item in median_sorted][0:10]
        data = [item[1] for item in median_sorted][0:10]
            
        fig, ax = plt.subplots() 
        ax.set_xlabel('Feature Name')
        ax.set_ylabel('SHAP Value')     
        bp = ax.boxplot(data, patch_artist=True)
        for i in range(len(data)):
            x = np.random.normal(i, 0.04, size=len(data[i]))
            ax.scatter([a+1 for a in x], data[i], color='orange', zorder=100)

        plt.xticks(list(range(1,len(data)+1)), keys)
        plt.xticks(rotation=90)
        
        #colors = ['red', 'blue', 'green', 'cyan', 'orange', 'purple', 'pink', 'olive', 'brown', 'white', 'olive', 'lawngreen', 'indigo', 'skyblue', 'bisque']
        i = 0
        for box in bp['boxes']:
            box.set(color='black', linewidth=2)
            #box.set(facecolor = colors[i] )
            box.set(facecolor = 'olive' )
            #box.set(hatch = '/')
            i+=1
        
        plt.show()
        
    return
    
def hyperparam_tuning(data="VIP", intermediate_cycles=5, cycles=10, min_ngram_size=1, max_ngram_size=5, min_learning_rate=1e-8, max_learning_rate=1e-2, min_layers=1, max_layers=4, min_dropout_rate=0.2, max_dropout_rate=0.5, min_units=8, max_units=64):
    parsers = argparse.ArgumentParser(description='Training handler')
    parsers.add_argument('--pos_size', type=float, default=1.0, help='Percentage of positives to be included in training/validation/testing')
    training_size = 0.8
    testing_size = 0.2
    validation_size = 0.0
    
    seed=123244
    
    args = parsers.parse_args()

    logger.info('Logging results for number of epochs: {0}'.format(215))
        
    num_folds = 6
    
    (X_train_, y_train_), (X_val_, y_val_), (X_test, y_test) = load_data("/Users/rahulnatarajan/Research/", positive_percentage=0.3, seed=seed, dataset_size=3000, training_size=training_size,testing_size=testing_size,validation_size=validation_size,data=data)
    X_train_list, y_train_list_, X_val_list, y_val_list = split_into_kfolds(X_train_, y_train_, num_folds)  
    
    ngram_sizes = list(range(min_ngram_size, max_ngram_size+1))
    print(ngram_sizes)
    learning_rates = list()
    curr_learn_rate = min_learning_rate
    while curr_learn_rate <= max_learning_rate:
        learning_rates.append(curr_learn_rate)
        curr_learn_rate *= 10
    print(learning_rates)
    layers = list(range(min_layers, max_layers+1))
    print(layers)
    dropout_rates = list()
    curr_dropout_rate = min_dropout_rate
    while curr_dropout_rate <= max_dropout_rate:
        dropout_rates.append(curr_dropout_rate)
        curr_dropout_rate += 0.1
    print(dropout_rates)    
    units_per_layer = list(range(min_units, max_units+1, 8))
    print(units_per_layer)
    
    n_gram_dict = list()
    
    markerfacecolors = ['grey', 'red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
    markershapes = ['.','o','v','^','>','<','s','p','*','h','H','D','d','1']
    markeredgecolors = ['grey', 'red', 'blue', 'green', 'black', 'cyan', 'magenta']
    
    itera = 0
    for ngram_size in ngram_sizes: 
        curr_n_gram_data = list()
        for learning_rate in learning_rates:
            for layer in layers:
                for dropout_rate in dropout_rates:
                    for unit in units_per_layer:
                        curr_auc = list()
            
                        for (X_train, y_train, X_val, y_val) in zip(X_train_list, y_train_list_, X_val_list, y_val_list):
                            #(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data("/Users/rahulnatarajan/Research/", positive_percentage=0.3, seed=seed, dataset_size=1500, training_size=training_size,testing_size=testing_size,validation_size=validation_size,data=data)
                            curr_n_gram = (ngram_size, ngram_size)
                            print("training on ", len(X_train), " samples, validating on ", len(X_val))
                            print(itera)

                            model, history, acc, loss, r_acc, r_loss, classifier = train_ngram_model(((X_train, y_train), (X_val, y_val), (X_test, y_test)), epochs=215, ngram_range=curr_n_gram, learning_rate=learning_rate, layers=layer, units=unit, dropout_rate=dropout_rate)
                            x_train, x_test, names, vectorizer = ngram_vectorize(
                                    X_train, y_train, X_test, ngram_range=curr_n_gram)
                            y_pred_keras = model.predict(x_test).ravel()
                            
                            """train_curr_acc.append(history['acc'][-1])
                            val_curr_acc.append(history['val_acc'][-1])
                            res_curr_acc.append(r_acc)
                                
                            train_curr_loss.append(history['loss'][-1])
                            val_curr_loss.append(history['val_loss'][-1])
                            res_curr_loss.append(r_loss)"""
                            
                            fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
                            auc_keras = auc(fpr_keras, tpr_keras)
                            curr_auc.append(auc_keras)
                            itera += 1
                            
                        auc_ave = float(sum(curr_auc))/len(curr_auc)
                        curr_n_gram_data.append(auc_ave)
        n_gram_dict.append(curr_n_gram_data)
        
    fig, ax = plt.subplots() 
    ax.set_xlabel('N-Gram Size')
    ax.set_ylabel('AUC (Area Under Curve) Value')     
    
    unit_count = 0
    dropout_count = 0
    layer_count = 0
    learning_count = 0
    
    for i in range(len(n_gram_dict)):
        x = np.random.normal(i, 0.04, size=len(n_gram_dict[i]))
        ax.scatter([a+1 for a in x], n_gram_dict[i], color=markerfacecolors[unit_count], edgecolors=markeredgecolors[dropout_count], marker=markershapes[layer_count], size=6, zorder=100)
        unit_count += 1
        if unit_count > 8:
             unit_count = 0
             dropout_count += 1
        if dropout_count > 4:
            dropout_count = 0
            layer_count += 1
        if layer_count > 4:
            layer_count = 0
            learning_count += 1
        if learning_count > 7:
            learning_count = 0

    plt.xticks(list(range(1,6)), ['1-Gram', '2-Gram', '3-Gram', '4-Gram', '5-Gram'])
    plt.xticks(rotation=90)
    
    plt.show()
              
    return

def main():
    #create_comparison(cycles=10, comp=False, box_plot=True, data="update_bacteria", dataset_size=354, comparison_type="VIP") #354 6400
    hyperparam_tuning()
    return
    
if __name__ == '__main__':
    main()

