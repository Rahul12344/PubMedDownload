import numpy as np
import pandas as pd
import logging
import argparse
from sklearn.metrics import f1_score
from collections import defaultdict
import math
import statistics
from collections import defaultdict

import logging

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

import pickle

#tf.compat.v1.disable_v2_behavior()

stop_words = set(stopwords.words('english'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NGRAM_RANGE = (1,1)

TOP_K = 20000

TOKEN_MODE = 'word'

MIN_DOCUMENT_FREQUENCY = 2

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

class HyperModel:
    def __init__(self, args, path, data_path, output_path):
        self.n_gram_range = args['ngram']
        self.learning_rate_range = args['learning_rate']
        self.dropout_rate_range = args['dropout_rate']
        self.layer_ranges = args['layer']
        self.unit_range = args['unit']
        self.path = path
        
        data_file = open(data_path, "rb")
        data_file.seek(0)
        self.data = pickle.load(data_file)
        print(self.data)
        data_file.close()
        
        self.output_path = output_path
    
    def run_model(self, curr_ngram_size, curr_learning_rate, curr_dropout_rate, curr_layer_count, curr_unit_per_layer):
        curr_n_gram = (ngram_size, ngram_size)
        
        folded_data = self.data['train_validation']
        test_data = self.data['test']
        
        logger.info('Averaging the output of 5-fold training on n-gram {}, learning_rate {}, dropout_rate {}, layer_count {}, unit_per_layer {}'.format(curr_ngram_size, curr_learning_rate, curr_dropout_rate, curr_layer_count, curr_unit_per_layer))
        
        total_auc = 0.0
        total_f1 = 0.0
        
        
        for (X_train, y_train, X_val, y_val) in folded_data:
            
            model, history, acc, loss, r_acc, r_loss, classifier = train_ngram_model(((X_train, y_train), (X_val, y_val), (X_test, y_test)), epochs=215, ngram_range=curr_n_gram, learning_rate=learning_rate, layers=layer, units=unit, dropout_rate=dropout_rate)
            x_train, x_test, names, vectorizer = ngram_vectorize(
                    X_train, y_train, X_test, ngram_range=curr_n_gram)
            y_pred_keras = model.predict(x_test).ravel()
 
            
            fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
        
            auc_keras = auc(fpr_keras, tpr_keras)
            total_auc += auc_keras
            
            f1 = f1_score(y_test, y_pred_keras.round(), average='macro')
            total_f1 += f1
            
        auc_ave = float(total_auc)/len(folded_data)
        f1_ave = float(total_f1)/len(folded_data)
        
        output_file = open(self.output_path, "w")
        output_file.write("{},{},{},{},{},{},{}\n".format(curr_ngram_size, curr_learning_rate, curr_dropout_rate, curr_unit_per_layer, curr_layer_count, auc_ave, f1_ave))
        output_file.close()
        
    def parse_config_and_run(self, config_dict, outFile):
        n_grams=config_dict['n_grams']
        learning_rates=config_dict['learning_rates']
        dropout_rates=config_dict['dropout_rates']
        units_per_layers=config_dict['units_per_layers']
        layers=config_dict['layers']

        for n_gram in n_grams:
            for learning_rate in learning_rates:
                for dropout_rate in dropout_rates:
                    for units_per_layer in units_per_layers:
                        for layer in layers:
                            self.run_model(
                                n_gram, 
                                learning_rate, 
                                dropout_rate, 
                                dropout_rate, 
                                units_per_layer, 
                                layer)


if __name__ == '__main__':
    
    # read in command-line arguments
    parser = argparse.ArgumentParser(description= "Grid search for tuning hyperparameters")
    parser.add_argument('-outFile', type = str, default = 'summary_statistics.txt', help = "OutFile for saving summary statistics")
    parser.add_argument('-configFile', type = str, default = 'none', help = "Config file for running the code")

    arg_vals = parser.parse_args()
    outFile = arg_vals.outFile
    configFile=arg_vals.configFile
    
    config_dict = {}
    
    # Parse the config file and run the code!
    with open(configFile, 'r') as fp:
        contents = fp.read()
        
        n_grams, learning_rates, layers, dropout_rates, units = contents.split("\t")
        
        config_dict['ngram'] = n_grams
        config_dict['learning_rate'] = learning_rates
        config_dict['dropout_rate'] = dropout_rates
        config_dict['layer'] = layers
        config_dict['unit'] = units
    
            
    train = HyperModel(args=config_dict, path=".", data_path="./data/vip_data", output_path="./grid_search/grid_search_output_02162022.csv")

    train.parse_config_and_run(config_dict, outFile)