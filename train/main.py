import numpy as np
import pandas as pd
import logging
import argparse
import shap


import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import StratifiedKFold
import nltk
from nltk.corpus import stopwords 

from tensorflow.python.keras import models
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import os
import random
from collections import Counter
import matplotlib.pyplot as plt

#shap.initjs()
tf.compat.v1.disable_v2_behavior()

stop_words = set(stopwords.words('english'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NGRAM_RANGE = (1,1)#(2, 3)

TOP_K = 20000

TOKEN_MODE = 'word'

MIN_DOCUMENT_FREQUENCY = 2

# https://developers.google.com/machine-learning/guides/text-classification/
def load_data(data_path, positive_percentage=1.0, seed=123, dataset_size=1000):
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
                    for w in f.read().split(' '): 
                        if w not in stop_words: 
                            filtered_sentence.append(w) 
                    train_texts.append(' '.join(filtered_sentence))
                train_labels.append(0 if category == 'neg' else 1)
                
    val_texts = []
    val_labels = []
    validation_pos = len(os.listdir(os.path.join(path_to_data, 'malaria_validation', "pos")))
    for category in ['pos', 'neg']:
        val_path = os.path.join(path_to_data, 'malaria_validation', category)
        for fname in sorted(os.listdir(val_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(val_path, fname)) as f:
                    val_texts.append(f.read())
                val_labels.append(0 if category == 'neg' else 1)
                
    test_texts = []
    test_labels = []
    testing_pos = len(os.listdir(os.path.join(path_to_data, 'malaria_testing', "pos")))
    for category in ['pos', 'neg']:
        test_path = os.path.join(path_to_data, 'malaria_testing', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    test_texts.append(f.read())
                test_labels.append(0 if category == 'neg' else 1)
                
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
    for i in range(len(train_labels)):
        if train_labels[i] == 1 and num_pos < pos_size:
            train_texts_.append(train_texts[i])
            train_labels_.append(train_labels[i])
            num_pos += 1
        elif train_labels[i] == 0 and num_neg < neg_size:
            train_texts_.append(train_texts[i])
            train_labels_.append(train_labels[i])
            num_neg += 1
    
    for i in range(len(val_labels)):
        val_texts_.append(val_texts[i])
        val_labels_.append(val_labels[i])
            
    for i in range(len(test_labels)):
        test_texts_.append(test_texts[i])
        test_labels_.append(test_labels[i])

    return ((train_texts_, np.array(train_labels_)),
            (val_texts_, np.array(val_labels_)),
            (test_texts_, np.array(test_labels_)))
    
def load_all_data(data_path, seed=123):
    path_to_data = os.path.join(data_path, 'Garud')
    texts = []
    labels = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(path_to_data, 'malaria_training', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    texts.append(f.read())
                labels.append(0 if category == 'neg' else 1)
                
    for category in ['pos', 'neg']:
        val_path = os.path.join(path_to_data, 'malaria_testing', category)
        for fname in sorted(os.listdir(val_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(val_path, fname)) as f:
                    texts.append(f.read())
                labels.append(0 if category == 'neg' else 1)
                
    for category in ['pos', 'neg']:
        test_path = os.path.join(path_to_data, 'malaria_testing', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    texts.append(f.read())
                labels.append(0 if category == 'neg' else 1)
                
    random.seed(seed)
    random.shuffle(texts)
    random.seed(seed)
    random.shuffle(labels)

    return texts, np.array(labels)
    

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
    
#non-sequenced n-grams
def ngram_vectorize(train_texts, train_labels, val_texts, ngram_range=NGRAM_RANGE):
    kwargs = {
            'ngram_range': NGRAM_RANGE,
            'dtype': 'int32',
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
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    return x_train, x_val, vectorizer.get_feature_names()

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
    x_train, x_val, name_train = ngram_vectorize(
        train_texts, train_labels, val_texts, ngram_range)
    
    x_train, x_test, name_test = ngram_vectorize(
        train_texts, train_labels, test_texts, ngram_range)

    # Create model instance.
    model = mlp_model(layers=layers,
                                  units=units,
                                  dropout_rate=dropout_rate,
                                  input_shape=x_train.shape[1:],
                                  num_classes=num_classes)

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    #callbacks = [tf.keras.callbacks.EarlyStopping(
        #monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            #callbacks=callbacks,
            #validation_split=0.2,
            validation_data=(x_val, val_labels),
            verbose=2, 
            batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    #model.save('VIP_mlp_model.h5')
    
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, test_labels, batch_size=batch_size)
    print('test loss, test acc:', results)
    
    return model, history, history['val_acc'][-1], history['val_loss'][-1]


def main():
    parsers = argparse.ArgumentParser(description='Training handler')
    parsers.add_argument('--pos_size', type=float, default=1.0, help='Percentage of positives to be included in training/validation/testing')
    
    args = parsers.parse_args()
    
    train_acc = []
    val_acc = []
    
    train_loss = []
    val_loss = []
    
    total_SHAP = []
    
    roc = []
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data("/Users/rahulnatarajan/Research", positive_percentage=0.3, dataset_size=1000)

    logger.info('Logging results for number of epochs: {0}'.format(215))

    for iterations in range(0, 15): 
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data("/Users/rahulnatarajan/Research", positive_percentage=0.3, dataset_size=1000)
        model, history, acc, loss = train_ngram_model(((X_train, y_train), (X_val, y_val), (X_test, y_test)), epochs=215, ngram_range=NGRAM_RANGE)

        x_train, x_test, names = ngram_vectorize(
                    X_train, y_train, X_test, ngram_range=NGRAM_RANGE)
        y_pred_keras = model.predict(x_test).ravel()
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
        roc.append((fpr_keras, tpr_keras, thresholds_keras))
    
        plt.figure(1)
        shap.initjs()
        x_arr = x_train.toarray()
        test_arr = x_test.toarray()
        background = x_arr[np.random.choice(x_arr.shape[0], 200, replace=False)]

        #flattened_arr = arr.flatten()
        #tf.convert_to_tensor(flattened_arr, dtype=tf.float32)
        explainer = shap.DeepExplainer(model, x_arr)
        shap_values = explainer.shap_values(x_arr)#test_arr[1:5])
        shap.summary_plot(shap_values, x_arr, feature_names=names, plot_type="bar")

    

        train_acc.append(history['acc'])
        val_acc.append(history['val_acc'])
                
        train_loss.append(history['loss'])
        val_loss.append(history['val_loss'])

    #np_total_SHAP = np.array(total_SHAP)
    #ave = np.average(np_total_SHAP)
    #print(np_total_SHAP)
    #print(ave)
    
    """
    plt.plot([0, 1], [0, 1], 'k--')
    for n_gram_range in range(0,1): 
        fpr_keras, tpr_keras, thresholds_keras = roc[n_gram_range]
        auc_keras = auc(fpr_keras, tpr_keras)
        plt.plot(fpr_keras, tpr_keras, label='Area = {:.3f} for ({},{})'.format(auc_keras, n_gram_range+1, n_gram_range+2))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    
    
    for n_gram_range in range(0,1):
        #plt.plot(train_loss[percentage], label='Train, Percentage {}'.format(percentage))
        plt.plot(val_loss[n_gram_range], label='Val, N-Gram({},{})'.format(n_gram_range+1, n_gram_range+2))
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
    
    for n_gram_range in range(0,1):
        #plt.plot(train_acc[percentage], label='Train, Percentage {}'.format(percentage))
        plt.plot(val_acc[n_gram_range], label='Val, N_gram ({},{})'.format(n_gram_range+1, n_gram_range+2))
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
    """

if __name__ == '__main__':
    main()

