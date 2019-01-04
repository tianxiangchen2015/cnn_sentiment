# coding: utf-8
import tensorflow as tf
import pandas as pd
import numpy as np
import json, argparse, os
import re
import io
import sys
from keras.models import Model
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import num2words
import gensim
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import enchant
from enchant.checker import SpellChecker
from nltk.metrics.distance import edit_distance
from keras.layers import (Input, Dense, Embedding, LSTM, Concatenate, Reshape, GRU, Conv1D,
                            Bidirectional, BatchNormalization, Activation, Dropout, GlobalMaxPooling1D)
from keras.utils import to_categorical
from keras import optimizers


numDimensions = 300
maxSeqLength = 250
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'l", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"\'em", " them", phrase)
    phrase = re.sub(r"\'nt", " not", phrase)
    phrase = re.sub(r"\'", " ", phrase)
    
    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’l", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)
    phrase = re.sub(r"\’em", " them", phrase)
    phrase = re.sub(r"\’nt", " not", phrase)
    phrase = re.sub(r"\’", " ", phrase)
    
    
    return phrase

class MySpellChecker():

    def __init__(self, dict_name='en_US', max_dist=2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = max_dist

    def replace(self, word):
        suggestions = self.spell_dict.suggest(word)

        if suggestions:
            for suggestion in suggestions:
                if edit_distance(word, suggestion) <= self.max_dist:
                    return suggestions[0]

        return word


# In[51]:


# label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
# emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}

def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """

    # use these three lines to do the replacement
    labels = []
    u1 = []
#     indices = []
    df = pd.read_csv(dataFilePath, sep='\t')

#     indices = df.ItemID.values
    u1 = df.SentimentText.tolist()
    labels = df.Sentiment.tolist()

    return labels, u1

# In[52]:


def getMetrics(predictions, ground):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification  
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1))
    
    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)
    
    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)
    
    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))
    
    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))   
    
    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()    
    
    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))
    
    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)
    
    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------
    
    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)
    
    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


# In[53]:


def getEmbeddingMatrix(out_of_vocab):

    embeddingsIndex = {}
    vocab = []
    wordIndex = {}
    with io.open(gloveDir, encoding="utf8") as f:
        for line in f:
            values = line.split(' ')
           # print(values)
            word = values[0]
            vocab.append(word)
            embeddingVector = np.array([float(val) for val in values[1:]])
            embeddingsIndex[word] = embeddingVector
  
    print('Found %s word vectors.' % len(embeddingsIndex))

    # Minimum word index of any word is 1. 
    embeddingMatrix = np.zeros((len(vocab) + 2, EMBEDDING_DIM), dtype='float32')
    for i, (word, vector) in enumerate(embeddingsIndex.items()):
        if len(vector) > 199:
            pos = i + 1
            wordIndex[word] = pos
            embeddingMatrix[pos] = vector
    pos += 1
    wordIndex["<unk>"] = pos
    embeddingMatrix[pos] = np.random.uniform(low=-0.05, high=0.05, size=EMBEDDING_DIM)

    return embeddingMatrix, wordIndex

def buildModel(embeddingMatrix, weights_path=None):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    x1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input1')
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    emb = embeddingLayer(x1)

    cnn_1 = Conv1D(200, 1, padding='same', activation='relu')(emb)
    cnn_2 = Conv1D(200, 2, padding='same', activation='relu')(emb)
    cnn_3 = Conv1D(200, 3, padding='same', activation='relu')(emb)
    
    vect_1 = GlobalMaxPooling1D()(cnn_1)
    vect_2 = GlobalMaxPooling1D()(cnn_2)
    vect_3 = GlobalMaxPooling1D()(cnn_3)

    inp = Concatenate(axis=-1)([vect_1, vect_2, vect_3])

    inp = Reshape((3*LSTM_DIM,))(inp)
    
    dense_a = Dense(50, activation='relu')(inp)
    dense_a = Dropout(rate=0.5)(dense_a)
    out = Dense(1, activation='sigmoid')(dense_a)
    
    adam = optimizers.adam(lr=LEARNING_RATE)
    model = Model(x1,out)
    if weights_path:
        model.load_weights(weights_path)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])
    print(model.summary())
    return model

# parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
# parser.add_argument('-config', help='Config to read details', required=True)
# args = parser.parse_args()

# with open('testBaseline.config') as configfile:
#     config = json.load(configfile)

global trainDataPath, testDataPath, solutionPath, gloveDir
global NUM_FOLDS, NUM_CLASSES, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE    

trainDataPath = "/home/tianxiangchen1/semeval/data/Sentiment-Analysis-Dataset_norm.csv"
gloveDir = "/home/tianxiangchen1/semeval/word_embedding/datastories.twitter.200d.txt"

# NUM_FOLDS = config["num_folds"]
NUM_CLASSES = 2
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 200
BATCH_SIZE = 200
LSTM_DIM = 200
DROPOUT = 0.5
LEARNING_RATE = 0.001
NUM_EPOCHS = 7

print("Processing training data...")
labels, trainTexts = preprocessData(trainDataPath, mode="train")
out_of_vocab = []
embeddingMatrix, wordIndex = getEmbeddingMatrix(out_of_vocab)


def text_to_seq(wordIndex, sent):
    vocab = wordIndex.keys()
    seq = []
    tokens = sent.split(' ')
    for w in tokens:
        if w == '</allcaps>':
            w = '<allcaps>'
        if w not in vocab:
            w = '<unk>'
        seq.append(wordIndex[w])
        
    return seq


trainSequences = [text_to_seq(wordIndex, x) for x in trainTexts]
                                                                                                         
# wordIndex = tokenizer.word_index
print("Found %s unique tokens." % len(wordIndex))
print("Populating embedding matrix...")

data = pad_sequences(trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.asarray(labels)
# labels = to_categorical(np.asarray(labels))

# Randomize data
# np.random.shuffle(trainIndices)
# u1_data = u1_data[trainIndices]
# u2_data = u2_data[trainIndices]
# u3_data = u3_data[trainIndices]
# labels = labels[trainIndices]
print("Starting k-fold cross validation...")
print('-'*40)
print("Building model...")
print(len(out_of_vocab))
weights_path = 'distant-pretrain-weights.h5'
model = buildModel(embeddingMatrix, weights_path=weights_path)
model.fit(data, labels, validation_split=0.2,
          epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
model.save_weights('distant-pretrain-weights-8e.h5')

