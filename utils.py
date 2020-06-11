import warnings
import string
import re, os
import nltk
import pickle
import numpy as np
#from lemmatizer import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from stemming.porter2 import stem as porter2_stem
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from pprint import pprint
from docx import Document

from sklearn.metrics import precision_recall_fscore_support


def warn(*args, **kwargs):
    pass

warnings.warn = warn

punkt_param = PunktParameters()
abbreviation = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'sr', 'jr', 'inc', 'u.s', 'u.s.a']
punkt_param.abbrev_types = set(abbreviation)
sent_tokenizer = PunktSentenceTokenizer(punkt_param)

lemma = WordNetLemmatizer()
BULLET_SET = set(['*', '-', '\u2022', '\u2023', '\u25e6', '\u2043', '\u204c', '\u204d', '\u2219'])
PUNCTUATION_SET = set(string.punctuation)

stopWords = ['them', "haven't", 'again', 'this', "should've", 'wouldn', 'between', 'nor', 'very',
            'being', 'themselves', 'and', 'then', 'once', 'hadn', 'the', 'won', 'during', "she's",
            'yourselves', "wasn't", 'out', 'aren', "shouldn't", "you've", 'to',
            'didn', 'up', "mightn't", 'isn', 'were', 'further', 'by', 'i', 'too', "wouldn't", 'as',
            'doing', 'are', "don't", 'myself', 'his', "hadn't", 'him', 'in', 'haven', 'itself',
            'more', 'now', 'all', 'been', 'own', "won't", "it's", 'm', "aren't", 'over', 'until', 'from',
            'not', 'mightn', 'can', "you'll", 'after', "weren't", "that'll", 'down', 'her', 'only',
            'here', "couldn't", "didn't", 'no', 'does', 'against', 'ain', 'wasn', 'those',
            'd', 'of', 'herself', 're', 'or', "shan't", "you'd", 'a', 'yours', 'himself', 'me',
            'mustn', 'ourselves', 'couldn', 'above', 'while', 'other', 'll', 'into', "mustn't", 'needn',
            'before', 'if', 's', 'through', 'ours', 'shan', 'under', "doesn't", 've', 'weren', 'doesn'
            'each', 'both', 'their', 'shouldn', 'an', 'whom', 'these', 'ma', 'hasn', 'o', 'he',
            'was', 'same', "isn't", 'don', 'had', 'theirs', 'she', "needn't", 'be', 'will', 'its',
            'than', 'off', 'with', 'on', 'hers', 'because', 'at', 'so', 'am', "hasn't", 'is']

translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))


def textTokenizer(text):
    text = text.replace("\n", " ")
    text = ''.join([i if ord(i) < 128 else ' ' for i in text])
    text = text.strip().lower()
    text = "".join([i for i in text if(i not in PUNCTUATION_SET)])
    text = "".join([i for i in text if(i not in BULLET_SET)])
    text = " ".join([porter2_stem(lemma.lemmatize(word)) for word in text.split() if word not in stopWords])
    regex = re.compile('[^a-zA-Z]')
    text = " ".join([regex.sub('', word) for word in text.split()])
    text = re.sub(r" +", " ", text)
    text = " ".join([i for i in text.split() if len(i)>2])
    return text

def getMetric(y_test, preds):
    t = precision_recall_fscore_support(y_test, preds, average='micro', warn_for=('precision', 'recall', 'f-score', 'support'))
    return t

def getClassAccuracy(y_test, preds):
    class_0 = list()
    class_1 = list()
    preds_list = preds.tolist()
    for i in range(len(y_test)):
        if(int(y_test[i])==0):
            class_0.append(preds_list[i])
        elif(int(y_test[i])==1):
            class_1.append(preds_list[i])
    return (class_0.count(0)/len(class_0), class_1.count(1)/len(class_1))

def getPreds(probs, threshold=0.5):
    preds = list()
    for p in probs:
        if(np.argmax(p)==1):    preds.append(1)
        else:
            if(p[0]>threshold): preds.append(0)
            else:   preds.append(1)
    return preds
