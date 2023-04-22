import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor 
import soundfile as sf
import tensorflow as tf
import pandas as pd
import numpy as np

import tensorflow as tf

import torch 
import torch.nn as nn
from surgeon_pytorch import Inspect, get_layers

import os
import glob

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

##################################################################################
import wandb
wandb.init()
#####################--READ_DATA--##################################################

path_df = "/home/u917/PROJECT/alzheimer/datos/df_filter.csv"
path_audio = "/home/u917/PROJECT/alzheimer/datos/audios/"

df = pd.read_csv(path_df)

df["file_cut_path"] = path_audio + df["file_name"] 
df = df.reset_index(drop=True)
dataset = df["file_cut_path"]

#####################--READ_INFERENCE_TENSORS--########################################

log = []
for i in range (len(df)):
    log.append(torch.load("/home/u917/PROJECT/alzheimer/torch_files14/" + f"file{i}.pt"))

######################--SPLIT-DATASETS--##############################################
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(log, y, random_state = 42, stratify = y)

for i in range(len(X_train)):
    X_train[i] = X_train[i][0].detach().numpy()
    X_train[i] = np.concatenate(X_train[i])

for i in range(len(X_test)):
    X_test[i] = X_test[i][0].detach().numpy()
    X_test[i] = np.concatenate(X_test[i])

###################--CLASIFICADOR- #################################################

enc = LabelEncoder()
y_train_trans = enc.fit_transform(y_train)
y_test_trans = enc.fit_transform(y_test)

classifier = XGBClassifier(objective="binary:logistic", missing=None, seed=42)

classi = classifier.fit(X_train,y_train_trans, verbose=True, eval_metric="aucpr", eval_set=[(X_test,y_test_trans)])

y_pred = classi.predict(X_test)

score = accuracy_score(y_test_trans, y_pred)

print(score)

def metricas(objetivo, prediccion):
    matriz_conf = confusion_matrix(objetivo, prediccion)
    score = accuracy_score(objetivo, prediccion)
    reporte = classification_report(objetivo, prediccion)
    metricas = [matriz_conf, score, reporte]
    return(metricas)

metricas = metricas(y_test_trans, y_pred)

for i in range(len(metricas)):
    print(metricas[i])
