#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 20:00:42 2020

@author: chiefguti
"""

import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.metrics import classification_report
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import matplotlib.pyplot as plt

def visualize_classifier(model, X, y,namef, ax=None, cmap='tab20b'):
    ax = ax or plt.gca()
 


    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, marker='.', cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=500),
                         np.linspace(*ylim, num=500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)
    plt.savefig(namef,dpi=300)
    plt.close()
    
    
def NormData(DataSet,data):
    for i in range(len(data[:,0])):
        DataSet[i,0]=data[:,0][i]
        DataSet[i,1]=data[:,1][i]
    DataSet[:,0]=(DataSet[:,0]-min(DataSet[:,0]))/(max(DataSet[:,0])-min(DataSet[:,0]))
    DataSet[:,1]=(DataSet[:,1]-min(DataSet[:,1]))/(max(DataSet[:,1])-min(DataSet[:,1]))
def ColorMap(Word,Map,M):
    Cmap=np.arange(M)
    for i in range(len(Word)):
        for j in range(len(Cmap)):
            if(Word[i]==Org[j]):
                Map[i]=Cmap[j]

def BER(x,y):
    j=0
    for i in range(len(x)):
        if(x[i]!=y[i]):
            j+=1
    return(j/len(x))
    
Code = np.loadtxt('PBRS.txt',skiprows=5)


INT=[]
for i in range(int(len(Code)/4)):
    INT.append(i*4)
CodeWord=[]
for i in range(len(Code)):
    CodeWord.append(str(int(Code[i])))

Word=[]
Org=[]
j=0
for i in INT:
    Word.append(CodeWord[i]+CodeWord[i+1]+CodeWord[i+2]+CodeWord[i+3])
    
    if((Word[j] in Org)==False ):
        Org.append(Word[j])
    j=j+1
Map=np.zeros(len(Word))
ColorMap(Word,Map,16)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 500, num = 1)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 20, num = 1)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1,2]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
print(random_grid)

N=np.arange(0,10,1)
L=np.arange(20,90,10)
p=np.arange(5,25,5)
SERforest=np.zeros((10,len(N)))
SERforestmit=np.zeros(10)
best=[]
bestmit=[]
for n in range(len(N)):
    for i in range(len(p)):
        data = np.loadtxt('N%iL20kmP%imW.txt'%(N[n],p[i]))
        datamit=[]
        datamit.append(data[:,2])
        datamit.append(data[:,3])
        datamit=np.transpose(datamit)
        DataSet=np.zeros((len(data[:,0]),2))
        NormData(DataSet,data)
        DataSetmit=np.zeros((len(data[:,0]),2))
        NormData(DataSetmit,datamit)
        plt.scatter(DataSet[:,0],DataSet[:,1],c=Map,s=1,cmap='tab20b')
        plt.tick_params(axis='both', which='both',length=0)
        plt.savefig('DataN%iL20kmP%imW.png'%(N[n],p[i]),dpi=250)
        plt.close()
        X_train, X_test, y_train, y_test = train_test_split(DataSet, Map, test_size = 0.20)
        gridF = GridSearchCV(RandomForestClassifier(), random_grid, cv = 2, verbose = 1, n_jobs = -1)
        bestF = gridF.fit(X_train, y_train)
        best.append(gridF.best_params_)
        forest=RandomForestClassifier(max_depth=best[i]['max_depth'],max_features=best[i]['max_features'],min_samples_leaf=best[i]['min_samples_leaf'],min_samples_split=best[i]['min_samples_split'],n_estimators=best[i]['n_estimators'])
        forest.fit(X_train,y_train)
        y_prediction=forest.predict(X_test)
        visualize_classifier(forest,X_train,y_train,'ConstN%iL20kmP%imW.png'%(N[n],p[i]))
        SERforest[i,n]=BER(y_test,y_prediction)
        Xmit_train, Xmit_test, ymit_train, ymit_test = train_test_split(DataSetmit, Map, test_size = 0.20)
        gridFmit = GridSearchCV(RandomForestClassifier(), random_grid, cv = 2, verbose = 1, n_jobs = -1)
        bestFmit = gridFmit.fit(Xmit_train, ymit_train)
        bestmit.append(gridFmit.best_params_)
        forestmit=RandomForestClassifier(max_depth=bestmit[i]['max_depth'],max_features=bestmit[i]['max_features'],min_samples_leaf=bestmit[i]['min_samples_leaf'],min_samples_split=bestmit[i]['min_samples_split'],n_estimators=bestmit[i]['n_estimators'])
        forestmit.fit(Xmit_train,ymit_train)
        ymit_prediction=forestmit.predict(Xmit_test)
        SERforestmit[i]=BER(ymit_test,ymit_prediction)
