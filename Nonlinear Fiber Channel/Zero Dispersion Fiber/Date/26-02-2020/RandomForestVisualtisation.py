#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 20:00:42 2020

@author: chiefguti
"""

import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from scipy import stats



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
    plt.show( )
    plt.close()
    
    
def NormData(DataSet,data):
    for i in range(len(data[:,0])):
        DataSet[i,0]=data[:,0][i]
        DataSet[i,1]=data[:,1][i]
    DataSet[:,0]=(DataSet[:,0]-min(DataSet[:,0]))/(max(DataSet[:,0])-min(DataSet[:,0]))
    DataSet[:,1]=(DataSet[:,1]-min(DataSet[:,1]))/(max(DataSet[:,1])-min(DataSet[:,1]))
    
def ColorMap(Word,Org,Map,M):
    Cmap=np.arange(M)
    for i in range(len(Word)):
        for j in range(len(Cmap)):
            if(Word[i]==Org[j]):
                Map[i]=Cmap[j]
def SER(x,y):
    j=0
    for i in range(len(x)):
        if(x[i]!=y[i]):
            j+=1
    return(j/len(x))
    


def VoronoiPlot(DataSet,Map,ConPoint,NameFig):
    points=[]
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Map, test_size = 0.2)
    kmeans = KMeans(n_clusters=ConPoint, random_state=0).fit(X_train)
    mapK=kmeans.predict(X_test)
    center=kmeans.cluster_centers_
    Voromap=np.zeros((ConPoint,2))
    Class=np.arange(0,ConPoint)
    for i in Class:
        Voromap[:,0][i]=np.where(y_test==i)[0][0]
        Voromap[:,1][i]=mapK[np.where(y_test==i)[0][0]]
    mtest=np.zeros(len(mapK))
    for i in range(len(Voromap)):
        serch=np.where(mapK==Voromap[:,1][i])[0]
        for j in serch:
            mtest[j]=i
    mapK=mtest 
    vor=Voronoi(center)
    voronoi_plot_2d(vor, show_vertices=False, line_colors='black',line_width=2, line_alpha=0.6, point_size=2)
    plt.scatter(X_test[:,0],X_test[:,1],c=mapK,s=1)
    plt.colorbar()
    plt.savefig(NameFig,dpi=300)
    plt.show()
    plt.close()
    return y_test, mapK

def RandForest(DataSet,Map,NameFig):
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Map, test_size = 0.2)
    gridF = GridSearchCV(RandomForestClassifier(), random_grid, cv = 5, verbose = 1, n_jobs = -1)
    bestF = gridF.fit(X_train, y_train)
    best=gridF.best_params_
    forest=RandomForestClassifier(max_depth=best['max_depth'],max_features=best['max_features'],min_samples_leaf=best['min_samples_leaf'],min_samples_split=best['min_samples_split'],n_estimators=best['n_estimators'])
    forest.fit(X_train,y_train)
    y_prediction=forest.predict(X_test)
    visualize_classifier(forest,X_train,y_train,NameFig)
    return y_test , y_prediction




def SVCfun(DataSet,Map,NameFig):
    X_train, X_test, y_train, y_test = train_test_split(DataSet, Map, test_size = 0.20)
    for score in scores:
        clf = RandomizedSearchCV(SVC(), tuned_parameters, cv=4,
                       scoring='%s_macro' % score, n_jobs=-1, error_score=0,n_iter=20)
        clf.fit(X_train, y_train)
    bes=clf.best_params_
    svclassifier = SVC(kernel='rbf',C=bes['C'],gamma= bes['gamma'],class_weight=bes['class_weight'])
    svclassifier.fit(X_train, y_train)
    y_prediction= svclassifier.predict(X_test)
    visualize_classifier(svclassifier,X_train,y_train,NameFig)
    return y_test, y_prediction

def PBRSdecod(Code,bits):
    INT=[]
    for i in range(int(len(Code)/bits)):
        INT.append(i*bits)
        CodeWord=[]
    for i in range(len(Code)):
        CodeWord.append(str(int(Code[i])))
    Word=[]
    Org=[]
    j=0
    for i in INT:
        Word.append(''.join(CodeWord[i:i+bits]))
        if((Word[j] in Org)==False ):
            Org.append(Word[j])
        j=j+1
    return Word , Org








#Random Forest grid
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 500, num = 3)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 20, num = 3)]
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
    
    
#Suport Vector Machine grid
tuned_parameters ={'C': stats.expon(scale=10), 'gamma': stats.expon(scale=5),
                 'kernel': ['linear','rbf'], 'class_weight':['balanced', None]}
    
scores = ['precision', 'recall']












Code = np.loadtxt('PBRS.txt',skiprows=5)

Word, Org=PBRSdecod(Code,4)

Map=np.zeros(len(Word))

ColorMap(Word,Org,Map,16)

ConPoint=16





N=np.arange(0,30,1)
L=np.arange(20,120,20)
p=[5,10,15,20,25,30,35,40,44,50,54,60]

SERforest=np.zeros((len(N),len(L),len(p)))
SERVoro=np.zeros((len(N),len(L),len(p)))
SERSVC=np.zeros((len(N),len(L),len(p)))
SERVoromit=np.zeros((len(N),len(L),len(p)))





for l in range(len(L)):
    for n in range(len(N)):
        for i in range(len(p)):
            data = np.loadtxt('ZDF4bQAM%imW%iKm%in.txt'%(p[i],L[l],N[n]))
            datamit=[]
            datamit.append(data[:,2])
            datamit.append(data[:,3])
            datamit=np.transpose(datamit)
            DataSet=np.zeros((len(data[:,0]),2))
            NormData(DataSet,data)
            DataSetmit=np.zeros((len(data[:,0]),2))
            NormData(DataSetmit,datamit)
            plt.scatter(DataSetmit[:,0],DataSetmit[:,1],c=Map[:len(DataSet)],s=1,cmap='tab20b')
            plt.tick_params(axis='both', which='both',length=0)
            plt.savefig('4QAMmit%iN%ikm%imW.png'%(N[n],L[l],p[i]),dpi=250)
            plt.close()
            y_Vorotest,y_Voroprediction=VoronoiPlot(DataSet,Map[:len(DataSet)],ConPoint,'Voro4QAM%iN%ikm%imW.png'%(N[n],L[l],p[i]))
            y_Voromittest,y_Voromitprediction=VoronoiPlot(DataSetmit,Map[:len(DataSet)],ConPoint,'Voro4QAMmit%iN%ikm%imW.png'%(N[n],L[l],p[i]))
            y_Foretest,y_Foreprediction=RandForest(DataSet,Map[:len(DataSet)],'Fore4QAM%iN%ikm%imW.png'%(N[n],L[l],p[i]))
            y_SVCtest,y_SVCprediction=SVCfun(DataSet,Map[:len(DataSet)],'SVC4QAM%iN%ikm%imW.png'%(N[n],L[l],p[i]))
            SERSVC[n,l,i]=SER(y_SVCtest,y_SVCprediction)
            SERforest[n,l,i]=SER(y_Foretest,y_Foreprediction)
            SERVoro[n,l,i]=SER(y_Vorotest,y_Voroprediction)
            SERVoromit[n,l,i]=SER(y_Voromittest,y_Voromitprediction)
            print( 'Length %i'%L[l], 'Span %i'%N[n],'Power %i'%p[i],SER(y_Vorotest,y_Voroprediction),SER(y_Voromittest,y_Voromitprediction), SER(y_Foretest,y_Foreprediction),SER(y_SVCtest,y_SVCprediction))   







 
for l in range(len(L)):
    for i in range(len(p)):
            os.system('convert -delay 40 $(ls -v 4QAMmit*N%ikm%imW.png) 4QAMmit%ikm%imW.gif'%(L[l],p[i],L[l],p[i]))
            os.system('convert -delay 40 $(ls -v 4QAM*N%ikm%imW.png) 4QAM%ikm%imW.gif'%(L[l],p[i],L[l],p[i]))
            os.system('convert -delay 60 $(ls -v Fore4QAM*N%ikm%imW.png) Fore4QAM%ikm%imW.gif'%(L[l],p[i],L[l],p[i]))
            os.system('convert -delay 60 $(ls -v Voro4QAM*N%ikm%imW.png) Vor4QAM%ikm%imW.gif'%(L[l],p[i],L[l],p[i]))
            os.system('convert -delay 60 $(ls -v SVC4QAM*N%ikm%imW.png) SVC4QAM%ikm%imW.gif'%(L[l],p[i],L[l],p[i]))








#
#
#data = np.loadtxt('ZDF4bQAM60mW20Km20n.txt')
#datamit=[]
#datamit.append(data[:,2])
#datamit.append(data[:,3])
#datamit=np.transpose(datamit)
#DataSet=np.zeros((len(data[:,0]),2))
#NormData(DataSet,data)
#DataSetmit=np.zeros((len(data[:,0]),2))
#NormData(DataSetmit,datamit)
#plt.scatter(DataSet[:,0],DataSet[:,1],c=Map[:len(DataSet)],s=1,cmap='tab20b')
#plt.tick_params(axis='both', which='both',length=0)
#
#
##y_Voroprediction=
#VoronoiPlot(DataSetmit,Map[:len(DataSet)],ConPoint,'test.png')
##print( SER(y_SVCtest,y_SVCprediction)) 
#
#
#
#


#
#
#
#for l in range(len(L)):
#    for n in range(len(N)):
#        for i in range(len(p)):
#            data = np.loadtxt('ZDF4bQAM%imW%iKm%in.txt'%(p[i],L[l],N[n]))
#            datamit=[]
#            datamit.append(data[:,2])
#            datamit.append(data[:,3])
#            datamit=np.transpose(datamit)
#            DataSet=np.zeros((len(data[:,0]),2))
#            NormData(DataSet,data)
#            DataSetmit=np.zeros((len(data[:,0]),2))
#            NormData(DataSetmit,datamit)
#            y_SVCtest,y_SVCprediction=SVCfun(DataSet,Map[:len(DataSet)],'SVC4QAM%iN%ikm%imW.png'%(N[n],L[l],p[i]))
#            SERSVC[n,l,i]=SER(y_SVCtest,y_SVCprediction)
#            print( p[i],L[l],N[n],SER(y_SVCtest,y_SVCprediction))   