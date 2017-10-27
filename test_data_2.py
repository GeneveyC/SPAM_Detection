# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:51:23 2017

@author: clemc
"""


import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import nltk, sys

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

#Some useful group
number = ['1','2','3','4','5','6','7','8','9','0']
lettre = ['a','z','e','r','t','y','u','i','o','p','q','s','d','f','g','h','j','k','l','m','w','x','c','v','b','n']
lettre_maj= ['A','Z','E','R','T','Y','U','I','O','P','Q','S','D','F','G','H','J','K','L','M','W','X','C','V','B','N']

def contain_number(sms):
    s=0
    for elt in sms:
        if elt in number:
            s+=1
    return s

def contain_maj(sms):
    s=0
    for elt in sms:
        if elt in lettre_maj:
            s+=1
    return s

def contain_caract(sms):
    s=0
    for elt in sms:
        if elt not in (number+lettre+lettre_maj):
            s+=1
    return s



def init_data():
    '''
    Fct which read the file 'spam.csv' and create a dataframe with label in {0,1} and text message
    '''
    sms = pd.read_csv('spam.csv',encoding='latin-1') 
    sms = sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
    sms = sms.rename(columns = {'v1':'label','v2':'message'})
    sms=sms.replace(['ham','spam'],[0,1])
    return sms


def add_expert_features(sms,df,Mt=0):
    '''
     we add expert features to the dataframe : length,counter_maj,counter_number,counter_speCaract
    '''
    # Add some features contain in the data
    df['length']=0
    for i in np.arange(0,len(df)):
        df.loc[i,'length'] = len(sms.loc[Mt+i,'message'])
    df['number']=False
    for i in range(0,len(df)):
        df.loc[i,'number']=contain_number(sms.loc[Mt+i,'message'])
    df['maj']=0
    for i in range(0,len(df)):
        df.loc[i,'maj']=contain_maj(sms.loc[Mt+i,'message'])
    df['caract']=0
    for i in range(0,len(df)):
        df.loc[i,'caract']=contain_caract(sms.loc[Mt+i,'message'])
    return df

def get_spam_ratio(df_train):
    '''
    Calculate the spam ratio of the word of the dataframe.
    df : DataFrame object where there is a columns 'label' with values in {0,1}
    Return vector spam_ratio of each word
    '''
    # Second , get sub-dataframe of ham and spam to calculate the frequency of each word for each label
    spam_train=df_train[df_train.label==1].drop('label',axis=1)
    ham_train=df_train[df_train.label==0].drop('label',axis=1)
    # Then Compute the ratio of spamness of a word.
    spam_ratio=((spam_train.sum()+1.0)/len(spam_train)) / ((ham_train.sum() + 1.0 ) / len(ham_train))
    return spam_ratio

def filter_df(df_train,rs=10,rh=0.2):
    '''
    Filtre features by deleting all feature with a spam_ratio between [rh,rs]
    Return new dataframe 
    '''
    spam_ratio=get_spam_ratio(df_train)
    df_train_T=df_train.drop('label',axis=1).T
    df_train_T['spam_ratio']=spam_ratio # add spam_ratio to the datafram 
    df_train_2=df_train_T[df_train_T['spam_ratio']>rs].drop('spam_ratio',axis=1).T # get the datafram only with word whose spam_ratio>rs
    df_train_3=df_train_T[df_train_T['spam_ratio']<rh].drop('spam_ratio',axis=1).T # get the datafram only with word whose spam_ratio<rh
    df_train=pd.concat([df_train_2,df_train_3], axis = 1)
    return df_train
    

def get_data(option=1,rs=10,rh=0.2,Mt=4000,random_state_=None):
    '''
    Fct which work on the dataframe to return several representation of data
    according to which option is choosen.
    
    
    Return : X_train,X_test,y_train,y_test,names
    Option 1 : return the features space
    Option 2 : return a smaller features space where we have a cutoff based on
                to the spamness of the word.
    Option 3 : We take the output of option 2 and we add other features : length,counter_maj,counter_number,counter_speCaract
    
    '''
    sms=init_data()     
    text=sms.message
    label=sms.label
    train_size_=Mt*1.0/len(sms)
    X_train,X_test,y_train,y_test=train_test_split(text,label,train_size=train_size_,random_state=random_state_,shuffle=True)
    # Transform data into numerical matrix (dtm)
    vect=CountVectorizer()
    vect.fit(X_train)
    X_train_dtm=vect.transform(X_train)
    X_test_dtm=vect.transform(X_test)
    feature_names=vect.get_feature_names()
    
    if option==1 :
        return X_train_dtm.toarray(),X_test_dtm.toarray(),y_train,y_test,feature_names
        
    else :
        df_train=pd.DataFrame(np.concatenate((y_train.T[:,None],X_train_dtm.toarray()),axis=1),columns=np.concatenate((np.array(['label']),vect.get_feature_names()),axis=0))
        df_train=filter_df(df_train,rs,rh) #deleted all non pertinent features with a criteria based on spam ratio
        new_feature_names=df_train.columns
        feature_names_deleted=[ elt  for elt in feature_names if(elt not in new_feature_names)]

        df_test=pd.DataFrame(X_test_dtm.toarray(),columns=feature_names)
        df_test=df_test.drop(feature_names_deleted,axis=1) #deleted all non pertinent features
        if option == 2 :
            X_train=df_train.as_matrix()
            X_test=df_test.as_matrix(columns=new_feature_names)
            return X_train,X_test,y_train,y_test,new_feature_names

        elif option==3 :
            df_train=add_expert_features(sms,df_train)
            df_test=add_expert_features(sms,df_test,Mt)
            new_feature_names=np.concatenate((new_feature_names,['length','number','maj','caract']),axis=0)
            X_train=df_train.as_matrix(columns=new_feature_names)
            X_test=df_test.as_matrix(columns=new_feature_names)
            return X_train,X_test,y_train,y_test,new_feature_names


X_train,X_test,y_train,y_test,feature_names=get_data(option=2,rs=300,rh=0.2,random_state_=1)
#print X_train
#print X_test
#print y_train.shape
#print y_test.shape
#print len(feature_names)

from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(X_train,y_train)
y_pred_class=nb.predict(X_test)
from sklearn import metrics
print metrics.accuracy_score(y_test,y_pred_class)
print metrics.confusion_matrix(y_test,y_pred_class)
