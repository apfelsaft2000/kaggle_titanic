import argparse
import pandas as pd
import numpy as np
import sys
import os
import re


def preproces(data_path):
    #データ読み込み
    labels = pd.read_csv(data_path + "gender_submission.csv")
    train  = pd.read_csv(data_path + "train.csv")
    test   = pd.read_csv(data_path + "test.csv")
    #前処理
    #train_data = pd.merge(train,labels,on="PassengerId")
    test_data  = pd.merge(test,labels,on="PassengerId")
    #male->1,female->2に置換
    train_data = train.replace({"male":0,"female":1})
    test_data  = test_data.replace({"male":0,"female":1})
    #S->1,C->2,Q->3に置換
    train_data = train_data.replace({"S":0,"C":1,"Q":2})
    test_data  = test_data.replace({"S":0,"C":1,"Q":2})
    #PassengerId,Name,Ticket,Cabinのカラムを削除
    train_data.drop(["PassengerId","Name","Ticket","Cabin"],axis="columns",inplace=True)
    test_data.drop(["PassengerId","Name","Ticket","Cabin"],axis="columns",inplace=True)
    #labelをcolumsの最後尾に移動
    train_data = train_data.loc[:,["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Survived"]]
    #全体を0~1の範囲で正規化
    #print(train_data)
    train_data = ((train_data - train_data.min()) / (train_data.max() - train_data.min()))
    test_data = ((test_data - test_data.min()) / (test_data.max() - test_data.min()))
    #Ageの欠損値を平均値で補完
    train_data.fillna({"Age":train_data["Age"].mean()},inplace=True)
    test_data.fillna({"Age":test_data["Age"].mean()},inplace=True)
    print(train_data["Embarked"].value_counts())
    print(test_data["Embarked"].value_counts())
    #print(test_data.isnull().all())
    
    #Fareを0~1の範囲で正規化
    #train_data["Fare"] = (train_data["Fare"] -
    return train_data,test_data

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='前処理')
    parser.add_argument('--input_data_path',default="./data/")
    parser.add_argument('--output_data_path',default="./model_input_data/")
    args = parser.parse_args()
    #前処理
    train_data,test_data =  preproces(args.input_data_path)
    #print(train_data["Survived"].value_counts())
    #print(test_data)
    #print(train_data)
    #データ保存
    train_data.to_csv(args.output_data_path + "train.csv",index=None)
    test_data.to_csv( args.output_data_path  + "test.csv",index=None)
