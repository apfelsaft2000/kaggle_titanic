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
    train_data = pd.merge(train,labels,on="PassengerId")
    test_data  = pd.merge(test,labels,on="PassengerId")
    
    return train_data,test_data

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='前処理')
    parser.add_argument('--input_data_path',default="./data/")
    parser.add_argument('--output_data_path',default="./model_input_data/")
    args = parser.parse_args()
    #前処理
    train_data,test_data =  preproces(args.input_data_path)
    print(train_data)
    #データ保存
    #train_data.to_csv(args.output_data_path + "train.csv")
    #test_data.to_csv( args.output_data_path  + "test.csv")
