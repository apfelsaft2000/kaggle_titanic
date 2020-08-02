from torch.utils.data import TensorDataset,DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import pandas as pd
import argparse
import numpy as np
import model
import sys
import os

def data_loader(path,batch_size):
    #訓練・テストデータの読み込み
    train = pd.read_csv(path + "train.csv")
    test = pd.read_csv(path + "test.csv")
    #pandas -> numpy -> torch.Tensor型に変換し、データとラベルを分割
    train_tensor = torch.from_numpy(train.values)
    test_tensor =  torch.from_numpy(test.values)
    X_train = train_tensor[:-1]
    Y_train = train_tensor[-1]
    X_test = test_tensor[:-1]
    Y_test = test_tensor[-1]
    #読み込んだデータをdataloader型に変換
    train_dataset = TensorDataset(X_train,Y_train)
    test_dataset = TensorDataset(X_test,Y_test)
    train_loader = DataLoader(train_dataset,batch_size=batch_size)
    test_loader = DataLoader(test_dataset,batch_size=batch_size)

    return train_loader,test_loader

def train(model,train_loader,device,optimizer,epoch,batch_size):
    model.train()
    sum_loss = 0
    sum_correct = 0
    sum_total = 0
    for (inputs,labels) in train_loader:
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        _,predicted = outputs.max(1)
        sum_total += labels.size(0)
        sum_correct += (predicted == labels).sum().item()
        print("train:epoch={},[{}/{}],loss={},accuracy{}".format(epoch,sum_total,len(train_loader),sum_loss*batch_size/len(train_loader), float(sum_correct/sum_total)))

def test(model,test_loader,device,optimizer,epoch,batch_size):
    model.eval()
    sum_loss = 0
    sum_correct = 0
    for (inputs,labels) in test_loader:
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        sum_loss += loss.item()
        _,predicted = outputs.max(1)
        sum_correct += (predicted == labels).sum().item()
    print("test:epoch{},loss={},accuracy={}".format(epoch,sum_loss*batch_size/len(train_loader), float(sum_correct/sum_total)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='構築したCNNモデルの訓練・テストを行う')
    parser.add_argument('data_path',default="./model_input_data/")
    parser.add_argument('batch_size',default=100)
    parser.add_argument('weight_decay',default=0.005)
    parser.add_argument('learing_late',default=0.0001)
    parser.add_argument('epoch',default=100)
    parser.add_argument('output_data_path',default="./output/data/")
    parser.add_argument('trained_model',default="./output/model")
    args = parser.parse_args()

    #データ読み込み
    print("data loading...")
    train_loader,test_loader = data_loader(args.data_path,args.batch_size)

    #ハイパーパラメータなど、各種設定
    print("model creating...")
    device = torch.device("cpu")
    model = CNN(input_size=len(train_loader[0]))
    model = model.to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=args.learing_late,momentum=0.9,weight_decay=args.weight_decay)
    print("model training...")
    for epoch in range(args.epoch):
        train(model,train_loader,device,optimizer,criterion,epoch,batch_size)
        test(model,train_loader,device,optimizer,criterion,epoch,batch_size)
    print("finish!!")
    #train = pd.read_csv("./data/train.csv")
    #print(train["Cabin"])
    
