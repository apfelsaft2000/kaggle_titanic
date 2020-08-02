import pandas as pd
import numpy as np
import sys
import os

if __name__ == "__main__":
    train = pd.read_csv("./data/train.csv")
    print(train.columns)
