from DecisionTree import DecisionTree
from RandomForest import RandomForest
import pandas as pd

def getData():
    data_train = pd.read_csv("./hw6_train.dat", sep=" ", header=None).rename(columns={10:"y"})
    data_test = pd.read_csv("./hw6_test.dat", sep=" ", header=None).rename(columns={10:"y"})
    return data_train, data_test

if __name__ == "__main__":
    train_set, test_set = getData()
    RF = RandomForest(train=train_set, test=test_set)
    RF.train(n_tree=2000, get_oob=True)
    # RF.predict(mode="train")
    # RF.predict(mode="test")
    # DT = DecisionTree(train=train_set, test=test_set)
    # DT.train()
    # DT.predict(mode="test")