import numpy as np
import pandas as pd
import time
from DecisionTree import DecisionTree

class RandomForest:
    def __init__(self, train, test):
        self.dataset_train = train
        self.dataset_test = test
        self.root_list = list()
        self.oob_table = None
        self.pred_oob = list()
        self.pred_test = list()
        self.pred_train = list()
        self.error_oob = -1
        self.error_train = -1
        self.error_test = -1

    def BootStrapping(self):
        mask = np.random.randint(1000, size=500)
        sampled_dataset = self.dataset_train.iloc[mask]
        return sampled_dataset

    def train(self, n_tree, get_oob=False):
        self.oob_table = np.zeros((len(self.dataset_train), n_tree), dtype=bool)
        self.Testing_error = list()
        self.root_list = list()
        for i in range(n_tree):
            sampled_train_set = self.BootStrapping()
            if get_oob == True:
                df_all = self.dataset_train.merge(sampled_train_set.drop_duplicates(), how="left", indicator=True)
                mask_oob = df_all['_merge'] == 'left_only'  
                self.oob_table[:, i] = mask_oob
            print("\nPlanting %d-th tree: "%i)
            DT = DecisionTree(train=sampled_train_set, test=self.dataset_test)
            DT.train()
            self.root_list.append(DT.root)
        print("Training is finished!\n----------------------------")
        if get_oob == True:
            self.calc_Eoob()

    def calc_Eoob(self):
        oob_table = pd.DataFrame(data=np.transpose(self.oob_table))
        print(oob_table)
        self.pred_oob = list()
        for i in range(len(self.dataset_train)):
            not_used_tree = oob_table[i][oob_table[i]==True].index.tolist()
            pred_G = 0.0
            if len(not_used_tree) != 0:
                for root_index in not_used_tree:
                    pred_G += self.predict_procedure(self.root_list[root_index], self.dataset_train[i:i+1])
                self.pred_oob.append(np.sign(pred_G))
            else:
                self.pred_oob.append(-1.0)
        self.evaluate(mode="oob")

    def predict(self, mode):
        start_time = time.time()
        if mode == "test":
            self.pred_test = list()
            self.error_test = -1
            for i in range(len(self.dataset_test)):
                pred_G = 0.0
                for root in self.root_list:
                    pred_G += self.predict_procedure(root, self.dataset_test[i:i+1])
                self.pred_test.append(np.sign(pred_G))
            self.evaluate(mode="test")
        else:
            self.pred_train = list()
            self.error_train = -1
            for i in range(len(self.dataset_train)):
                pred_G = 0.0
                for root in self.root_list:
                    pred_G += self.predict_procedure(root, self.dataset_train[i:i+1])
                self.pred_train.append(np.sign(pred_G))
            self.evaluate(mode="train")
        elapsed_time = time.time() - start_time
        print("Elapsed time: %.3fs"%elapsed_time)

    def predict_procedure(self, node, data):
        if data[node["separate_feature"]].values < node["separate_value"]:
            if isinstance(node["left"], dict):
                return self.predict_procedure(node["left"], data)
            else:
                return node["left"]
        else:
            if isinstance(node["right"], dict):
                return self.predict_procedure(node["right"], data)
            else:
                return node["right"]

    def evaluate(self, mode):
        if mode == "train":
            pred = np.array(self.pred_train)
            target = self.dataset_train["y"]
            err_amount = (pred != target).sum()
            self.error_train = err_amount/len(self.dataset_train)
            print("Training error is: %.3f"%(self.error_train))
        elif mode == "test":
            pred = np.array(self.pred_test)
            target = self.dataset_test["y"]
            err_amount = (pred != target).sum()
            self.error_test = err_amount/len(self.dataset_test)
            print("Testing error is: %.3f"%(self.error_test))
        else:
            pred = np.array(self.pred_oob)
            target = self.dataset_train["y"]
            err_amount = (pred != target).sum()
            self.error_oob = err_amount/len(self.dataset_train)
            print("Oob validation error is: %.3f"%(self.error_oob))