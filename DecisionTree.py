import pandas as pd
import numpy as np
import time
    
class DecisionTree:
    def __init__(self, train, test):
        self.dataset_train = train
        self.dataset_test = test
        self.root = None
        self.pred_train = list()
        self.pred_test = list()
        self.error_test = -1
        self.error_train = -1

    def Split_data(self, feature, theta, data):
        mask = data[feature] < theta
        left_group = data[mask]
        right_group = data[~mask]
        return (left_group, right_group)

    def Get_Gini_Index(self, groups):
        gini = 0.0
        for group in groups:
            set_size = group["y"].count()
            if set_size != 0:
                set_size = group["y"].count()
                number_of_class1 = (group["y"] == 1.0).sum()
                number_of_class2 = set_size-number_of_class1
                score1 = number_of_class1/set_size
                score2 = number_of_class2/set_size
                score = 1 - (score1**2 + score2**2)
                gini += score
            else:
                continue
        return gini
    
    def best_split(self, dataset):
        best_gini, best_separate_point, best_separate_feature, best_groups=9999, 9999, 9999, None
        for feature in dataset.columns.to_list()[:-1]:
            median_of_feature = dataset[feature].median()
            groups = self.Split_data(feature=feature, theta=median_of_feature, data=dataset)
            gini = self.Get_Gini_Index(groups=groups)
            if gini < best_gini:
                best_gini, best_groups, best_separate_feature, best_separate_point = gini, groups, feature, median_of_feature
        return {"separate_feature": best_separate_feature, "separate_value": best_separate_point, "groups": best_groups}
    
    def processing_tree(self, node):
        left, right = node["groups"]
        del(node["groups"])

        if left.empty or right.empty:
            node["left"] = node["right"] = pd.concat([left, right])["y"].unique()[0]
        else:
            """ Process the left branch """
            if len(left["y"].unique()) == 1:
                node["left"] = left["y"].unique()[0]    # Return the only one value left in the left branch.
            else:
                node["left"] = self.best_split(left)
                self.processing_tree(node["left"])

            """ Process the right branch """
            if len(right["y"].unique()) == 1:
                node["right"] = right["y"].unique()[0]    # Return the only one value left in the right branch.
            else:
                node["right"] = self.best_split(right)
                self.processing_tree(node["right"])
    
    def train(self):
        print("training...")
        start_time = time.time()
        self.root = self.best_split(self.dataset_train)
        self.processing_tree(self.root)
        elapsed_time = time.time() - start_time
        print("Elapsed time: %.3fs"%elapsed_time)

    def predict(self, mode="train"):
        print("Predicting...")
        start_time = time.time()
        if mode == "test":
            self.pred_test = list()
            self.error_test = -1
            for i in range(len(self.dataset_test)):
                self.pred_test.append(self.predict_procedure(self.root, self.dataset_test[i:i+1]))
            self.evaluate(mode="test")
        else:
            self.pred_train = list()
            self.error_train = -1
            for i in range(len(self.dataset_train)):
                self.pred_train.append(self.predict_procedure(self.root, self.dataset_train[i:i+1]))
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
        else:
            pred = np.array(self.pred_test)
            target = self.dataset_test["y"]
            err_amount = (pred != target).sum()
            self.error_test = err_amount/len(self.dataset_test)
            print("Testing error is: %.3f"%(self.error_test))

