from models import KNN
from pretraitement import DataSet
from models import KNN
import pickle
import os
import time

if __name__ == "__main__" :
    
    path = "train_data.pkl"
    test_path = "test_data.pkl"
    data_set: DataSet = DataSet.from_pickle(path)

    data_set.shuffle()
    data_set.normalize()
    data_set.flatten()

    data_set.set_test_data(test_path)

    # K = [151,152,153]
    # P = [21,22,23]
    # best = (0,0,0)
    # time_0 = time.time()
    # for k in K :
    #     """ print("==========\n" + f"Testing for k={k}\n" + "==========\n") """
    #     for p in P :
    #         """ print("==========\n"+f"Testing for p={p}\n"+"==========\n") """
    #         accuracy = data_set.evaluate(KNN, k=k, p=p)
    #         print(accuracy, f"p={p}, k={k}")
    #         if (accuracy > best[0]) :
    #             best = (accuracy, k, p)
    # delta_time = time.time() - time_0
    
    # print(f"Best KNN model : accuracy: {best[0]}, k={best[1]}, p={best[2]}" )

    # print("time training : ", delta_time)

    data_set.make_csv(KNN, k=152, p=22)
