import pickle
import numpy as np
from models import Model
import csv

class DataSet :

    def __init__(self, images, labels, seed=42, k=5):
        self.images: np.ndarray = images
        self.labels: np.ndarray = labels
        self.n = self.images.shape[0]
        self.seed = 42
        self.k = k
    
    @classmethod
    def from_pickle(cls, path) :
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        images = data['images']
        labels = data['labels']

        return cls(images, labels)
    
    def set_test_kaggle_data(self, path, shuffle=True, normalize=True, flatten=True) :
        with open(path, "rb") as f:
            test_data = pickle.load(f)
        self.test_images: np.ndarray = test_data['images']
        if shuffle :
            np.random.seed(self.seed)
            shuffled_idx = np.random.permutation(len(self.test_images))
            self.test_images = self.test_images[shuffled_idx]
        if normalize:
            self.test_images = self.test_images.astype("float32") / 255.0
        if flatten :
            self.test_images = self.test_images.reshape(len(self.test_images),-1)
        
    
    def normalize(self) :
        self.images = self.images.astype("float32") / 255.0
    
    def flatten(self) :
        
        self.images = self.images.reshape(self.n, -1)

    def shuffle(self) :
        np.random.seed(self.seed)
        shuffled_idx = np.random.permutation(self.n)
        self.images = self.images[shuffled_idx]
        self.labels = self.labels[shuffled_idx]
    
    def kfold_split(self) :
        images_fold = np.array_split(self.images, self.k)
        labels_fold = np.array_split(self.labels, self.k)
        
        return images_fold, labels_fold
    
    def evaluate(self, model_class, **model_kwargs) :
        model: Model = model_class(**model_kwargs)
        fold = self.kfold_split()
        accuracy = 0
        for i in range(self.k) :
            images_train = np.concatenate(fold[0][:i] + fold[0][i+1:], axis=0)
            images_val = fold[0][i]
            labels_train = np.concatenate(fold[1][:i] +fold[1][i+1:], axis=0)
            labels_val = fold[1][i]

            model.fit(images_train, labels_train)
            accuracy += model.accuracy(images_val, labels_val)

        return accuracy/self.k
    
    def make_csv(self, model_class, **model_kwargs) :
        model: Model = model_class(**model_kwargs)
        model.fit(self.images, self.labels)
        prediction = model.predict(self.test_images)

        with open("KNN_1", "x") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Label"])

            for i,pred in enumerate(prediction) :
                writer.writerow([i+1,pred])
            
            f.close()






        

