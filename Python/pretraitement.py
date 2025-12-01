import pickle
import numpy as np

class dataPipline :

    def __init__(self, train_data_path:str):
        with open(train_data_path, "rb") as f:
            data = pickle.load(f)
        
        self.images = data['images']
        self.labels = data['labels']
        N, H, W, C = self.images.shape
        self.N = N
        self.H = H
        self.W = W
        self.C = C
    
    def normalize(self) :
        images_norm = self.images.astype("float32") / 255.0
        return images_norm.reshape(self.N, -1)
    
    def train_val_split(self, seed, size) :
        np.random.seed(seed)

        indices = np.random.permutation(self.N)

        images_shuffles = self.images[indices]
        labels_shuffle = self.labels[indices]

        images_train = images_shuffles[:size]
        images_val = images_shuffles[size:]

        labels_train = labels_shuffle[:size]
        labels_val = labels_shuffle[size:]

        return images_train, images_val, labels_train, labels_val


class KNN :

    def __init__(self, images, labels) :
        self.images = images
        self.labels = labels

    @staticmethod
    def distance(x1, x2, p) :
        return (np.sum((np.abs(x1-x2))**p, axis=1))**(1/p)

    def predict(self, k, data, p) :
        distances = self.distance(data, self.images, p)
        nn_index = np.argsort(distances)[:k]
        return self.labels[nn_index]        



        

