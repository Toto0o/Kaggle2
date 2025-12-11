import numpy as np

class knn() :

    def __init__(self, k=5, p=2):
        self.k = k
        self.p = p

    def fit(self, images, labels) :
        self.images = images
        self.labels = labels
    
    def predict_one(self, images) :
        distances = self.distance(images)
        dist_idx = np.argsort(distances)[:self.k]
        values, counts = np.unique(self.labels[dist_idx], return_counts=True)
        return values[np.argmax(counts)]
    
    def predict(self, images):
        return np.array([self.predict_one(image) for image in images])

    def distance(self, images) -> np.ndarray :
        return np.sum(np.abs(images - self.images)**self.p, axis=1)**(1/self.p)
    
    def accuracy(self, images, labels) :
        predictions = self.predict(images)
        return np.mean(predictions == labels)
    