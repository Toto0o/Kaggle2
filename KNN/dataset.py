import pickle
import numpy as np
from KNN import KNN
import csv

class DataSet :

    def __init__(self, images, labels, seed=42, k=5):
        """
        Docstring for __init__
        
        :param self: DataSet pour un modèle K-NN
        :param images: np.ndarray d'images de formes 28x28x3
        :param labels: np.ndarray de labels 
        :param seed: pour la reproductiblité de l'expérience
        :param k: nombre de fold pour l'évaluation du modèle
        """
        self.images: np.ndarray = images
        self.labels: np.ndarray = labels
        self.n = self.images.shape[0]
        self.seed = seed
        self.k = k
    
    @classmethod
    def from_pickle(cls, path) :
        """
        Pour initialiser les données d'entrainement
        
        :param cls: DataSet class
        :param path: chemin vers les données
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        images = data['images']
        labels = data['labels']

        return cls(images, labels)
    
    def __getitem__(self) :
        images = self.images.copy()
        labels = self.labels.copy()
        return images, labels
    
    def set_test_kaggle_data(self, path, shuffle=True, normalize=True, flatten=True) :
        """
        Pour initialiser les données de test
        
        :param path: chemin vers les données de test
        :param shuffle: randomiser les données
        :param normalize: normaliser les données
        :param flatten: aplatir les données
        """
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
        """
        Normalisation des couleurs sur une échelle [0, 1]
        """
        self.images = self.images.astype("float32") / 255.0
    
    def flatten(self) :
        """
        Shape(N,28,28,3) -> Shape(N, 28x28x3=2352)
        """
        self.images = self.images.reshape(self.n, -1)

    def shuffle(self) :
        """
        Randomiser les données, avec une graine pour reproduire l'expérience
        """
        np.random.seed(self.seed)
        shuffled_idx = np.random.permutation(self.n)
        self.images = self.images[shuffled_idx]
        self.labels = self.labels[shuffled_idx]
    
    def kfold_split(self) :
        """
        Diviser les données en k folds (voir __init__) avec np.array_split
        """
        images_fold = np.array_split(self.images, self.k)
        labels_fold = np.array_split(self.labels, self.k)
        
        return images_fold, labels_fold
    
    def evaluate(self, model_class, **model_kwargs) :
        """
        Évalue et détermine la moyenne de précision du modèle selon k folds.
        
        :param self: Description
        :param model_class: la class du modèle knn
        :param model_kwargs: les paramètres du modèle
        """
        model: KNN = model_class(**model_kwargs)
        fold = self.kfold_split() #séparer en k folds
        accuracy = 0
        for i in range(self.k) :
            # concaténer les données d'entrainements en un array
            images_train = np.concatenate(fold[0][:i] + fold[0][i+1:], axis=0)
            images_val = fold[0][i]
            labels_train = np.concatenate(fold[1][:i] +fold[1][i+1:], axis=0)
            labels_val = fold[1][i]

            # prédire et calculer la précision
            model.fit(images_train, labels_train)
            accuracy += model.accuracy(images_val, labels_val)

        return accuracy/self.k
    
    def make_csv(self, model_class, **model_kwargs) :
        """
        Créer un fichier csv des prédictions sur les données de test; initialiser l'ensemble 
        de test avec set_test_kaggle_data()
        
        :param self: DataSet class
        :param model_class: la classe du modèle à évaluer
        :param model_kwargs: les paramètre du modèle
        """
        model: KNN = model_class(**model_kwargs)
        model.fit(self.images, self.labels)
        prediction = model.predict(self.test_images)

        with open("KNN_1", "x") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Label"])

            for i,pred in enumerate(prediction) :
                writer.writerow([i+1,pred])
            
            f.close()
