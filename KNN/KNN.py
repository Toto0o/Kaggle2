import numpy as np

class KNN() :

    def __init__(self, k=5, p=2):
        """
        Modèle knn : prédit selon un vote parmis k voisins, déterminé 
        selon un métrique de distance de paramètre p
        
        :param self: knn class
        :param k: nombres de voisins
        :param p: paramètre de la distance
        """
        self.k = k
        self.p = p

    def fit(self, images, labels) :
        """
        Stock les images et les labels
        
        :param self: knn class
        :param images: images du dataset
        :param labels: labels des images du dataset
        """
        self.images = images
        self.labels = labels
    
    def predict_one(self, image) :
        """
        Prédit le label majoritaire pour une image
        
        :param self: knn class
        :param images: image à prédire
        """
        # calculer les distances
        distances = self.distance(image)
        # prendre les k premières images classés et trouver l'index
        dist_idx = np.argsort(distances)[:self.k]
        # calculer le poids de chaque label
        values, counts = np.unique(self.labels[dist_idx], return_counts=True)
        # return le label avec le plus de poids
        return values[np.argmax(counts)]
    
    def predict(self, images):
        """
        Prédit pour un array d'image avec predict_one()
        
        :param self: knn class
        :param images: array d'images à prédire
        """
        return np.array([self.predict_one(image) for image in images])

    def distance(self, image) -> np.ndarray :
        """
        Calcule la distance (de paramètre p) d'une image par rapport à l'ensemble d'images du modèle

        return un array de distances
        
        :param self: knn class
        :param image: l'image de référence
        :return: array de distance
        :rtype: ndarray[_AnyShape, dtype[Any]]
        """
        return np.sum(np.abs(image - self.images)**self.p, axis=1)**(1/self.p)
    
    def accuracy(self, images, labels) :
        """
        Détermine la précision des prédictions sur un ensemble d'images
        
        :param self: knn class
        :param images: les images à prédire
        :param labels: les labels correspondant aux images à prédire
        """
        predictions = self.predict(images)
        return np.mean(predictions == labels)
    