# Retinal Classification — KNN & CNN

Ce projet implémente deux modèles pour la classification de rétinographies (fond d’œil) en 5 classes :

- un **KNN** (premier jalon, baseline classique),
- un **CNN** avec PyTorch (deuxième jalon, modèle plus performant).

Le script principal est `retinal_classification.py` et permet :
- d’entraîner un modèle,
- de générer un fichier CSV de prédictions pour Kaggle,
- de tester sur un autre fichier de données `.pkl`.

---

## Données attendues

Le script suppose l’existence de fichiers **pickle** au format suivant :

- `train_data.pkl`  
  ```python
  {
      "images": np.ndarray de shape (N, H, W, C),
      "labels": np.ndarray de shape (N,)
  }


## Exécution

Pour lancer le script, il suffit simplement de lancer la commande suivante dans le dossier **Kaggle2**:
- python3 retinal_classification.py cnn --> pour générer le fichier de prédiction du modèle du premier jalon
- python3 retinal_classification.py knn --> pour générer le fichier de prédiction du modèle du deuxième jalon
- python3 retinal_classification.py <model> <other_test_data.pkl> 
