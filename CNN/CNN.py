import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


    
class CNN(nn.Module):

    def __init__(self, n_classes=5, in_channels=3, dropout=0.5):
        super().__init__()

        """
        Définition des blocs convolutionnel 1,2 et 3

        Paramètres :
        ---------- 
        n_classes : le nombre de classe du jeu de donnée
        in_channels : le nombre de channel de couleur des images
        droupout : pour éviter l'overfiting
        ----------

        """

        # ============================
        # Bloc convolutionnel 1
        # Détection de caractéristiques de bas niveau :
        # contours, textures simples, gradients de couleur
        # ============================
        self.conv1 = nn.Sequential(
            # Conv 1.1 : extraction initiale de motifs locaux
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),     # stabilise l'entraînement
            nn.ReLU(),              # non-linéarité

            # Conv 1.2 : raffinement des motifs de bas niveau
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Réduction de la résolution spatiale (H/2, W/2)
            nn.MaxPool2d(2)
        )

        # ============================
        # Bloc convolutionnel 2
        # Détection de motifs intermédiaires :
        # formes, parties d'objets
        # ============================
        self.conv2 = nn.Sequential(
            # Conv 2.1
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Conv 2.2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Réduction de la résolution spatiale (H/2, W/2)
            nn.MaxPool2d(2)
        )

        # ============================
        # Bloc convolutionnel 3
        # Détection de caractéristiques de haut niveau :
        # structures complexes, concepts visuels
        # ============================
        self.conv3 = nn.Sequential(
            # Conv 3.1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Conv 3.2
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Réduction de la résolution spatiale (H/2, W/2)
            nn.MaxPool2d(2)
        )

        # ============================
        # Global Average Pooling
        # Transforme chaque carte de caractéristiques
        # en une valeur unique → vecteur 128
        # ============================
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Dropout pour réduire l’overfitting
        self.dropout = nn.Dropout(dropout)

        # ============================
        # Classifieur fully-connected
        # ============================
        self.fc = nn.Sequential(
            nn.Linear(128, 64),     # réduction de dimension
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)  # scores pour chaque classe
        )

    def forward(self, images):
        """
        Propagation avant du CNN

        Paramètres
        ----------
        images : Tensor (N, C, H, W)
            Batch d'images d'entrée

        Retour
        ------
        Tensor (N, n_classes)
            Scores (logits) pour chaque classe
        """

        # Extraction hiérarchique des caractéristiques
        images = self.conv1(images)
        images = self.conv2(images)
        images = self.conv3(images)

        # Pooling global → (N, 128, 1, 1)
        images = self.global_pool(images)

        # Aplatissement → (N, 128)
        images = images.view(images.size(0), -1)

        # Régularisation
        images = self.dropout(images)

        # Classification finale
        images = self.fc(images)

        return images




    