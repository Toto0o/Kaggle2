from KNN.dataset import DataSet as DataSet_KNN
from CNN.dataset import DataSet as DataSet_CNN
from KNN.KNN import KNN
from CNN.CNN import CNN
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys


if __name__ == "__main__" :

    def example_usage() :
        print("Exemple d'utilisation : ")
        print("\t - Pour générer le fichier csv du premier jalon, executez : retinal_classification.py knn \n")
        print("\t - Pour générer le fichier csv du deuxième jalon, executez : retinal_classification.py cnn \n")
    
    path = "train_data.pkl"
    test_path = "test_data.pkl"

    mode = sys.argv[1].lower()

    if (mode == 'knn') :
        data_set: DataSet_KNN = DataSet_KNN.from_pickle(path)
        data_set.shuffle()
        data_set.normalize()
        data_set.flatten()
        data_set.set_test_kaggle_data(test_path)
        data_set.make_csv(KNN, k=152, p=22)
    
    elif (mode == 'cnn') :
        
        with open(path, "rb") as f:
            train_data = pickle.load(f)
        images = train_data['images']
        labels = train_data['labels']

        with open(test_path, "rb") as f2 :
            test_data = pickle.load(f2)
        test_images = test_data['images']

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
        ])

        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])


        images_train = DataSet_CNN.preprocess_images(images)
        images_test = DataSet_CNN.preprocess_images(test_images)

        labels = np.array(labels).astype(int).reshape(-1)

        images_train, images_val, labels_train, labels_val = DataSet_CNN.stratified_split(images_train, labels)

        train_data = DataSet_CNN(images_train, labels_train, transform=train_transform)
        val_data = DataSet_CNN(images_val, labels_val, transform=val_transform)
        test_data = DataSet_CNN(images_test, transform=val_transform)

        class_count = np.bincount(labels_train.astype(int))

        # class_count = np.bincount(labels_train.astype(int))
        class_weights = 1.0 / class_count

        #sampler
        sample_weights = class_weights[labels_train.astype(int)]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        #dataloader
        train_loader = DataLoader(train_data, batch_size=32, sampler=sampler)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # CONFIG 1 
        #  - dropout = 0.3
        #  - weight_decay = 1e-4
        #  - lr = 1e-3

        # CONFIG 2 
        #  - dropout = 0.5
        #  - weight_decay = 1e-2
        #  - lr = 3e-3

        # model : CNN
        model = CNN(
            n_classes=5,
            in_channels=3,
            dropout=0.3
        )

        #loss function
        class_weight_tensor = torch.FloatTensor(class_weights / class_weights.sum()).to(device)
        criterion = nn.CrossEntropyLoss()


        optimizer = optim.AdamW(
            model.parameters(),
            weight_decay= 1e-4,
            lr = 1e-3
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=0.5,
            # verbose=True
        )

        start_time = time.time()

        model, history = train_data.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=100,
            device=device,
            patience=15
        )

        model.eval()
        all_preds = []

        with torch.no_grad():
            for images in test_loader: 
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)

        all_preds = np.array(all_preds)

        print("Final test \n")
        DataSet_CNN.write_preds_to_csv(preds=all_preds)

        training_time = time.time() - start_time

        print(f"\n" + "="*60)
        print(f"✅ ENTRAÎNEMENT TERMINÉ")
        print(f"="*60)
        print(f"   Temps total     : {training_time/60:.1f} minutes")
        print(f"   Meilleure val   : {max(history['val_acc']):.4f}")
        print(f"   Epochs effectués: {len(history['train_loss'])}")
    
    else :
        example_usage()

        


    """

    Le script ci-dessous à été utilisé pour otpimiser le modèle KNN. Pour obtenir le
    fichier CSV utilisé pour  le concours Kaggle, executez le code retinal_classification.py knn

    """

    

    # K = [151,152,153]
    # P = [21,22,23]
    # best = (0,0,0)
    # time_0 = time.time()
    # for k in K :
    #     for p in P :
    #         accuracy = data_set.evaluate(KNN, k=k, p=p)
    #         print(accuracy, f"p={p}, k={k}")
    #         if (accuracy > best[0]) :
    #             best = (accuracy, k, p)
    # delta_time = time.time() - time_0
    
    # print(f"Best KNN model : accuracy: {best[0]}, k={best[1]}, p={best[2]}" )

    # print("time training : ", delta_time)

    """
    
    Décommenter les lignes ci-desous pour effectuer  le code pour cnn
    
    """


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomRotation(degrees=10),
    #     transforms.ColorJitter(brightness=0.15, contrast=0.15),
    #     transforms.ToTensor(),
    # ])

    # val_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.ToTensor(),
    # ])

    # with open(path, "rb") as f:
    #     train_data = pickle.load(f)
    # images = train_data['images']
    # labels = train_data['labels']

    # with open(test_path, "rb") as f2 :
    #     test_data = pickle.load(f2)
    # test_images = test_data['images']


    # images_train = DataSet_CNN.preprocess_images(images)
    # images_test = DataSet_CNN.preprocess_images(test_images)

    # labels = np.array(labels).astype(int).reshape(-1)

    # images_train, images_val, labels_train, labels_val = DataSet_CNN.stratified_split(images_train, labels)

    # train_data = DataSet_CNN(images_train, labels_train, transform=train_transform)
    # val_data = DataSet_CNN(images_val, labels_val, transform=val_transform)
    # test_data = DataSet_CNN(images_test, transform=val_transform)
   
    # class_count = np.bincount(labels_train.astype(int))

    # # class_count = np.bincount(labels_train.astype(int))
    # class_weights = 1.0 / class_count

    # #sampler
    # sample_weights = class_weights[labels_train.astype(int)]
    # sampler = WeightedRandomSampler(
    #     weights=sample_weights,
    #     num_samples=len(sample_weights),
    #     replacement=True
    # )

    # #dataloader
    # train_loader = DataLoader(train_data, batch_size=32, sampler=sampler)
    # val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    # test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # # CONFIG 1 
    # #  - dropout = 0.3
    # #  - weight_decay = 1e-4
    # #  - lr = 1e-3

    # # CONFIG 2 
    # #  - dropout = 0.5
    # #  - weight_decay = 1e-2
    # #  - lr = 3e-3

    # # model : CNN
    # model = CNN(
    #     n_classes=5,
    #     in_channels=3,
    #     dropout=0.3
    # )

    # #loss function
    # class_weight_tensor = torch.FloatTensor(class_weights / class_weights.sum()).to(device)
    # criterion = nn.CrossEntropyLoss()


    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     weight_decay= 1e-4,
    #     lr = 1e-3
    # )

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     patience=5,
    #     factor=0.5,
    #     # verbose=True
    # )

    # start_time = time.time()

    # model, history = train_data.train_model(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     epochs=100,
    #     device=device,
    #     patience=15
    # )

    # model.eval()
    # all_preds = []

    # with torch.no_grad():
    #     for images in test_loader:      # test_loader = DataSet_CNN(..., labels=None)
    #         images = images.to(device)
    #         outputs = model(images)
    #         preds = outputs.argmax(dim=1).cpu().numpy()
    #         all_preds.extend(preds)

    # all_preds = np.array(all_preds)

    # print("Final test \n")
    # DataSet_CNN.write_preds_to_csv(preds=all_preds)

    # training_time = time.time() - start_time

    # print(f"\n" + "="*60)
    # print(f"✅ ENTRAÎNEMENT TERMINÉ")
    # print(f"="*60)
    # print(f"   Temps total     : {training_time/60:.1f} minutes")
    # print(f"   Meilleure val   : {max(history['val_acc']):.4f}")
    # print(f"   Epochs effectués: {len(history['train_loss'])}")








