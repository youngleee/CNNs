# erstes import statement aus dem pytorch tutorial
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np


# wir brauchen mainly pytorch vision da wir auch mit jpg dateien arbeiten
# statt die Daten direkt von kaggle.com runterzuladen haben wir die Daten schon lokal und können darauf zugreifen
# methode mit main wird wohl gemacht weil ich sonst immer nen runtime error hatte und das so funktioniert

def main():
    # Schritt 1: Transformations definieren (hab ich aus jupyter notebook über CNNs abgeleitet)
    transformation_anpassung = transforms.Compose(
        [transforms.Resize((128, 128)),  # Bilder auf 128 x 128 Pixel
         transforms.ToTensor(),  # PyTorch Tensors
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Normalisiert auf [-1, 1] → beschreibt die Pixelwerte von dunkel bis hell; bessere Performance

    # Schritt 2: lokales Dateset laden
    datensatz = './data'  # relativer Pfad

    # Verwende ImageFolder, um die Daten zu laden
    trainingssatz = datasets.ImageFolder(root=datensatz, transform=transformation_anpassung)

    # Schritt 3: DataLoader erstellen
    batch_size = 4

    training_loader = DataLoader(trainingssatz, batch_size=batch_size, shuffle=True, num_workers=2)

    classes = trainingssatz.classes
    print(f"Erkannte Klassen: {classes}")

    # Testen: Ein paar Batch-Daten abrufen
    data_iter = iter(training_loader)
    images, labels = next(data_iter)
    print(f"Batch size: {images.shape}, Labels: {labels}")


if __name__ == "__main__":
    main()

