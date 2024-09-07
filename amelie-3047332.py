import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split

epoch_size = 3
batch_size = 4
# Festlegen, welches Gerät für das Training verwendet werden soll
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def imshow(img):
    img = img / 2 + 0.5  # Denormalisieren
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Definiere ein benutzerdefiniertes Dataset, um beschädigte Bilder zu überspringen
class FilteredDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.valid_samples = []
        for index in range(len(self.dataset)):
            try:
                self.valid_samples.append(self.dataset[index])
            except (UnidentifiedImageError, OSError):
                print(f"Überspringe beschädigtes Bild: {self.dataset.samples[index][0]}")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        return self.valid_samples[idx]


def split_dataset(dataset, train_size):
    total_size = len(dataset)
    train_size = int(train_size * total_size)
    val_size = total_size - train_size
    # `random_split` direkt auf den Datensatz anwenden
    train_set, val_set = random_split(dataset, [train_size, val_size])
    return train_set, val_set


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128 * 128 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def main():
    print(f"Verwendetes Gerät: {device}")

    # Schritt 1: Datenvorverarbeitung definieren
    transformation_anpassung = transforms.Compose(
        [transforms.Resize((128, 128)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Schritt 2: Trainings- und Testdatensätze laden
    datensatz = './data'
    # Verwende ImageFolder, um die Rohdaten zu laden
    trainingssatz = datasets.ImageFolder(root=datensatz, transform=transformation_anpassung)

    # Erstelle das gefilterte Dataset, das beschädigte Bilder überspringt
    gefiltertes_dataset = FilteredDataset(trainingssatz)

    # Erstelle das Trainingsset
    total_size = len(gefiltertes_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    # `random_split` verwenden, um Datensätze aufzuteilen
    train_set, test_set = random_split(gefiltertes_dataset, [train_size, test_size])

    # Erstelle DataLoader für Training und Testen
    training_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    global classes
    classes = trainingssatz.classes
    print(f"Erkannte Klassen: {classes}")

    # Modell erstellen und auf das Gerät verschieben
    model = NeuralNetwork().to(device)
    print(model)

    # Loss function und Optimizer definieren
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Trainingszyklus
    for e in range(epoch_size):
        running_loss = 0.0
        for i, data in enumerate(training_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch_size + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Training abgeschlossen')

    # Modell speichern
    path = './datensatz.pth'
    torch.save(model.state_dict(), path)

    # Testen des Modells
    print("Modell wird nun getestet")
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    # Modell laden
    model.load_state_dict(torch.load(path))
    outputs = model(images.to(device))

    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

    # Netzwerkleistung auf dem gesamten Datensatz prüfen
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).float().sum().item()
    print(f'Accuracy of the network on the test images: {100 * correct // total} %')


if __name__ == "__main__":
    main()

