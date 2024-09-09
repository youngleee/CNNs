import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from PIL import UnidentifiedImageError
from torch.utils.data import random_split

# Epochen - und Batchanzahl festlegen
epoch_size = 10
batch_size = 16

# Festlegen, welches Gerät für das Training verwendet werden soll
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# beschädigte Bilder werden beim Laden des Datensatzes herausgefiltert
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


# Erstellen eines Convolutional Neural Networks (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# main-Funktion
def main():
    print(f"Verwendetes Gerät: {device}")

    # Daten in das Netzwerk einladen und transformieren
    transformation_anpassung = transforms.Compose(
        [transforms.Resize((128, 128)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # normalisieren, um die Trainingsleistung zu verbessern
    )

    datensatz = './data'
    trainingssatz = datasets.ImageFolder(root=datensatz, transform=transformation_anpassung)

    # gefilterter Datensatz, der beschädigte Bilder überspringt
    gefiltertes_dataset = FilteredDataset(trainingssatz)

    # Trainings - und Testset
    train_size = int(0.8 * len(gefiltertes_dataset))
    test_size = len(gefiltertes_dataset) - train_size
    print(train_size, test_size)

    # random_split sortiert zufällig die Daten in die Trainings/Testsets
    train_set, test_set = random_split(gefiltertes_dataset, [train_size, test_size])

    # Erstelle DataLoader für Training und Testen
    training_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Klassen des Datensatzes definieren
    classes = trainingssatz.classes
    print(f"Erkannte Klassen: {classes}")

    # Modell erstellen und auf das Gerät verschieben
    model = CNN().to(device)
    print(model)

    # Verlustfunktion und Optimizer definieren
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
            if i % 1000 == 999:
                print(f'[{epoch_size + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
                running_loss = 0.0

    print('Training abgeschlossen')

    # Modell speichern
    path = './datensatz.pth'
    torch.save(model.state_dict(), path)

    # Testen des Modells
    print("Modell wird nun getestet")
    data_test = iter(test_loader)
    images, labels = next(data_test)

    # Modell laden
    model.load_state_dict(torch.load(path))
    outputs = model(images.to(device))

    # Vorhersagen treffen
    predicted = torch.round(torch.sigmoid(outputs)).squeeze()
    print('Predicted: ', ' '.join(f'{classes[int(predicted[j])]:5s}' for j in range(4)))

    # Netzwerkleistung auf dem gesamten Datensatz prüfen
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.round(torch.sigmoid(outputs)).squeeze()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Genauigkeit des Modells auf Basis von Beispielbildern {100 * correct / total:.2f} %')


if __name__ == "__main__":
    main()


