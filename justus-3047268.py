import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Überprüfen, welches Gerät für das Training verwendet werden soll
device = (
    "cuda"
    if torch.cuda.is_available()  # Falls eine GPU verfügbar ist
    else "mps"
    if torch.backends.mps.is_available()  # Falls Apple Silicon GPU verfügbar ist
    else "cpu"  # Andernfalls die CPU verwenden
)
print(f"Verwendetes Gerät: {device}")

# Definition des neuronalen Netzwerks
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten-Schicht zum Umwandeln von 2D-Bildern in 1D-Arrays
        self.flatten = nn.Flatten()
        # Stapel von linearen Schichten mit ReLU-Aktivierung
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128 * 128 * 3, 512),  # Erste lineare Schicht, angepasst für 128x128 RGB-Bilder
            nn.ReLU(),  # ReLU-Aktivierung
            nn.Linear(512, 512),  # Zweite lineare Schicht
            nn.ReLU(),  # ReLU-Aktivierung
            nn.Linear(512, len(classes))  # Ausgabe-Schicht, Anzahl der Klassen im Datensatz
        )

    def forward(self, x):
        x = self.flatten(x)  # Daten flatten
        logits = self.linear_relu_stack(x)  # Forward-Pass durch das Netzwerk
        return logits

def main():
    # Schritt 1: Datenvorverarbeitung definieren
    transformation_anpassung = transforms.Compose(
        [transforms.Resize((128, 128)),  # Bilder auf 128 x 128 Pixel skalieren
         transforms.ToTensor(),  # Umwandeln in PyTorch-Tensoren
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Normalisierung der Pixelwerte

    # Schritt 2: Lokalen Datensatz laden
    datensatz = './data'  # Pfad zum Datensatz

    trainingssatz = datasets.ImageFolder(root=datensatz, transform=transformation_anpassung)  # Laden des Datensatzes

    # Schritt 3: DataLoader erstellen
    batch_size = 4  # Batch-Größe
    training_loader = DataLoader(trainingssatz, batch_size=batch_size, shuffle=True, num_workers=2)  # DataLoader initialisieren

    global classes
    classes = trainingssatz.classes  # Klassen des Datensatzes abrufen
    print(f"Erkannte Klassen: {classes}")

    # Modell erstellen und auf das Gerät verschieben
    model = NeuralNetwork().to(device)
    print(model)  # Struktur des Modells ausgeben

    # Testen: Ein Batch von Daten abrufen
    data_iter = iter(training_loader)
    images, labels = next(data_iter)
    print(f"Batch-Größe: {images.shape}, Labels: {labels}")

    # Forward-Pass durch das Modell
    images = images.to(device)  # Bilder auf das Gerät verschieben
    logits = model(images)  # Vorhersagen des Modells
    pred_probab = nn.Softmax(dim=1)(logits)  # Wahrscheinlichkeiten berechnen
    y_pred = pred_probab.argmax(1)  # Vorhergesagte Klassen
    print(f"Vorhergesagte Klassen: {y_pred}")

if __name__ == "__main__":
    main()
