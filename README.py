"# projetotransferlearning-gato-cachorro" 
import torch
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Definir os diretórios de treino
train_dir = "/content/drive/MyDrive/gato-cachorro/gato-cachorro"  # Atualize conforme seu caminho no Google Drive

# Definir transformações (redimensionamento, normalização)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Carregar dataset
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Usando o modelo pré-treinado ResNet50
model = models.resnet50(weights='IMAGENET1K_V1')  # Carregando pesos pré-treinados
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Ajustando a última camada para 2 classes (gato e cachorro)

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Configurar critério de perda e otimizador
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Caminho do checkpoint deve ser um arquivo
checkpoint_path = "/content/drive/MyDrive/gato-cachorro/model_checkpoint.pth"

# Função para salvar o checkpoint
def save_checkpoint(epoch, model, optimizer, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint salvo após a época {epoch+1}")

# Função para carregar o checkpoint
def load_checkpoint(model, optimizer):
    if os.path.isfile(checkpoint_path):  # Confirma que é um arquivo
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Retomando treinamento da época {epoch+1}")
        return epoch
    else:
        print(f"Checkpoint não encontrado no caminho: {checkpoint_path}")
        return 0


# Carregar checkpoint se existir
epoch_start = load_checkpoint(model, optimizer)

# Número de épocas para o treinamento
num_epochs = 2

# Treinamento
for epoch in range(epoch_start, num_epochs):
    model.train()  # Coloca o modelo em modo de treinamento
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zera os gradientes acumulados
        outputs = model(inputs)  # Faz a previsão
        loss = criterion(outputs, labels)  # Calcula a perda
        loss.backward()  # Backpropagation
        optimizer.step()  # Atualiza os pesos

        running_loss += loss.item()  # Acumula a perda

        # Exibir progresso a cada 10 lotes (batches)
        if i % 10 == 0:
            print(f"Época {epoch+1}/{num_epochs}, Lote {i}/{len(train_loader)}, Perda: {running_loss / (i + 1):.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"Época {epoch+1}/{num_epochs}, Perda média: {avg_loss:.4f}")

    # Salvar checkpoint após cada época
    save_checkpoint(epoch, model, optimizer, avg_loss)

print("Treinamento concluído!")
