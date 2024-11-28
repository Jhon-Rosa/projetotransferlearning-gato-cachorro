"# projetotransferlearning-gato-cachorro" 
# Parte 01

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



# Parte 02

import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import os

# Caminho do checkpoint salvo
checkpoint_path = "/content/drive/MyDrive/gato-cachorro/model_checkpoint.pth"

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função para carregar o modelo
def carregar_modelo(checkpoint_path):
    # Inicializar o modelo
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Ajustar para 2 classes
    # Carregar os pesos do checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Modo de avaliação
    print("Modelo carregado com sucesso!")
    return model

# Função para processar uma imagem
def processar_imagem(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # Adiciona batch dimension
    return input_tensor

# Função para realizar a previsão
def prever_imagem(model, image_path, class_names):
    input_tensor = processar_imagem(image_path)
    outputs = model(input_tensor)
    _, predicted_class = outputs.max(1)  # Índice da classe prevista
    return class_names[predicted_class.item()]

# Testar o modelo em uma única imagem
def testar_imagem_unica(model, image_path, class_names):
    classe = prever_imagem(model, image_path, class_names)
    print(f"Imagem: {image_path}, Classe prevista: {classe}")


# Main
if __name__ == "__main__":
    # Montar o Google Drive
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

    # Diretório das novas imagens para teste
    test_dir = "/content/drive/MyDrive/gato-cachorro/gato-cachorro/Teste/teste1.jpg"  # Atualize conforme necessário
    image_path_teste = "/content/drive/MyDrive/gato-cachorro/gato-cachorro/Teste/teste1.jpg"  # Substitua por uma imagem real

    # Diretório do conjunto de treino para obter os nomes das classes
    train_dir = "/content/drive/MyDrive/gato-cachorro/gato-cachorro"
    train_dataset = ImageFolder(train_dir)
    class_names = train_dataset.classes  # Exemplo: ['Cat', 'Dog']

    # Carregar o modelo
    model = carregar_modelo(checkpoint_path)

    # Testar uma única imagem
    print("\nTestando uma única imagem:")
    testar_imagem_unica(model, image_path_teste, class_names)


from PIL import Image

# Recarregar a imagem para garantir que a versão mais recente seja carregada
imagem = Image.open(image_path_teste).convert('RGB')
imagem.show()  # Mostrar a imagem carregada


# Parte 03

import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import os

# Caminho do checkpoint salvo
checkpoint_path = "/content/drive/MyDrive/gato-cachorro/model_checkpoint.pth"

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função para carregar o modelo
def carregar_modelo(checkpoint_path):
    # Inicializar o modelo
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Ajustar para 2 classes
    # Carregar os pesos do checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Modo de avaliação
    print("Modelo carregado com sucesso!")
    return model

# Função para processar uma imagem
def processar_imagem(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # Adiciona batch dimension
    return input_tensor

# Função para realizar a previsão
def prever_imagem(model, image_path, class_names):
    input_tensor = processar_imagem(image_path)
    outputs = model(input_tensor)
    _, predicted_class = outputs.max(1)  # Índice da classe prevista
    return class_names[predicted_class.item()]

# Testar o modelo em uma única imagem
def testar_imagem_unica(model, image_path, class_names):
    classe = prever_imagem(model, image_path, class_names)
    print(f"Imagem: {image_path}, Classe prevista: {classe}")


# Main
if __name__ == "__main__":
    # Montar o Google Drive
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

    # Diretório das novas imagens para teste
    test_dir = "/content/drive/MyDrive/gato-cachorro/gato-cachorro/Teste/teste1.jpg"  # Atualize conforme necessário
    image_path_teste = "/content/drive/MyDrive/gato-cachorro/gato-cachorro/Teste/teste1.jpg"  # Substitua por uma imagem real

    # Diretório do conjunto de treino para obter os nomes das classes
    train_dir = "/content/drive/MyDrive/gato-cachorro/gato-cachorro"
    train_dataset = ImageFolder(train_dir)
    class_names = train_dataset.classes  # Exemplo: ['Cat', 'Dog']

    # Carregar o modelo
    model = carregar_modelo(checkpoint_path)

    # Testar uma única imagem
    print("\nTestando uma única imagem:")
    testar_imagem_unica(model, image_path_teste, class_names)


from PIL import Image

# Recarregar a imagem para garantir que a versão mais recente seja carregada
imagem = Image.open(image_path_teste).convert('RGB')
imagem.show()  # Mostrar a imagem carregada


# Parte 04

from IPython.display import Image, display
import torch
from PIL import Image as PILImage

# Caminho para a imagem de teste
image_path = "/content/drive/MyDrive/gato-cachorro/gato-cachorro/Teste/teste1.jpg"

# Baixar o modelo YOLOv5 pré-treinado (se necessário)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Detectar objetos
results = model(image_path)

# Exibir a imagem com a detecção diretamente sem salvar no disco
results.show()  # Exibe a imagem com a detecção

# Para carregar e exibir a imagem original (com as detecções):
imagem = PILImage.open(image_path).convert('RGB')
imagem.show()  # Mostrar a imagem carregada com as detecções
