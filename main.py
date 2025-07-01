# %%
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
import torch.optim as optim 
from imblearn.under_sampling import RandomUnderSampler
import kornia.augmentation as K
import matplotlib.pyplot as plt
import os

# %%
print("codigo começou")
limpar_imagens = False
usar_imagem_minima = True
aumentar_dados = True
vezes = 3

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
lr = 0.01
# %%

def salvar_graficos_erros_por_classe(erros, pasta="graficos_classes"):
    os.makedirs(pasta, exist_ok=True)
    epocas = np.arange(erros.shape[0])

    for classe in range(erros.shape[1]):
        plt.figure(figsize=(8, 5))
        plt.plot(epocas, erros[:, classe], marker='o', label=f"Classe {classe}")
        plt.xlabel("Época")
        plt.ylabel("Quantidade de Erros")
        plt.title(f"Variação de Erros - Classe {classe}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{pasta}/classe_{classe}_erro_por_epoca.png")
        plt.close()

def salvar_grafico_barras_erro_total(erros, nome_arquivo="erro_total_por_classe.png"):
    erro_total_por_classe = erros.sum(axis=0)
    classes = np.arange(len(erro_total_por_classe))

    plt.figure(figsize=(10, 6))
    plt.bar(classes, erro_total_por_classe, color='steelblue')
    plt.xlabel("Classe")
    plt.ylabel("Erro Total")
    plt.title("Erro Total por Classe ao longo das Épocas")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(nome_arquivo)
    plt.close()

def salvar_graficos_chute_por_classe(chutes, pasta="chutes_no_teste"):
    os.makedirs(pasta, exist_ok=True)
    classes = np.arange(chutes.shape[0])

    for classe in range(chutes.shape[1]):
        plt.figure(figsize=(8, 5))
        plt.plot(classes, chutes[:, classe], marker='o', label=f"Classe {classe}")
        plt.xlabel("Classes")
        plt.ylabel("Quantidade de chutes")
        plt.title(f"Variação de chutes - Classe {classe}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{pasta}/classe_{classe}_chutes_por_classe.png")
        plt.close()


def gerar_dados(tensor, labels, aumentar_dados, vezes):
    if aumentar_dados:
        tensor_aux = tensor
        labels_aux = labels
        for i in range(vezes):
            augmentation_pipeline = K.AugmentationSequential(
            K.RandomRotation(degrees=(-20, 20), p=0.8),
            K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.7),
            K.RandomHorizontalFlip(p=0.5),
            data_keys=['input']
            )
            tensor_aumentado = augmentation_pipeline(tensor)
            tensor_aux = torch.cat((tensor_aux, tensor_aumentado), dim=0)
            labels_aux = torch.cat((labels_aux, labels), dim=0)
        tensor = tensor_aux
        labels = labels_aux
        #print(tensor.shape)


    x_treino, x_teste, y_treino, y_teste = train_test_split(
        tensor, labels, test_size=0.3, random_state=1, stratify=labels
    )

    treino_dataset = TensorDataset(x_treino, y_treino)
    teste_dataset = TensorDataset(x_teste, y_teste)

    batch_size = 32
    trainloader = DataLoader(dataset=treino_dataset, batch_size=batch_size,shuffle=True)
    testloader = DataLoader(dataset=teste_dataset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader, tensor, labels

# %%
class Net(nn.Module):
    def __init__(self, entrada):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(3250, 512)

        self.fc2 = nn.Linear(512, 25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 3250)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

# %%
def train(model, device, train_loader, optimizer, loss_fn, i):
    loss = 0
    for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)      
            optimizer.zero_grad()                            
            output = model(data)                               
            loss = loss_fn(output, target)                   
            loss.backward()                                 
            optimizer.step()

    print(f'Epoch {i} TRAIN: Last Loss: {loss:.5f}')

# %%
def test(model, device, test_loader, epoca):
    model.eval()
    all_predictions = []
    all_targets = []
    correct = 0

    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):            # intera sobre cada dado e seu rotulo, _ descarta os index que não serão usados na função
            data, target = data.to(device), target.to(device)       # transfere os dados e os rotulos para o dispositivo escolhido
            outputs = model(data)                                   # gera as predictions
            loss = F.nll_loss(outputs, target).item()               # calcula o erro
            pred = outputs.argmax(1, keepdim=True)                  #
            resposta = pred.eq(target.view_as(pred)).sum().item()
            for i in range(len(target.cpu().detach().numpy())):
                chutes[target[i]][pred[i][0]] += 1
                if(target[i] != pred[i][0]):
                    erros[epoca][target[i]] +=1

            correct += resposta   # calcula o numero predições corretas

            all_predictions.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())

        total_samples = len(test_loader.dataset)
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted', zero_division=1)
        accuracy = (correct / total_samples) * 100

        with open(f"Relatorio do Treinamento.txt", 'a') as file:
            file.write(f'Last Loss: {loss:.4f} | Accuracy: {accuracy:.2f}% | Correct: {correct}/{total_samples} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1_score:.4f}\n')
        print(f'Last Loss: {loss:.4f} | Accuracy: {accuracy:.2f}% | Correct: {correct}/{total_samples} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1_score:.4f}')


# %%
caminho_conjuntos = Path(f'conjuntos')
caminhos_jpg = list(caminho_conjuntos.rglob('*.jpg'))
caminhos_jpeg = list(caminho_conjuntos.rglob('*.jpeg'))
caminhos_imagens = caminhos_jpg + caminhos_jpeg
#print(f'encontrados {len(caminhos_imagens)}')

largura = 32
altura = 64
tamanho_alvo = (largura, altura)

lista_imagens = []
lista_rotulos = []
i = 0
rotuloAnterior = -1
for caminho_da_imagem in caminhos_imagens:
    imagem = Image.open(caminho_da_imagem)
    imagem = imagem.convert('L')
    imagem_redimensionada = imagem.resize(tamanho_alvo, Image.Resampling.LANCZOS)
    
    if limpar_imagens:
        imagem_redimensionada = imagem_redimensionada.point(lambda v: 0 if v < 90 else 255)


    array_imagem = np.array(imagem_redimensionada)
    lista_imagens.append(array_imagem)

    rotulo = caminho_da_imagem.parent.name
    lista_rotulos.append(int(rotulo))
    # if rotuloAnterior != rotulo and i<10:
    #     i+=1
    #     rotuloAnterior = rotulo
    #     imagem_redimensionada.show()
    #     print(rotulo)

    # print(f'processado: {caminho_da_imagem}')
    # print(f'rotulo: {caminho_da_imagem.parent.name}')
    #print(f'array imagem tipo: {array_imagem.shape}')

print()
# preparação de dados para treino
# transpor imagnes
lista_de_imagens_t = []
lista_tensores = []
#for img in lista_imagens:
    #lista_de_imagens_t.append(img.transpose((2,0,1)))
    #print(f'transposta: {lista_de_imagens_t[0].shape}')  # deu certo
    
for img in lista_imagens:
    lista_tensores.append(torch.from_numpy(img))
    #print(f"Tipo de dado do tensor após 'from_numpy': {lista_tensores[0].dtype}")

lista_tensores = [img.float()/255.0 for img in lista_tensores]
#print(f'tensorfinal: {lista_tensores[0].shape}') # formato [3,64,32] dado: float32

tensor = torch.stack(lista_tensores, dim=0) # empilha as dimensões
#print(tensor.shape)

#print(f"Distribuição original das classes: {Counter(lista_rotulos)}")
numero_amostras = tensor.shape[0]
#print(numero_amostras)

# %%
if usar_imagem_minima:
    tensor_achatado = tensor.reshape((numero_amostras, altura*largura))
    rus = RandomUnderSampler(random_state=1)
    tensor_balanceado, lista_rotulos = rus.fit_resample(tensor_achatado, lista_rotulos)
    #print(f"Distribuição após undersampling: {Counter(labels_balanceadas)}")
    tensor = torch.from_numpy(tensor_balanceado.reshape((-1, altura, largura)))
    #print(f"Formato do tensor balanceado: {tensor.shape}")
    #print(type(tensor))

labels = torch.tensor(lista_rotulos, dtype=torch.long)
#print(labels)
tensor = tensor.unsqueeze(1)
#print(f'formato tensor {tensor.shape}')



# x_treino, x_teste, y_treino, y_teste = train_test_split(
#     tensor,
#     labels_balanceadas,
#     test_size=0.3,
#     random_state=1,
#     stratify=labels_balanceadas
# )

# %%
modelo = Net(largura*altura).to(device=device)
optimizer = optim.SGD(modelo.parameters(), lr=lr)
loss = F.nll_loss

trainloader, testloader, tensor, labels = gerar_dados(tensor, labels, aumentar_dados, vezes)

EPOCA = 50

erros = np.zeros((EPOCA, 25))
chutes = np.zeros((25, 25))
print("vai inicar treinamento\n")
for epoca in range(EPOCA):
    train(modelo, device, trainloader, optimizer, loss,epoca)
    test(modelo, device, testloader, epoca)

    
with open("erros_por_classe.txt", "w") as nota:
    for i in range(EPOCA):
        # Agora str(erros[i]) vai gerar uma única linha, sem quebras internas
        linha = f"epoca {i-1}: {str(erros[i])}\n" # i+2 para bater com seu exemplo
        nota.write(linha)

    erros_classe = [int(erros[:,i].sum()) for i in range(25)]
    nota.write("\nclasses / erros\n")
    for i in range(25):
        nota.write(f"{i}: {erros_classe[i]}\n")

salvar_graficos_erros_por_classe(erros)
salvar_grafico_barras_erro_total(erros)

salvar_graficos_chute_por_classe(chutes)