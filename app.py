# import torch
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
# from torchtext.datasets import TabularDataset
# from torch.utils.data import DataLoader
# import random

# # Definimos el tokenizador
# tokenizer = get_tokenizer("spacy", language="es")

# # Cargamos los datos
# dataset = TabularDataset(
#     path="ruta/a/tu/dataset.csv",
#     format="csv",
#     fields=[("Oracion", TEXT), ("Clave", LABEL)],
# )

# # Dividimos los datos en conjuntos de entrenamiento y prueba
# train_data, test_data = dataset.split(random_state=random.seed(1234))


# # Construimos el vocabulario
# def yield_tokens(data_iter):
#     for text, _ in data_iter:
#         yield tokenizer(text)


# vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>"])
# vocab.set_default_index(vocab["<unk>"])


# # Creamos los iteradores
# def collate_batch(batch):
#     label_list, text_list = [], []
#     for _label, _text in batch:
#         label_list.append(label_pipeline(_label))
#         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
#         text_list.append(processed_text)
#     return label_list, text_list


# train_iterator = DataLoader(
#     train_data, batch_size=8, shuffle=False, collate_fn=collate_batch
# )

# # Aquí puedes definir tu modelo, entrenarlo y evaluarlo utilizando los iteradores

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Función de tokenización
def simple_tokenizer(sentence):
    return sentence.split()


# Función de preprocesamiento de datos
def preprocess_data(data):
    # Preprocesamiento de los datos
    X = data["Oraciones"]
    y = data["Valor"]

    # Codificar las etiquetas
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y, label_encoder


# Clase Dataset para manejar los datos
class TextDataset(Dataset):
    def __init__(self, X, y, tokenizer, word_to_idx, max_len):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sentence = self.X.iloc[idx]
        tokens = self.tokenizer(sentence)
        indexed_tokens = [self.word_to_idx.get(token, 0) for token in tokens]
        padded_tokens = indexed_tokens + [0] * (self.max_len - len(indexed_tokens))
        return {
            "tokens": torch.tensor(padded_tokens, dtype=torch.long),
            "label": torch.tensor(self.y[idx], dtype=torch.long),
        }


class CNNTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_filters,
        filter_sizes,
        hidden_dim,
        output_dim,
        dropout,
    ):
        super(CNNTextClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # Capa de embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Capas de convolución
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs
                )
                for fs in filter_sizes
            ]
        )

        # Capa completamente conectada
        self.fc = nn.Linear(len(filter_sizes) * num_filters, hidden_dim)

        # Capa de salida
        self.output = nn.Linear(hidden_dim, output_dim)

        # Capa de dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        fc_output = torch.relu(self.fc(cat))
        output = self.output(fc_output)
        return output

    def save_parameters(self, filepath, word_to_idx, max_len, label_encoder):
        parameters = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_filters": self.num_filters,
            "filter_sizes": self.filter_sizes,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout.p,
            "word_to_idx": word_to_idx,
            "max_len": max_len,
            "label_encoder": label_encoder,
        }
        torch.save(parameters, filepath)


# Parámetros del modelo
def train_model(model, train_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            tokens, labels = batch["tokens"].to(device), batch["label"].to(device)
            output = model(tokens)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}"
        )


# Evaluación del modelo
def evaluate_model(model, test_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            tokens, labels = batch["tokens"].to(device), batch["label"].to(device)
            output = model(tokens)
            predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")


# Cargar el dataset desde el archivo Excel
data = pd.read_excel("dataset.xlsx")

# Preprocesamiento de datos
X, y, label_encoder = preprocess_data(data)

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tokenización y padding de las secuencias
tokenizer = simple_tokenizer
X_train_tokens = [tokenizer(sentence) for sentence in X_train]
X_test_tokens = [tokenizer(sentence) for sentence in X_test]
max_len = max(len(tokens) for tokens in X_train_tokens)

# Crear un vocabulario a partir de los tokens en el conjunto de entrenamiento
vocab = set(token for tokens in X_train_tokens for token in tokens)
word_to_idx = {word: idx + 1 for idx, word in enumerate(vocab)}
word_to_idx["<pad>"] = 0

# Convertir a tensores de PyTorch y mover a GPU si está disponible
X_train_indices = [
    [word_to_idx.get(token, 0) for token in tokens] for tokens in X_train_tokens
]
X_test_indices = [
    [word_to_idx.get(token, 0) for token in tokens] for tokens in X_test_tokens
]
X_train_padded = [
    sequence + [0] * (max_len - len(sequence)) for sequence in X_train_indices
]
X_test_padded = [
    sequence + [0] * (max_len - len(sequence)) for sequence in X_test_indices
]
X_train_tensor = torch.tensor(X_train_padded, dtype=torch.long, device=device)
X_test_tensor = torch.tensor(X_test_padded, dtype=torch.long, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

# Crear conjuntos de datos y cargadores de datos
train_dataset = TextDataset(X_train, y_train, simple_tokenizer, word_to_idx, max_len)
test_dataset = TextDataset(X_test, y_test, simple_tokenizer, word_to_idx, max_len)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Definir los parámetros del modelo
vocab_size = len(word_to_idx)
embedding_dim = 100
num_filters = 100
filter_sizes = [3, 4, 5]
hidden_dim = 100
output_dim = len(label_encoder.classes_)
dropout = 0.5

# Inicializar y entrenar el modelo en CUDA si está disponible
model = CNNTextClassifier(
    vocab_size,
    embedding_dim,
    num_filters,
    filter_sizes,
    hidden_dim,
    output_dim,
    dropout,
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenar el modelo
train_model(model, train_loader, optimizer, criterion, num_epochs=10)

# Evaluar el modelo
evaluate_model(model, test_loader)

# Guardar el modelo y los parámetros
model.save_parameters(
    "./models/model_parameters.pth", word_to_idx, max_len, label_encoder
)
torch.save(model.state_dict(), "./models/model.pth")
