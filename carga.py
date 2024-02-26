import torch
import torch.nn as nn
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def timer(funcion):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        resultado = funcion(*args, **kwargs)
        end_time = time.time()
        tiempo_ejecucion = end_time - start_time
        print(f"Tiempo de ejecución de {funcion.__name__}: {tiempo_ejecucion} segundos")
        return resultado

    return wrapper


@timer
def cargar_modelo(oracion):
    # Cargar los parámetros
    parameters = torch.load("./models/model_parameters.pth")

    # Crear una nueva instancia del modelo utilizando los parámetros guardados
    loaded_model = CNNTextClassifier(
        parameters["vocab_size"],
        parameters["embedding_dim"],
        parameters["num_filters"],
        parameters["filter_sizes"],
        parameters["hidden_dim"],
        parameters["output_dim"],
        parameters["dropout"],
    )

    # Cargar los pesos del modelo entrenado
    loaded_model.load_state_dict(torch.load("./models/model.pth"))

    # Asegurarse de que el modelo esté en el mismo dispositivo que el que estaba durante el entrenamiento
    loaded_model.to(device)

    # Definir la función de tokenización
    def simple_tokenizer(sentence):
        return sentence.split()

    # Preprocesar la oración
    tokenizer = simple_tokenizer
    word_to_idx = parameters["word_to_idx"]
    max_len = parameters["max_len"]
    label_encoder = parameters["label_encoder"]

    # Tokenizar la oración
    tokens = tokenizer(oracion)
    indexed_tokens = [word_to_idx.get(token, 0) for token in tokens]
    padded_tokens = indexed_tokens + [0] * (max_len - len(indexed_tokens))
    input_tensor = torch.tensor([padded_tokens], dtype=torch.long, device=device)

    # Hacer predicciones con el modelo cargado
    loaded_model.eval()
    with torch.no_grad():
        predictions = loaded_model(input_tensor)
    predicted_class = torch.argmax(predictions, dim=1).item()

    # Convertir la clase predicha en etiqueta original
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_label


print(cargar_modelo("hola como estas?"))

# genera una lista de oraciones
oraciones = [
    "El sol brilla en un cielo azul y despejado.",
    "Los pájaros cantan alegremente en el jardín.",
    "La lluvia cae suavemente sobre el tejado de la casa.",
    "Los niños juegan felices en el parque.",
    "El viento sopla fuerte y hace que las hojas de los árboles se muevan.",
    "El río fluye tranquilamente entre las montañas.",
    "El aroma de las flores perfuma el aire primaveral.",
    "El perro ladra emocionado al ver a su dueño llegar a casa.",
    "Las estrellas brillan en el cielo nocturno como pequeños diamantes.",
    "La nieve cubre el paisaje creando un manto blanco y frío.",
]

for i in range(len(oraciones)):
    print(cargar_modelo(oraciones[i]))
