import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import TabularDataset
from torch.utils.data import DataLoader
import random

# Definimos el tokenizador
tokenizer = get_tokenizer("spacy", language="es")

# Cargamos los datos
dataset = TabularDataset(
    path="ruta/a/tu/dataset.csv",
    format="csv",
    fields=[("Oracion", TEXT), ("Clave", LABEL)],
)

# Dividimos los datos en conjuntos de entrenamiento y prueba
train_data, test_data = dataset.split(random_state=random.seed(1234))


# Construimos el vocabulario
def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


# Creamos los iteradores
def collate_batch(batch):
    label_list, text_list = [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    return label_list, text_list


train_iterator = DataLoader(
    train_data, batch_size=8, shuffle=False, collate_fn=collate_batch
)

# Aquí puedes definir tu modelo, entrenarlo y evaluarlo utilizando los iteradores
