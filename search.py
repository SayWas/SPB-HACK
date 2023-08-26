import time
from typing import List

from sklearn.metrics.pairwise import paired_cosine_distances
import seaborn as sns

from config import TRAIN_EX, TEST_EX, EPOCHS, TEST_PATH
from dataset import load_dataset, create_pair_data, get_corpus
from sentence_transformers import SentenceTransformer, InputExample, evaluation, losses, util
import torch
from torch.utils.data import DataLoader

from exceptions import NotFoundException
from schemas import Address


def get_labels(path: str, ex: int):
    data = load_dataset(path)
    label_1, label_0 = create_pair_data(text=data['target_address'],
                                        labels=data['target_building_id'], size=ex)
    return label_1, label_0


def get_data_loader(path: str, ex: int):
    label_1, label_0 = get_labels(path, ex)

    examples = [InputExample(texts=x, label=1.0) for x in label_1] + \
               [InputExample(texts=x, label=0.0) for x in label_0]

    dataloader = DataLoader(examples, shuffle=True, batch_size=8)
    return dataloader


def get_model(model_name: str, save: bool = False):
    model = SentenceTransformer(model_name)
    if save:
        model.save(model_name)
    return model


def fit_save_model(name: str, model: SentenceTransformer, dataloader, ex):
    train_loss = losses.CosineSimilarityLoss(model=model)
    model.fit(train_objectives=[(dataloader, train_loss)],
              epochs=EPOCHS,
              warmup_steps=int(0.1 * (ex / 8)))
    model.save(name)


def evaluate_model(model, path: str):
    test_label_1, test_label_0 = get_labels(path, TEST_EX)
    embeddings1 = model.encode([x[0] for x in test_label_1] + [x[0] for x in test_label_0], batch_size=8,
                               show_progress_bar=True, convert_to_numpy=True)
    embeddings2 = model.encode([x[1] for x in test_label_1] + [x[1] for x in test_label_0], batch_size=8,
                               show_progress_bar=True, convert_to_numpy=True)
    labels = [1] * len(test_label_1) + [0] * len(test_label_0)

    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

    print(cosine_scores[:int(TEST_EX / 2)].mean())
    print(cosine_scores[int(TEST_EX / 2):].mean())

    sns.distplot(cosine_scores[:int(TEST_EX / 2)], label="1")
    sns.distplot(cosine_scores[int(TEST_EX / 2):], label="0")


async def get_addresses(query: list[str], model: SentenceTransformer, corpus_embeddings: torch.Tensor, ids, corpus: list[str]) -> List[Address]:
    query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=True, device="cuda")

    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=5)
    res = []
    for score, idx in zip(top_results[0], top_results[1]):
        if score < 0.40:
            raise NotFoundException
        print(corpus[idx], "(Score: {:.4f})".format(score))
        res.append(Address(target_building_id=ids[idx], target_address=corpus[idx]))
    return res


# def test_evaluate(model: SentenceTransformer, corpus_embeddings: torch.Tensor, ids, corpus: list[str]):
    


def make_save_corpus(name: str, path: str, model: SentenceTransformer):
    ids, corpus = get_corpus(path)
    embeddings = model.encode(corpus, batch_size=128, show_progress_bar=True, convert_to_tensor=True, device="cuda")
    torch.save(embeddings, name + ".pt")


def load_corpus_embeddings(path: str):
    return torch.load(path)


if __name__ == '__main__':
    path = "not_simple_2000.csv"
    dataloader = get_data_loader(path, TRAIN_EX)
    model = get_model("ai-forever/sbert_large_nlu_ru")
    fit_save_model("dev_model", model, dataloader, TRAIN_EX)
    # evaluate_model(model, path)
    # model = get_model("model_dataset")

    # make_save_corpus("dev_corpus", TEST_PATH, model)
    # corpus_embeddings = load_corpus_embeddings("corpus.pt")

    # model = get_model("sentence-transformers/multi-qa-mpnet-base-dot-v1")

    # print(get_addresses(['Пушкин, Кедринская 12'], model, corpus_embeddings, get_corpus(TEST_PATH)))
