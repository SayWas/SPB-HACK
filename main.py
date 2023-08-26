from fastapi import FastAPI, HTTPException
from starlette import status
from transliterate import translit

from dataset import get_corpus
from exceptions import NotFoundException, NOT_FOUND_EXCEPTION_TEXT
from schemas import AddressResponse

from sentence_transformers import SentenceTransformer

from search import get_addresses, load_corpus_embeddings

app = FastAPI()


class MlMeta:
    def __init__(self):
        self.model = None
        self.corpus_embeddings = None
        self.ids = None
        self.corpus = None


MlMeta = MlMeta()
translation = str.maketrans(dict(zip('QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm',
                                     'ЙЦУКЕНГШЩЗФЫВАПРОЛДЯЧСМИТЬйцукенгшщзфывапролдячсмить')))


@app.on_event("startup")
async def startup_event():
    # MlMeta.model = SentenceTransformer("ai-forever/sbert_large_nlu_ru")
    # MlMeta.model = SentenceTransformer("_dev_model")
    MlMeta.model = SentenceTransformer("model_dataset_v2")
    MlMeta.corpus_embeddings = load_corpus_embeddings("corpus_v2.pt")
    MlMeta.ids, MlMeta.corpus = get_corpus("additional_data/building_20230808.csv")
    pass


@app.get(
    "/",
    tags=["Search"],
    status_code=status.HTTP_200_OK,
    response_model=AddressResponse,
    name="search:get_address",
    responses={
        status.HTTP_200_OK: {
            "description": "OK",
            "model": AddressResponse
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Not Found",
            "content": {
                "application/json": {
                    "examples": {
                        "not_found": {
                            "summary": "Not Found",
                            "value": {
                                "success": False,
                                "details": NOT_FOUND_EXCEPTION_TEXT
                            }
                        }
                    }
                }
            }
        }
    }
)
async def get_address(
        query: str
):
    try:
        query1 = translit(query, 'ru')
        query2 = query.translate(translation)
        result = await get_addresses([query1, query2], MlMeta.model, MlMeta.corpus_embeddings, MlMeta.ids, MlMeta.corpus)
        print(result)
        return {
            "success": True,
            "query": query,
            "result": result
        }
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=NOT_FOUND_EXCEPTION_TEXT
        )
    except Exception as e:
        print(e)
