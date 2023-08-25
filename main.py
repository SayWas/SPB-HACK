from fastapi import FastAPI
from starlette import status

from dataset import get_corpus
from exceptions import NotFoundException, NOT_FOUND_EXCEPTION_TEXT
from schemas import AddressResponse

from sentence_transformers import SentenceTransformer

from search import get_addresses

app = FastAPI()


class MlMeta:
    def __init__(self):
        self.model = None
        self.corpus = None


MlMeta = MlMeta()


@app.on_event("startup")
async def startup_event():
    MlMeta.model = SentenceTransformer("model_dataset")
    MlMeta.corpus = get_corpus("additional_data/building_20230808.csv")
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
        result = await get_addresses(query, MlMeta.model, MlMeta.corpus)
        return {
            "success": True,
            "query": query,
            "result": result
        }
    except NotFoundException:
        return {
            "success": False,
            "details": NOT_FOUND_EXCEPTION_TEXT,
        }
    except Exception as e:
        print(e)
