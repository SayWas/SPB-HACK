from pydantic import BaseModel


class Address(BaseModel):
    target_building_id: str
    target_address: str


class AddressResponse(BaseModel):
    success: bool
    query: list[str]
    result: list[str]
