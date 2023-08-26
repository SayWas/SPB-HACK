from typing import List

from pydantic import BaseModel


class Address(BaseModel):
    target_building_id: int
    target_address: str


class AddressResponse(BaseModel):
    success: bool
    query: str
    result: List[Address]
