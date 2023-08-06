from typing import List

from pydantic import BaseModel


class GemstoneBaseSchema(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


class ListGemstoneResponse(BaseModel):
    status: str
    results: int
    gemstone: List[GemstoneBaseSchema]
