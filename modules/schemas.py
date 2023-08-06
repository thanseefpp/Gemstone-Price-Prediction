from datetime import datetime
from typing import List
from pydantic import BaseModel
from typing import Union


class GemstoneBaseSchema(BaseModel):
    # id: Union[str, None]
    carat : float
    cut : str 
    color : str 
    clarity : str 
    depth : float
    table : float
    x : float
    y : float
    z : float
    # price : Union[float, None]
    # createdAt: Union[datetime, None]
    # updatedAt: Union[datetime, None]

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        arbitrary_types_allowed = True


class ListGemstoneResponse(BaseModel):
    status: str
    results: int
    gemstone: List[GemstoneBaseSchema]

