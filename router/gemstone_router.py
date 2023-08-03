import sys
from typing import Optional

from fastapi import APIRouter, Request

from db.config import async_session
from db.queries import GemstoneDAL
from Gemstone.config.exception import CustomException
from Gemstone.config.logger import logging
from Gemstone.pipeline.predict_pipeline import CustomData, PredictPipeline

router = APIRouter()


async def gemstone_price_prediction(carat, depth, table, x, y, z, cut, color, clarity):
    try:
        data = CustomData(
            carat=float(carat),
            depth=float(depth),
            table=float(table),
            x=float(x),
            y=float(y),
            z=float(z),
            cut=cut,
            color=color,
            clarity=clarity
        )
        df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(df)
        return round(results[0], 2)
    except Exception as e:
        logging.info(
            "Exited the gemstone_price_prediction method prediction goes wrong!")
        raise CustomException(e, sys) from e


@router.post("/gemstones_data")
async def predict_gemstone(gemstone_data_collection: Request):
    collected_data = await gemstone_data_collection.json()
    carat = collected_data['carat']
    depth = collected_data['depth']
    table = collected_data['table']
    x = collected_data['x']
    y = collected_data['y']
    z = collected_data['z']
    cut = collected_data['cut']
    color = collected_data['color']
    clarity = collected_data['clarity']
    async with async_session() as session:
        async with session.begin():
            gemstone_dal = GemstoneDAL(session)
            price = await gemstone_price_prediction(carat, depth, table, x, y, z, cut, color, clarity)
            await gemstone_dal.create_gemstone_data(carat, depth, table, cut, color, clarity, x, y, z, price)
            return {
                "predicted_result": price
            }


@router.get("/gemstones_data")
async def get_all_gemstones():
    async with async_session() as session:
        async with session.begin():
            gemstone_dal = GemstoneDAL(session)
            return await gemstone_dal.get_all_gemstone_data()


@router.put("/gemstones/{gemstones_id}")
async def update_gemstone(
    gemstone_id: int, carat: Optional[float] = None, depth: Optional[float] = None, table: Optional[float] = None,
    cut: Optional[str] = None, color: Optional[str] = None, clarity: Optional[str] = None,
    x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None, price: Optional[float] = None
):
    async with async_session() as session:
        async with session.begin():
            gemstone_dal = GemstoneDAL(session)
            return await gemstone_dal.update_gemstone_db(
                gemstone_id, carat, depth, table, cut, color, clarity,
                x, y, z, price)
