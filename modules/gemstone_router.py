import sys
from .schemas import GemstoneBaseSchema
from .models import GemstoneModel
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status, APIRouter, Response
from Gemstone.config.exception import CustomException
from Gemstone.config.logger import logging
from Gemstone.pipeline.predict_pipeline import CustomData, PredictPipeline
from .database import get_db
from sqlalchemy import select


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


@router.get('/')
def get_all_gemstones(db: Session = Depends(get_db)):
    gemstones = db.query(GemstoneModel).all()
    return {'status': 'success', 'results': len(gemstones), 'gemstones': gemstones}

@router.post('/', status_code=status.HTTP_201_CREATED)
async def create_gemstone(payload: GemstoneBaseSchema, db: Session = Depends(get_db)):
    carat = payload.dict()['carat']
    cut = payload.dict()['cut']
    color = payload.dict()['color']
    clarity = payload.dict()['clarity']
    depth = payload.dict()['depth']
    table = payload.dict()['table']
    x = payload.dict()['x']
    y = payload.dict()['y']
    z = payload.dict()['z']
    price = await gemstone_price_prediction(carat, depth, table, x, y, z, cut, color, clarity)
    new_dict = {
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z,
        "price": price
    }
    new_gemstone = GemstoneModel(**new_dict)
    db.add(new_gemstone)
    db.commit()
    db.refresh(new_gemstone)
    return {"status": "success", "note": new_gemstone}

@router.patch('/{gemstoneId}')
def update_gemstone(gemstoneId: str, payload: GemstoneBaseSchema, db: Session = Depends(get_db)):
    gem_query = db.query(GemstoneModel).filter(GemstoneModel.id == gemstoneId)
    db_gem = gem_query.first()

    if not db_gem:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'No gemstone with this id: {gemstoneId} not found')
    update_data = payload.dict(exclude_unset=True)
    gem_query.filter(GemstoneModel.id == gemstoneId).update(update_data,
                                                       synchronize_session=False)
    db.commit()
    db.refresh(db_gem)
    return {"status": "success", "gemstone": db_gem}

@router.get('/{gemstoneId}')
def get_item(gemstoneId: str, db: Session = Depends(get_db)):
    if gemstone := db.query(GemstoneModel).filter(GemstoneModel.id == gemstoneId).first():
        return {"status": "success", "gemstone": gemstone}
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"No gemstone with this id: {id} not found")
        
@router.delete('/{gemstoneId}')
def delete_item(gemstoneId: str, db: Session = Depends(get_db)):
    gem_query = db.query(GemstoneModel).filter(GemstoneModel.id == gemstoneId)
    gemstone = gem_query.first()
    if not gemstone:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'No gemstone with this id: {gemstoneId} not found')
    gem_query.delete(synchronize_session=False)
    db.commit()
    return Response(content=f"Item Deleted Successful {gemstoneId}",status_code=status.HTTP_204_NO_CONTENT)