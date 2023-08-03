from sqlalchemy import Column, Float, Integer, String

from db.config import Base


class GemstoneModel(Base):
    __tablename__ = 'gemstone'

    id = Column(Integer, primary_key=True)
    carat = Column(Float, nullable=False)
    cut = Column(String, nullable=False)
    color = Column(String, nullable=False)
    clarity = Column(String, nullable=False)
    depth = Column(Float, nullable=False)
    table = Column(Float, nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    z = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
