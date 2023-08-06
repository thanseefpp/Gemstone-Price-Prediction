from fastapi_utils.guid_type import GUID, GUID_SERVER_DEFAULT_POSTGRESQL
from sqlalchemy import TIMESTAMP, Column, Float, String
from sqlalchemy.sql import func

from .database import Base


class GemstoneModel(Base):
    __tablename__ = 'gemstone'
    id = Column(GUID, primary_key=True,
                server_default=GUID_SERVER_DEFAULT_POSTGRESQL)
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
    createdAt = Column(TIMESTAMP(timezone=True),
                       nullable=False, server_default=func.now())
    updatedAt = Column(TIMESTAMP(timezone=True),
                       default=None, onupdate=func.now())
