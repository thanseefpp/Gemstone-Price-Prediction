from typing import List, Optional

from sqlalchemy import update,delete
from sqlalchemy.future import select
from sqlalchemy.orm import Session

from db.gemstone_model import GemstoneModel


class GemstoneDAL():
    """
        DAL means -> Data Access Layer
    """

    def __init__(self, db_session: Session):
        self.db_session = db_session

    async def create_gemstone_data(
        self,
        carat: float,
        depth: float,
        table: float,
        cut: str,
        color: str,
        clarity: str,
        x: float,
        y: float,
        z: float,
        price: float
    ):
        new_entry = GemstoneModel(carat=carat, depth=depth, table=table,
                                  cut=cut, color=color, clarity=clarity,
                                  x=x, y=y, z=z, price=price)
        self.db_session.add(new_entry)
        await self.db_session.flush()

    async def get_all_gemstone_data(self) -> List[GemstoneModel]:
        q = await self.db_session.execute(select(GemstoneModel).order_by(GemstoneModel.id))
        return q.scalars().all()

    async def update_gemstone_db(self,
                                 gemstone_id: int,
                                 carat: Optional[float], depth: Optional[float], table: Optional[float],
                                 cut: Optional[str], color: Optional[str], clarity: Optional[str],
                                 x: Optional[float], y: Optional[float], z: Optional[float], price: Optional[float]):
        q = update(GemstoneModel).where(GemstoneModel.id == gemstone_id)
        if carat:
            q = q.values(carat=carat)
        if depth:
            q = q.values(depth=depth)
        if table:
            q = q.values(table=table)
        if cut:
            q = q.values(cut=cut)
        if color:
            q = q.values(color=color)
        if clarity:
            q = q.values(clarity=clarity)
        if x:
            q = q.values(x=x)
        if y:
            q = q.values(y=y)
        if z:
            q = q.values(z=z)
        if price:
            q = q.values(price=price)
        q.execution_options(synchronize_session="fetch")
        await self.db_session.execute(q)
        
    async def drop_gemstone_table_item(self,
                                 gemstone_id: int):
        q = delete(GemstoneModel).where(GemstoneModel.id == gemstone_id)
        q.execution_options(synchronize_session="fetch")
        await self.db_session.execute(q)
