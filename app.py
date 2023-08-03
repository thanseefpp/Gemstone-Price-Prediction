import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.config import Base, engine
from router.gemstone_router import router

app = FastAPI()
app.include_router(router)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# You can comment this function if don't want to create the db when even it start
# only comment if you have have the db mentioned in config section with same location


@app.on_event("startup")
async def startup():
    # create db tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

if __name__ == '__main__':
    uvicorn.run("app:app", port=5100, host='127.0.0.1')
