from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from modules.database import Base, engine
from modules.gemstone_router import router

Base.metadata.create_all(bind=engine)

app = FastAPI()
# "http://localhost:8000",
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router, tags=['Gemstone'], prefix='/api/gemstone')


@app.get("/")
def root():
    return {
        "message": "Welcome to Gemstone Prediction API's", 'info': "for more information visit /docs"}


if __name__ == '__main__':
    uvicorn.run(f"{Path(__file__).stem}:app",
                host="127.0.0.1", port=8888, reload=True)
