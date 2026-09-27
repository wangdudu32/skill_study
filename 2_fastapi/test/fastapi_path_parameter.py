from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/square/{num}")
async def get_square(num:int):
    return {"result":f"{num}'s square is {num*num}"}
