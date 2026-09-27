from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/items/")
async def get_items_by_type(type:str|None = None):
    return {"ret":f"this is /items/, and the type you choose is {type}"}
