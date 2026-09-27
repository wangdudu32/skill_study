from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name:str
    description:str | None = None
    price:float

@app.post("/Item/")
async def get_item(item:Item):
    return {"name":item.name,"description":item.description,"price":item.price}
