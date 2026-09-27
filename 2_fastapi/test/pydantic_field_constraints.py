from fastapi import FastAPI
from pydantic import BaseModel,Field

app = FastAPI()

class Book(BaseModel):
    title:str = Field(max_length = 50)
    author:str
    pages:int = Field(gt=0)

@app.post("/books/")
async def get_book(book:Book):
    return {"messages":"成功添加一本书","content":book}
