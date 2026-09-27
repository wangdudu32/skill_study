from fastapi import FastAPI
from pydantic import BaseModel,Field,validator

app = FastAPI()

class Book(BaseModel):
    title:str = Field(max_length = 50)
    author:str
    pages:int = Field(gt=0)

    @validator("author")
    def author_no_digits(cls,v):
        # 判断是否包含数字
        if any(char.isdigit() for char in v):
            raise ValueError("作者名不能包含数字")
        return v

class BookOut(BaseModel):
    title:str
    author:str

@app.post("/books/",response_model=BookOut)
async def get_book(book:Book):
    return book
