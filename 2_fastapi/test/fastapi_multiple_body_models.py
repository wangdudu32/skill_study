from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    name:str
    email:str

class People(BaseModel):
    name:str
    phone:str
    sex:str


@app.post("/users/")
async def post_user(user:User,people:People):
    return {
            "messages":"this is port_user function calling",
            "user_name":user.name,
            "user_email":user.email,
            "people_name":people.name,
            "people_phone":people.phone,
            "people_sex":people.sex
            }
