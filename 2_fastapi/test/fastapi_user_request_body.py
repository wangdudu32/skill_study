from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    username:str
    email:str

@app.post("/users/")
async def post_user(user:User):
    return {
            "messages":"this is port_user function calling",
            "user_name":user.username,
            "user_email":user.email
            }
