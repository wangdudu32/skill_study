from datetime import datetime
from pydantic import BaseModel

class User(BaseModel):
    id:int
    name:str = "Join"
    signup_ts : datetime | None = None
    friends : list[int] = []

external_data = {
    "id":"123",
    "signup_ts":"2026-07-23 12:22",
    "friends":[1,"2",b"3"]
}

user = User(**external_data)
print(user)

print(user.id)
