from fastapi import FastAPI,Form

app = FastAPI()

@app.post("/login/")
async def login(
        username:str = Form(),
        passwd:str = Form()
        ):
    return {"msg":f"your username is {username}, and your passwd is {passwd}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_form_login:app",host="localhost",port=8000,reload=True)
