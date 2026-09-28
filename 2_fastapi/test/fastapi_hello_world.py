from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message":"Hello World,你好","text":"i love you"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_hello_world:app",host="localhost",port=8000,reload=True)
