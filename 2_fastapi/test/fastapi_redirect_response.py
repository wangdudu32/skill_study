from fastapi import FastAPI
from fastapi.responses import RedirectResponse

app = FastAPI()

@app.get("/")
async def get_root():
    return {"msg":"hello,this is root dir"}

@app.get("/direct/")
async def get_direct():
    return {"msg":"and this is /direct/ dir"}

@app.get("/redirect")
async def get_redirect():
    return RedirectResponse("/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_redirect_response:app",host="localhost",port=8000,reload=True)
