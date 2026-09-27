from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()

cnt = 0

@app.get("/")
async def root():
    global cnt
    print("this is root")
    cnt = cnt + 1
    return {"msg":f"hello, this is root dir,count{cnt}"}

@app.get("/file/{file_name}")
async def get_file(file_name:str):
    with open(file_name) as f:
        print(f"return file: {file_name}")
        return FileResponse(file_name)

@app.get("/item/{item_id}")
def get_item(item_id):
    global cnt
    print(item_id)
    cnt = cnt + 1
    return {"message":f"Hello, your request:{item_id},count{cnt}"}

