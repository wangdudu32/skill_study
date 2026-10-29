"""示例 6：在路由装饰器中声明只需执行的依赖。

在当前目录启动（每次只启动一个示例）：
    source ../.venv_fastapi/bin/activate
    python -m uvicorn fastapi_depends_decorator:app --reload
打开 http://127.0.0.1:8000/docs，在受保护接口的 x-token 中填入 demo-token。

请求与预期结果：
    curl http://127.0.0.1:8000/public/
        200，返回 {"message": "这个接口不需要令牌"}。
    curl -H 'X-Token: demo-token' http://127.0.0.1:8000/items/
        200，返回 {"items": ["apple", "banana"]}。
    curl -H 'X-Token: wrong-token' http://127.0.0.1:8000/items/
        401，返回 {"detail": "无效的演示令牌"}。
    curl http://127.0.0.1:8000/items/
        422，缺少必填请求头 X-Token。

观察重点：校验通过后才执行接口，依赖的返回值不会传入接口参数。
参考：https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/
"""

from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException

app = FastAPI(title="示例 6：装饰器中的依赖")


def verify_token(
    x_token: Annotated[str, Header(description="演示值：demo-token")],
) -> None:
    print("1. verify_token：校验请求头")
    if x_token != "demo-token":
        raise HTTPException(status_code=401, detail="无效的演示令牌")


@app.get("/public/")
async def read_public():
    return {"message": "这个接口不需要令牌"}


@app.get("/items/", dependencies=[Depends(verify_token)])
async def read_items():
    # 只需要依赖完成校验，因此将 Depends 放在装饰器中。
    print("2. read_items：校验通过，返回数据")
    return {"items": ["apple", "banana"]}
