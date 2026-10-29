"""示例 1：用依赖函数复用查询参数。

在当前目录启动（每次只启动一个示例）：
    source ../.venv_fastapi/bin/activate
    python -m uvicorn fastapi_depends_basic:app --reload
打开 http://127.0.0.1:8000/docs，可直接发送请求。

请求与预期结果：
    GET /items/?q=ap&skip=0&limit=2
        200，items 为 ["apple", "apricot"]。
    GET /users/?q=a&skip=1&limit=2
        200，users 为 ["anna", "charlie"]。
    GET /items/?skip=-1 或 /users/?limit=0
        422，依赖函数中的参数约束也会自动校验。

观察重点：两个接口都使用 common_parameters，参数只需定义一次。
参考：https://fastapi.tiangolo.com/tutorial/dependencies/
"""

from typing import Annotated

from fastapi import Depends, FastAPI, Query

app = FastAPI(title="示例 1：基础依赖函数")

ITEMS = ["apple", "banana", "apricot", "blueberry", "avocado"]
USERS = ["alice", "bob", "anna", "charlie", "amy"]


def common_parameters(
    q: Annotated[str | None, Query(description="名称包含的关键字")] = None,
    skip: Annotated[int, Query(ge=0, description="跳过多少条记录")] = 0,
    limit: Annotated[int, Query(ge=1, le=100, description="最多返回多少条")] = 2,
) -> dict:
    return {"q": q, "skip": skip, "limit": limit}


@app.get("/items/")
async def read_items(
    common: Annotated[dict, Depends(common_parameters)],
):
    # dict 是收到的值的类型；Depends 告诉 FastAPI 这个值由哪个函数提供。
    # 传入函数本身 common_parameters，不要写成 common_parameters()。
    keyword = (common["q"] or "").lower()
    filtered = [item for item in ITEMS if keyword in item.lower()]
    start = common["skip"]
    return {"params": common, "items": filtered[start : start + common["limit"]]}


@app.get("/users/")
async def read_users(
    common: Annotated[dict, Depends(common_parameters)],
):
    # 换一个接口，仍然可以复用同一组查询参数及其校验规则。
    keyword = (common["q"] or "").lower()
    filtered = [user for user in USERS if keyword in user.lower()]
    start = common["skip"]
    return {"params": common, "users": filtered[start : start + common["limit"]]}
