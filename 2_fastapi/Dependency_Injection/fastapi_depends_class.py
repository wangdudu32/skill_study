"""示例 2：用类封装依赖参数。

在当前目录启动（每次只启动一个示例）：
    source ../.venv_fastapi/bin/activate
    python -m uvicorn fastapi_depends_class:app --reload
打开 http://127.0.0.1:8000/docs，可直接发送请求。

请求与预期结果：
    GET /items/?skip=1&limit=2
        200，items 为 ["banana", "apricot"]，skip 为 1，limit 为 2。
    GET /items/shortcut/?skip=1&limit=2
        与上一个接口返回相同的内容，比较两种 Depends 写法。
    GET /items/?limit=101 或 /items/?skip=abc
        422，构造函数中的查询参数也有范围和类型校验。

观察重点：注入的是 Pagination 实例，可以通过属性读取参数。
参考：https://fastapi.tiangolo.com/tutorial/dependencies/classes-as-dependencies/
"""

from typing import Annotated

from fastapi import Depends, FastAPI, Query

app = FastAPI(title="示例 2：类作为依赖")

ITEMS = ["apple", "banana", "apricot", "blueberry", "avocado"]


class Pagination:
    def __init__(
        self,
        skip: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 2,
    ):
        # FastAPI 解析请求参数后调用构造函数，将生成的实例传给接口。
        self.skip = skip
        self.limit = limit


@app.get("/items/")
async def read_items(page: Annotated[Pagination, Depends(Pagination)]):
    return {
        "skip": page.skip,
        "limit": page.limit,
        "items": ITEMS[page.skip : page.skip + page.limit],
    }


@app.get("/items/shortcut/")
async def read_items_shortcut(page: Annotated[Pagination, Depends()]):
    # 省略 Depends 的参数时，FastAPI 使用类型注解中的 Pagination 类。
    return {
        "skip": page.skip,
        "limit": page.limit,
        "items": ITEMS[page.skip : page.skip + page.limit],
    }
