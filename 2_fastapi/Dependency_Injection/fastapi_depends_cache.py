"""示例 4：观察同一次请求中的依赖缓存。

在当前目录启动（每次只启动一个示例）：
    source ../.venv_fastapi/bin/activate
    python -m uvicorn fastapi_depends_cache:app --reload
打开 http://127.0.0.1:8000/docs，可直接发送请求。

请求与预期结果：
    GET /cached/
        200，返回 {"first": 1, "second": 1, "dependency_calls": 1}。
        两个参数复用一次执行的结果，终端只打印一次调用日志。
    GET /uncached/
        200，返回 {"first": 1, "second": 2, "dependency_calls": 2}。
        两处依赖都禁用缓存读取，终端打印两次调用日志。
    再次请求上述任意接口，结果不变，每个新请求都从 1 开始计数。

观察重点：这里的缓存只在同一次请求中复用，不会跨请求保存结果。
参考：https://fastapi.tiangolo.com/tutorial/dependencies/sub-dependencies/
"""

from typing import Annotated

from fastapi import Depends, FastAPI, Request

app = FastAPI(title="示例 4：依赖缓存")


async def counted_dependency(request: Request) -> int:
    # Request 由 FastAPI 提供；state 用于保存当前请求的数据。
    # 计数器属于本次请求，多个请求之间不会共享。
    request.state.dependency_calls = getattr(request.state, "dependency_calls", 0) + 1
    count = request.state.dependency_calls
    print(f"counted_dependency：本次请求第 {count} 次执行")
    return count


@app.get("/cached/")
async def read_cached(
    request: Request,
    first: Annotated[int, Depends(counted_dependency)],
    second: Annotated[int, Depends(counted_dependency)],
):
    # 同一依赖在本次请求中再次被需要时，默认复用缓存值。
    return {
        "first": first,
        "second": second,
        "dependency_calls": request.state.dependency_calls,
    }


@app.get("/uncached/")
async def read_uncached(
    request: Request,
    first: Annotated[int, Depends(counted_dependency, use_cache=False)],
    second: Annotated[int, Depends(counted_dependency, use_cache=False)],
):
    # use_cache=False 让这一处依赖重新执行，而不读取之前缓存的值。
    return {
        "first": first,
        "second": second,
        "dependency_calls": request.state.dependency_calls,
    }
