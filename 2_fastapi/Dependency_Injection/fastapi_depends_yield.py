"""示例 5：使用 yield 管理资源的创建和清理。

在当前目录启动（每次只启动一个示例）：
    source ../.venv_fastapi/bin/activate
    python -m uvicorn fastapi_depends_yield:app --reload
打开 http://127.0.0.1:8000/docs，可直接发送请求。

请求与预期结果：
    GET /items/
        200，items 为 ["apple", "banana"]，closed_during_handler 为 false。
        终端依次打印：1. 创建模拟会话 → 2. 读取数据 → 3. 关闭模拟会话。
    GET /error/
        400，返回 {"detail": "故意触发异常，观察会话是否关闭"}。
        终端依次打印：1. 创建模拟会话 → 2. 模拟业务异常 → 3. 关闭模拟会话。

观察重点：yield 提供资源，finally 负责清理，发生异常时也会执行。
响应中的 closed_during_handler 记录接口执行时的状态；清理结果看终端日志。
参考：https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/
"""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

app = FastAPI(title="示例 5：yield 与资源清理")


class MemorySession:
    """用一个普通对象模拟需要关闭的会话，无需连接数据库。"""

    def __init__(self):
        self.closed = False
        print("1. 创建模拟会话")

    def read_items(self) -> list[str]:
        if self.closed:
            raise RuntimeError("会话已经关闭")
        return ["apple", "banana"]

    def close(self) -> None:
        self.closed = True
        print("3. 关闭模拟会话")


def get_session() -> Iterator[MemorySession]:
    session = MemorySession()
    try:
        # yield 只执行一次：先把对象交给接口，随后再继续清理。
        yield session
    finally:
        # 将清理放进 finally，保证接口抛出异常时也会关闭会话。
        session.close()


@app.get("/items/")
async def read_items(session: Annotated[MemorySession, Depends(get_session)]):
    print("2. 读取数据")
    return {
        "items": session.read_items(),
        "closed_during_handler": session.closed,
    }


@app.get("/error/")
async def trigger_error(session: Annotated[MemorySession, Depends(get_session)]):
    print("2. 模拟业务异常")
    raise HTTPException(status_code=400, detail="故意触发异常，观察会话是否关闭")
