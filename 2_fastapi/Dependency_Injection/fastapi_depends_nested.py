"""示例 3：依赖函数也可以依赖其他函数。

在当前目录启动（每次只启动一个示例）：
    source ../.venv_fastapi/bin/activate
    python -m uvicorn fastapi_depends_nested:app --reload
打开 http://127.0.0.1:8000/docs，在 x-token 输入框填入 demo-token。

请求与预期结果：
    curl -H 'X-Token: demo-token' http://127.0.0.1:8000/users/me
        200，返回 {"username": "alice", "role": "student"}。
    curl -H 'X-Token: wrong-token' http://127.0.0.1:8000/users/me
        401，返回 {"detail": "无效的演示令牌"}，接口函数不会执行。
    curl http://127.0.0.1:8000/users/me
        422，缺少必填请求头 X-Token。

观察终端日志：read_token → get_current_user → read_current_user。
此处用固定令牌和内存字典模拟用户查询。
参考：https://fastapi.tiangolo.com/tutorial/dependencies/sub-dependencies/
"""

from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException

app = FastAPI(title="示例 3：嵌套依赖")

FAKE_USERS = {"demo-token": {"username": "alice", "role": "student"}}


def read_token(
    x_token: Annotated[str, Header(description="演示值：demo-token")],
) -> str:
    # Header 默认将参数名中的下划线转成连字符：x_token 对应 X-Token。
    print("1. read_token：读取请求头 X-Token")
    return x_token


def get_current_user(token: Annotated[str, Depends(read_token)]) -> dict:
    # FastAPI 先执行 read_token，再把返回的令牌传到这里。
    print("2. get_current_user：根据令牌查询模拟用户")
    user = FAKE_USERS.get(token)
    if user is None:
        raise HTTPException(status_code=401, detail="无效的演示令牌")
    return user


@app.get("/users/me")
async def read_current_user(user: Annotated[dict, Depends(get_current_user)]):
    # 接口只声明直接依赖，FastAPI 会自动处理整条依赖链。
    print("3. read_current_user：返回用户信息")
    return user
