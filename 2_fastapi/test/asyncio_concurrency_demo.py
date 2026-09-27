import asyncio
import time

async def order_food(food:str):
    print(f"开始制餐: {food}")
    await asyncio.sleep(3)
    print(f"{food} 制作完毕")
    return food

# 异步函数
async def main():
    start = time.time()

    # 并发执行，两个外卖同时点
    task1 = asyncio.create_task(order_food("披萨"))
    task2 = asyncio.create_task(order_food("汉堡"))

    print("下单完成，开始玩会手机...")
    time.sleep(2)
    food1 = await task1
    food2 = await task2

    print(f"吃到 {food1} 和 {food2} 了")
    end = time.time()
    print(f"总共花费 {end - start} s")

if __name__ == "__main__":
    asyncio.run(main())
