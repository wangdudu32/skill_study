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

    print("下单完成，开始玩会手机...")
    food1 = await order_food("披萨") 
    food2 = await order_food("汉堡") 

    print(f"吃到 {food1} 和 {food2} 了")
    end = time.time()
    print(f"总共花费 {end - start} s")

if __name__ == "__main__":
    asyncio.run(main())
