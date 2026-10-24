from mcp.server import MCPServer

mcp = MCPServer("inventory")

STOCK = {
    "鼠标": 35,
    "键盘": 18,
    "显示器": 0,
}

@mcp.tool()
def get_stock(product: str) -> str:
    """查询商品库存。product 为商品中文名称，例如：鼠标、键盘、显示器。"""
    if product not in STOCK:
        return f"未找到商品：{product}"
    return f"{product} 的库存为 {STOCK[product]} 件。"

if __name__ == "__main__":
    mcp.run(transport="stdio")
