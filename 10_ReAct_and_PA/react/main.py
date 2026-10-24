# 示例数据：先用字典模拟数据库和物流接口
ORDERS = {
    "A123": {"product": "机械键盘", "tracking_number": "SF001"}
}

SHIPMENTS = {
    "SF001": {"status": "运输中", "location": "上海转运中心"}
}


def get_order(order_id: str) -> dict:
    """根据订单号查询订单信息，包括运单号。"""
    return ORDERS.get(order_id, {"error": "订单不存在"})


def get_shipment(tracking_number: str) -> dict:
    """根据运单号查询物流状态。"""
    return SHIPMENTS.get(tracking_number, {"error": "运单不存在"})


if __name__ == "__main__":
    # 先手动调用，验证两个工具能正常工作
    order = get_order("A123")
    print("订单信息：", order)

    if "tracking_number" in order:
        shipment = get_shipment(order["tracking_number"])
        print("物流信息：", shipment)
