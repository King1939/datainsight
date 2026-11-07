"""生成一份假的电商订单数据 data/sales.csv。

固定随机种子，所以每次运行结果一致——评测要用。
用标准库就够了，不需要 pandas。
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sales.csv"

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安"]
CATEGORIES = ["数码", "家居", "服饰", "食品", "美妆"]
CHANNELS = ["App", "小程序", "网页"]

# 不同类目的单价区间，让数据有区分度
PRICE_RANGE = {
    "数码": (300, 6000),
    "家居": (60, 900),
    "服饰": (80, 700),
    "食品": (15, 200),
    "美妆": (50, 800),
}


def main(rows: int = 320, seed: int = 42) -> None:
    random.seed(seed)
    start = date(2025, 1, 1)
    records = []

    for i in range(1, rows + 1):
        category = random.choices(CATEGORIES, weights=[2, 2, 3, 3, 2])[0]
        low, high = PRICE_RANGE[category]
        price = round(random.uniform(low, high), 2)
        qty = random.choices([1, 1, 2, 2, 3, 4, 6, 10], k=1)[0]
        day = start + timedelta(days=random.randint(0, 300))
        records.append(
            {
                "order_id": f"O{i:05d}",
                "order_date": day.isoformat(),
                "city": random.choices(CITIES, weights=[18, 20, 14, 16, 12, 10, 6, 4])[0],
                "category": category,
                "channel": random.choices(CHANNELS, weights=[5, 3, 2])[0],
                "quantity": qty,
                "unit_price": price,
                "amount": round(price * qty, 2),
            }
        )

    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    total = sum(r["amount"] for r in records)
    print(f"已生成 {OUT}")
    print(f"行数 {len(records)}，总金额 {total:,.2f}")


if __name__ == "__main__":
    main()
