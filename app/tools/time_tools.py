"""本地时钟工具：为 Agent 提供准确的当前时间与相对时间区间参考。"""

from __future__ import annotations

from datetime import datetime, timedelta

from . import registry

_WEEKDAYS_ZH = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")


def _month_bounds(anchor: datetime) -> tuple[str, str]:
    """返回 anchor 所在月份的上一个月起止日期（YYYY-MM-DD）。"""
    first_this_month = anchor.date().replace(day=1)
    last_prev_month = first_this_month - timedelta(days=1)
    first_prev_month = last_prev_month.replace(day=1)
    return first_prev_month.isoformat(), last_prev_month.isoformat()


@registry.tool
def get_current_time(offset_days: int = 0) -> str:
    """获取本机当前时间、星期、时区及常用相对时间区间（今天/近7天/近30天/本月/上月/今年）。

    当用户提问包含"今天、昨天、最近、近一周、本月、上个月、今年、最新"等相对时间词，
    需要确定具体日期或时间范围（用于 SQL 筛选、Python 计算或联网检索）时，先调用本工具。

    Args:
        offset_days: 相对今天的偏移天数，默认 0 表示今天；负数表示几天前（如 -7），正数表示几天后。
    """
    now = datetime.now().astimezone()
    today = now.date()
    weekday = _WEEKDAYS_ZH[now.weekday()]
    tz_name = now.tzname() or "本地时区"
    utc_offset = now.strftime("%z")
    tz_label = f"{tz_name} (UTC{utc_offset[:3]}:{utc_offset[3:]})" if len(utc_offset) == 5 else tz_name

    lines = [
        f"当前本地时间：{now.strftime('%Y-%m-%d %H:%M:%S')}（{weekday}，{tz_label}）",
        f"Unix 时间戳：{int(now.timestamp())}",
    ]

    if offset_days:
        target = today + timedelta(days=int(offset_days))
        target_wd = _WEEKDAYS_ZH[target.weekday()]
        lines.append(f"偏移 {int(offset_days):+d} 天日期：{target.isoformat()}（{target_wd}）")

    yesterday = today - timedelta(days=1)
    last_7_start = today - timedelta(days=6)
    last_30_start = today - timedelta(days=29)
    month_start = today.replace(day=1)
    prev_m_start, prev_m_end = _month_bounds(now)
    year_start = today.replace(month=1, day=1)

    lines.extend(
        [
            "常用时间参考（含今天）：",
            f"- 今天：{today.isoformat()}",
            f"- 昨天：{yesterday.isoformat()}",
            f"- 近 7 天：{last_7_start.isoformat()} ~ {today.isoformat()}",
            f"- 近 30 天：{last_30_start.isoformat()} ~ {today.isoformat()}",
            f"- 本月至今：{month_start.isoformat()} ~ {today.isoformat()}",
            f"- 上个月：{prev_m_start} ~ {prev_m_end}",
            f"- 今年至今：{year_start.isoformat()} ~ {today.isoformat()}",
        ]
    )
    return "\n".join(lines)
