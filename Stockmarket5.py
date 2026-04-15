import json
import pandas as pd
from pathlib import Path



INPUT_FILES = {
    "nasdaq": r"D:\Work\Nasdak_10year_history_data_1.json",
    "sp500": r"D:\Work\S&P500_10year_history_data_1.json",
    "dow": r"D:\Work\DowJones_10year_history_data_1.json"
}


def clean_price(value):
    """Convert '3,245.67' → 3245.67"""
    if pd.isna(value):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except:
        return None


def load_json_as_df(filepath):
    """Load JSON → DataFrame with cleaned prices & datetime"""
    with open(filepath, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["open"] = df["open"].apply(clean_price)
    df["close"] = df["close"].apply(clean_price)
    df["date"] = pd.to_datetime(df["date"], format="%b %d, %Y", errors="coerce")

    return df.sort_values("date").reset_index(drop=True)


# ============================================================
# 3. CRASH DETECTION
# ============================================================
def detect_single_day_crashes(df, threshold=3):
    df = df.copy()
    df["drop_pct"] = ((df["open"] - df["close"]) / df["open"]) * 100
    return df[df["drop_pct"] >= threshold]


def detect_multi_day_crashes(df, threshold=3):
    multi = []

    for i in range(len(df) - 3):
        d0 = df.iloc[i]
        open0, close0 = d0["open"], d0["close"]

        if pd.isna(open0) or pd.isna(close0):
            continue

        d0_drop = ((open0 - close0) / open0) * 100
        if d0_drop < threshold:
            continue

        closes = df.iloc[i:i+4]["close"].tolist()
        fall_days = 0

        for d in range(1, 4):
            if closes[d] < closes[d - 1]:
                fall_days += 1
            else:
                break

        if fall_days >= 1:
            end = df.iloc[i+fall_days]["date"]
            total_drop_value = open0 - closes[fall_days]
            total_drop_pct = (total_drop_value / open0) * 100

            multi.append({
                "start_date": d0["date"],
                "end_date": end,
                "crash_length": fall_days + 1,
                "day0_drop_pct": d0_drop,
                "total_drop_value": total_drop_value,
                "total_drop_percent": total_drop_pct
            })

    return pd.DataFrame(multi)


# ============================================================
# 4. FIND COMMON SAME-DAY CRASHES
# ============================================================
def find_3index_same_day(a, b, c):
    return sorted(set(a["date"]) & set(b["date"]) & set(c["date"]))


def find_2index_same_day(single):
    return {
        "nasdaq_sp500": sorted(set(single["nasdaq"]["date"]) & set(single["sp500"]["date"])),
        "nasdaq_dow": sorted(set(single["nasdaq"]["date"]) & set(single["dow"]["date"])),
        "sp500_dow": sorted(set(single["sp500"]["date"]) & set(single["dow"]["date"]))
    }


def filter_out_3index_from_2index(two, three):
    """Remove all dates that appear in the 3-index list."""
    filtered = {}

    t3 = set(three)

    for k, v in two.items():
        filtered[k] = sorted(set(v) - t3)

    return filtered


# ============================================================
# 5. MULTI-DAY OVERLAPS
# ============================================================
def find_3index_multi(a, b, c):
    overlaps = []

    for _, A in a.iterrows():
        for _, B in b.iterrows():
            for _, C in c.iterrows():

                start = max(A["start_date"], B["start_date"], C["start_date"])
                end = min(A["end_date"], B["end_date"], C["end_date"])

                if start <= end:
                    overlaps.append({
                        "common_start": start,
                        "common_end": end,
                        "nasdaq": A,
                        "sp500": B,
                        "dow": C
                    })

    return overlaps


def find_overlap_between_two(a, b):
    overlaps = []

    for _, A in a.iterrows():
        for _, B in b.iterrows():

            start = max(A["start_date"], B["start_date"])
            end = min(A["end_date"], B["end_date"])

            if start <= end:
                overlaps.append({
                    "common_start": start,
                    "common_end": end,
                    "a": A,
                    "b": B
                })

    return overlaps


def find_2index_multi(multi):
    return {
        "nasdaq_sp500": find_overlap_between_two(multi["nasdaq"], multi["sp500"]),
        "nasdaq_dow": find_overlap_between_two(multi["nasdaq"], multi["dow"]),
        "sp500_dow": find_overlap_between_two(multi["sp500"], multi["dow"])
    }


def filter_out_3index_multi(two, three):
    """Remove 2-index windows fully covered by any 3-index window."""
    filtered = {}

    for pair, windows in two.items():
        keep = []

        for win in windows:
            two_start = win["common_start"]
            two_end = win["common_end"]

            covered = False

            for t3 in three:
                if two_start >= t3["common_start"] and two_end <= t3["common_end"]:
                    covered = True
                    break

            if not covered:
                keep.append(win)

        filtered[pair] = keep

    return filtered


# ============================================================
# 6. PRINT FUNCTIONS
# ============================================================
def print_3index_same_day_details(single, common):
    print("\n==============================")
    print("📊 SAME-DAY CRASHES (ALL 3 INDEXES)")
    print("==============================")

    for d in common:
        print(f"\n🟥 Date: {d.date()}\n")

        for idx in ["nasdaq", "sp500", "dow"]:
            row = single[idx][single[idx]["date"] == d].iloc[0]
            print(f"{idx.upper()}:")
            print(f"  Open:  {row['open']}")
            print(f"  Close: {row['close']}")
            print(f"  Drop:  {row['drop_pct']:.2f}%\n")


def print_2index_same_day_details(single, two):
    names = {
        "nasdaq_sp500": ("NASDAQ", "S&P500"),
        "nasdaq_dow": ("NASDAQ", "DOW"),
        "sp500_dow": ("S&P500", "DOW")
    }

    print("\n===================================")
    print("📊 SAME-DAY CRASHES (ANY 2 INDEXES ONLY)")
    print("===================================\n")

    for pair, dates in two.items():
        n1, n2 = names[pair]

        print(f"\n🟧 {n1} & {n2}")
        print("------------------------------")

        if not dates:
            print("✔ None")
            continue

        for d in dates:
            print(f"\n📅 {d.date()}\n")

            for idx, name in zip(pair.split("_"), [n1, n2]):
                df = single[idx]
                row = df[df["date"] == d].iloc[0]

                print(f"{name}:")
                print(f"  Open:  {row['open']}")
                print(f"  Close: {row['close']}")
                print(f"  Drop:  {row['drop_pct']:.2f}%")
            print()


def print_3index_multi_details(overlaps):
    print("\n====================================================")
    print("📊 MULTI-DAY CRASH WINDOWS (ALL 3 INDEXES)")
    print("====================================================\n")

    for c in overlaps:
        print(f"🟦 {c['common_start'].date()} → {c['common_end'].date()}\n")

        for name, row in [
            ("NASDAQ", c["nasdaq"]),
            ("S&P500", c["sp500"]),
            ("DOW", c["dow"])
        ]:
            print(f"{name}:")
            print(f"  Start: {row['start_date'].date()}")
            print(f"  End:   {row['end_date'].date()}")
            print(f"  Length: {row['crash_length']} days")
            print(f"  Day0 Drop: {row['day0_drop_pct']:.2f}%")
            print(f"  Total Drop: {row['total_drop_value']:.2f}")
            print(f"  Total %: {row['total_drop_percent']:.2f}%\n")

def print_single_index_crashes(single):
    print("\n====================================================")
    print("📊 FULL SINGLE-DAY CRASH LIST (PER INDEX)")
    print("====================================================\n")

    for idx, df in single.items():
        print(f"\n🟥 {idx.upper()} — Single-Day Crashes")
        print("------------------------------------------")

        if df.empty:
            print("✔ None\n")
            continue

        for _, row in df.iterrows():
            print(f"📅 {row['date'].date()}")
            print(f"  Open:  {row['open']}")
            print(f"  Close: {row['close']}")
            print(f"  Drop:  {row['drop_pct']:.2f}%\n")
def print_single_index_multi_crashes(multi):
    print("\n====================================================")
    print("📊 FULL MULTI-DAY CRASH WINDOWS (PER INDEX)")
    print("====================================================\n")

    for idx, df in multi.items():
        print(f"\n🟦 {idx.upper()} — Multi-Day Crash Windows")
        print("------------------------------------------")

        if df.empty:
            print("✔ None\n")
            continue

        for _, row in df.iterrows():
            print(f"📅 {row['start_date'].date()} → {row['end_date'].date()}")
            print(f"  Length: {row['crash_length']} days")
            print(f"  Day0 Drop: {row['day0_drop_pct']:.2f}%")
            print(f"  Total Drop: {row['total_drop_value']:.2f}")
            print(f"  Total %: {row['total_drop_percent']:.2f}%\n")

def print_2index_multi_details(overlaps):
    names = {
        "nasdaq_sp500": ("NASDAQ", "S&P500"),
        "nasdaq_dow": ("NASDAQ", "DOW"),
        "sp500_dow": ("S&P500", "DOW")
    }

    print("\n====================================================")
    print("📊 MULTI-DAY CRASH WINDOWS (ANY 2 INDEXES ONLY)")
    print("====================================================\n")

    for pair, wins in overlaps.items():
        n1, n2 = names[pair]

        print(f"\n🟨 {n1} & {n2}")
        print("------------------------------")

        if not wins:
            print("✔ None")
            continue

        for w in wins:
            print(f"\n📅 {w['common_start'].date()} → {w['common_end'].date()}\n")

            a = w["a"]
            b = w["b"]

            print(f"{n1}:")
            print(f"  Start: {a['start_date'].date()}")
            print(f"  End:   {a['end_date'].date()}")
            print(f"  Length: {a['crash_length']} days")
            print(f"  Day0 Drop: {a['day0_drop_pct']:.2f}%")
            print(f"  Total Drop: {a['total_drop_value']:.2f}\n")

            print(f"{n2}:")
            print(f"  Start: {b['start_date'].date()}")
            print(f"  End:   {b['end_date'].date()}")
            print(f"  Length: {b['crash_length']} days")
            print(f"  Day0 Drop: {b['day0_drop_pct']:.2f}%")
            print(f"  Total Drop: {b['total_drop_value']:.2f}\n")
import requests
from datetime import timedelta

# ============================================================
# GDELT NEWS SEARCH (NO API KEY REQUIRED)
# ============================================================
import requests

# ============================================================
# GNEWS.IO API — HISTORICAL NEWS SEARCH
# ============================================================
GNEWS_API_KEY = "212dbc3ebdb769bfcadc6242fd23d64d"   # <-- PUT YOUR API KEY HERE

import requests

NEWSDATA_API_KEY = "pub_32e092813b0c4217a3ccfc8a1c3f4ceb"

def search_newsdata(date_obj, query):
    date_str = date_obj.strftime("%Y-%m-%d")

    url = "https://newsdata.io/api/1/news"

    params = {
        "apikey": NEWSDATA_API_KEY,
        "q": query,
        "from_date": date_str,
        "to_date": date_str,
        "language": "en"
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        # If the API returns an error object
        if data.get("status") == "error":
            return []   # MUST return list

        # Safe fallback
        results = data.get("results", [])
        if not isinstance(results, list):
            return []

        return results

    except Exception as e:
        return []   # ALWAYS return list

def export_full_report_to_csv(
    single, multi,
    common_3_single, common_2_single,
    multi_3, multi_2,
    output_file="market_crash_report.csv"
):
    rows = []

    # ============================================================
    # RAW SINGLE-DAY CRASHES (PER INDEX)
    # ============================================================
    for idx, df in single.items():
        for _, r in df.iterrows():
            rows.append({
                "type": "single-day",
                "group": "1-index",
                "pair": idx.upper(),
                "index": idx.upper(),
                "date": r["date"].date(),
                "start_date": None,
                "end_date": None,
                "crash_length": 1,
                "open": r["open"],
                "close": r["close"],
                "drop_pct": r["drop_pct"],
                "total_drop_value": None,
                "total_drop_percent": None
            })

    # ============================================================
    # RAW MULTI-DAY CRASHES (PER INDEX)
    # ============================================================
    for idx, df in multi.items():
        for _, r in df.iterrows():
            rows.append({
                "type": "multi-day",
                "group": "1-index",
                "pair": idx.upper(),
                "index": idx.upper(),
                "date": None,
                "start_date": r["start_date"].date(),
                "end_date": r["end_date"].date(),
                "crash_length": r["crash_length"],
                "open": None,
                "close": None,
                "drop_pct": r["day0_drop_pct"],
                "total_drop_value": r["total_drop_value"],
                "total_drop_percent": r["total_drop_percent"]
            })

    # ============================================================
    # SAME-DAY — 3 INDEXES
    # ============================================================
    for d in common_3_single:
        for idx in ["nasdaq", "sp500", "dow"]:
            r = single[idx][single[idx]["date"] == d].iloc[0]

            rows.append({
                "type": "same-day",
                "group": "3-index",
                "pair": "ALL",
                "index": idx.upper(),
                "date": d.date(),
                "start_date": None,
                "end_date": None,
                "crash_length": 1,
                "open": r["open"],
                "close": r["close"],
                "drop_pct": r["drop_pct"],
                "total_drop_value": None,
                "total_drop_percent": None
            })

    # ============================================================
    # SAME-DAY — 2 INDEXES ONLY
    # ============================================================
    name_map = {
        "nasdaq_sp500": ("NASDAQ", "S&P500"),
        "nasdaq_dow": ("NASDAQ", "DOW"),
        "sp500_dow": ("S&P500", "DOW")
    }

    for pair, dates in common_2_single.items():
        n1, n2 = name_map[pair]
        idx1, idx2 = pair.split("_")

        for d in dates:
            r1 = single[idx1][single[idx1]["date"] == d].iloc[0]
            rows.append({
                "type": "same-day",
                "group": "2-index",
                "pair": f"{n1}-{n2}",
                "index": n1,
                "date": d.date(),
                "start_date": None,
                "end_date": None,
                "crash_length": 1,
                "open": r1["open"],
                "close": r1["close"],
                "drop_pct": r1["drop_pct"],
                "total_drop_value": None,
                "total_drop_percent": None
            })

            r2 = single[idx2][single[idx2]["date"] == d].iloc[0]
            rows.append({
                "type": "same-day",
                "group": "2-index",
                "pair": f"{n1}-{n2}",
                "index": n2,
                "date": d.date(),
                "start_date": None,
                "end_date": None,
                "crash_length": 1,
                "open": r2["open"],
                "close": r2["close"],
                "drop_pct": r2["drop_pct"],
                "total_drop_value": None,
                "total_drop_percent": None
            })

    # ============================================================
    # MULTI-DAY — 3 INDEXES
    # ============================================================
    for w in multi_3:
        for name, r in [
            ("NASDAQ", w["nasdaq"]),
            ("S&P500", w["sp500"]),
            ("DOW", w["dow"])
        ]:
            rows.append({
                "type": "multi-day",
                "group": "3-index",
                "pair": "ALL",
                "index": name,
                "date": None,
                "start_date": w["common_start"].date(),
                "end_date": w["common_end"].date(),
                "crash_length": r["crash_length"],
                "open": None,
                "close": None,
                "drop_pct": r["day0_drop_pct"],
                "total_drop_value": r["total_drop_value"],
                "total_drop_percent": r["total_drop_percent"]
            })

    # ============================================================
    # MULTI-DAY — 2 INDEXES ONLY
    # ============================================================
    for pair, wins in multi_2.items():
        n1, n2 = name_map[pair]

        for w in wins:
            a = w["a"]
            b = w["b"]

            rows.append({
                "type": "multi-day",
                "group": "2-index",
                "pair": f"{n1}-{n2}",
                "index": n1,
                "date": None,
                "start_date": w["common_start"].date(),
                "end_date": w["common_end"].date(),
                "crash_length": a["crash_length"],
                "open": None,
                "close": None,
                "drop_pct": a["day0_drop_pct"],
                "total_drop_value": a["total_drop_value"],
                "total_drop_percent": a["total_drop_percent"]
            })

            rows.append({
                "type": "multi-day",
                "group": "2-index",
                "pair": f"{n1}-{n2}",
                "index": n2,
                "date": None,
                "start_date": w["common_start"].date(),
                "end_date": w["common_end"].date(),
                "crash_length": b["crash_length"],
                "open": None,
                "close": None,
                "drop_pct": b["day0_drop_pct"],
                "total_drop_value": b["total_drop_value"],
                "total_drop_percent": b["total_drop_percent"]
            })

    # ============================================================
    # EXPORT CSV
    # ============================================================
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

    print(f"\n📁 CSV file generated: {output_file}\n")



# ============================================================
# 7. MAIN
# ============================================================
def main():

    print("\n📂 Loading Index Files...\n")
    dfs = {name: load_json_as_df(path) for name, path in INPUT_FILES.items()}

    print("\n📉 Detecting Crashes...\n")
    single = {k: detect_single_day_crashes(v) for k, v in dfs.items()}
    multi = {k: detect_multi_day_crashes(v) for k, v in dfs.items()}
    print_single_index_crashes(single)
    print_single_index_multi_crashes(multi)


    # ---------------------- SAME-DAY (3 INDEXES)
    common_3 = find_3index_same_day(single["nasdaq"], single["sp500"], single["dow"])
    print_3index_same_day_details(single, common_3)

    # ---------------------- SAME-DAY (2 INDEXES, EXCLUDING 3)
    common_2 = find_2index_same_day(single)
    common_2 = filter_out_3index_from_2index(common_2, common_3)

    print_2index_same_day_details(single, common_2)

    # ---------------------- MULTI-DAY (3 INDEXES)
    multi_3 = find_3index_multi(multi["nasdaq"], multi["sp500"], multi["dow"])
    print_3index_multi_details(multi_3)

    # ---------------------- MULTI-DAY (2 INDEXES, EXCLUDING 3)
    multi_2 = find_2index_multi(multi)
    multi_2 = filter_out_3index_multi(multi_2, multi_3)

    print_2index_multi_details(multi_2)
    export_full_report_to_csv(
    single, multi,
    common_3, common_2,
    multi_3, multi_2,
    output_file="market_crash_report_1.csv"
)
    
    all_crash_dates = set()

    # Single-day crashes
    for idx, df in single.items():
        for _, row in df.iterrows():
            if not pd.isna(row["date"]):
                all_crash_dates.add(row["date"])

    # Multi-day crashes → use start_date
    for idx, df in multi.items():
        for _, row in df.iterrows():
            all_crash_dates.add(row["start_date"])

    all_crash_dates = sorted(list(all_crash_dates))

    print("\n====================================================")
    print("📰 NEWS CAUSES FROM GNEWS API (ALL CRASH DATES)")
    print("====================================================\n")

    news_report = []  # store for CSV export

    for d in all_crash_dates:
        print(f"\n🔎 Searching news for crash date: {d.date()}")
        print("-----------------------------------------")

        articles = search_newsdata(
    d,
    query=(
        "\"market crash\" OR \"stock market crash\" OR \"market selloff\" OR "
        "\"stocks plunge\" OR \"stocks tumble\" OR \"market meltdown\" OR "
        "\"dow jones plunges\" OR \"nasdaq falls\" OR \"s&p 500 drops\" OR "
        "\"wall street tumbles\" OR \"broad market decline\" OR "
        "\"market rout\" OR \"market turmoil\" OR \"stock market panic\" OR "
        "\"selloff triggered\""
    )
)


        if not articles:
            print("❌ No news found.\n")
            continue

        for a in articles[:3]:   # print top 3
            print(f"📰 {a.get('title')}")
            print(f"URL: {a.get('url')}")
            print(f"Summary: {a.get('description')}\n")

            news_report.append({
                "date": d.date(),
                "title": a.get("title"),
                "url": a.get("url"),
                "description": a.get("description")
            })



if __name__ == "__main__":
    main()
