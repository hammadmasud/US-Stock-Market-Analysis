from flask import Flask, jsonify, render_template
import sqlite3
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "ScrapingLast1_Stocks.db")
CSV_PATH = os.path.join(BASE_DIR, "market5_crash_report_1.csv")


INDEX_LABELS = {
    "NASDAQ": "NASDAQ Composite",
    "S&P500": "S&P 500",
    "DOW": "Dow Jones Industrial Average",
}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_news(date_str: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT date, text, url, article_body, crash_reason
        FROM articles
        WHERE date = ?
        ORDER BY rowid ASC;
        """,
        (date_str,),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _first_line_text(s):
    if s is None:
        return ""
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return ""
    return s.splitlines()[0].strip()



def fetch_analysis_first_lines_by_date():
    """
    Returns { 'YYYY-MM-DD': 'analysis text' }
    Prefers main_cause_short. Falls back to crash_reason.
    Normalizes SQLite date to YYYY-MM-DD to match chart.
    """
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT date, main_cause_short, crash_reason
        FROM articles
        WHERE (main_cause_short IS NOT NULL AND TRIM(main_cause_short) <> '')
           OR (crash_reason IS NOT NULL AND TRIM(crash_reason) <> '')
        ORDER BY rowid ASC;
        """
    )
    rows = cur.fetchall()
    conn.close()

    out = {}
    for r in rows:
        d = pd.to_datetime(r["date"], errors="coerce")
        if pd.isna(d):
            continue
        key = d.strftime("%Y-%m-%d")
        if key in out:
            continue

        # Prefer main_cause_short, fallback to crash_reason
        main_short = _first_line_text(r["main_cause_short"])
        reason = _first_line_text(r["crash_reason"])
        out[key] = main_short or reason or ""

    return out


def _load_crash_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)

    # normalize index codes
    if "index" in df.columns:
        df["index"] = df["index"].astype(str).str.strip().replace({
            "SP500": "S&P500",
            "S&P-500": "S&P500",
            "SNP500": "S&P500",
        })

    # normalize type
    if "type" in df.columns:
        df["type_norm"] = df["type"].replace({"same-day": "single-day"})
    else:
        df["type_norm"] = None


    df = df.rename(columns={
        "recovery_d": "recovery_days",
        "recovery_days_": "recovery_days",
        "recovery": "recovery_date",
        "recovery_dt": "recovery_date",
        "recovery_date_": "recovery_date",
    })

    # parse dates
    for col in ["date", "start_date", "end_date", "recovery_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # numerics
    for col in ["drop_pct", "total_drop_percent", "crash_length", "recovery_days"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ensure drop_pct is negative for UI consistency
    if "drop_pct" in df.columns:
        df["drop_pct"] = -df["drop_pct"].abs()

    return df


crash_df = _load_crash_csv(CSV_PATH)
TOTAL_COL = "total_drop_percent" if (not crash_df.empty and "total_drop_percent" in crash_df.columns) else None


# ----------------------------
# Crash logic helpers
# ----------------------------
def _indexes_payload(indexes):
    indexes = sorted([i for i in indexes if isinstance(i, str) and i.strip()])
    return {
        "indexes": indexes,
        "index_names": [{"code": i, "name": INDEX_LABELS.get(i, i)} for i in indexes],
    }


def _fmt_date(dt):
    if dt is None or (isinstance(dt, float) and pd.isna(dt)) or pd.isna(dt):
        return None
    try:
        return pd.to_datetime(dt).strftime("%Y-%m-%d")
    except Exception:
        return None


def crash_metrics_for_date(date_str: str):
    """
    1-index only.

    single_day:
      rows where type_norm == single-day and date == selected date

    multi_day:
      rows where type_norm == multi-day and start_date == selected date
      (the crash START day only; aligns with /crash-drops chart)

    recovery_date:
      prefer CSV recovery_date, else fallback to end_date

    recovery_days:
      prefer CSV recovery_days, else compute from (end_date-start_date+1),
      else fallback to crash_length
    """
    if crash_df.empty:
        return {"error": f"CSV not found/loaded at: {CSV_PATH}"}

    d = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(d):
        return {"error": "Invalid date format. Use YYYY-MM-DD."}
    d = d.normalize()

    # --- Single-day: exact date match
    single = crash_df[
        crash_df["type_norm"].eq("single-day")
        & crash_df.get("date").notna()
        & (crash_df["date"].dt.normalize() == d)
    ].copy()

    # --- Multi-day: crash start day only (start_date == selected day)
    multi = crash_df[
        crash_df["type_norm"].eq("multi-day")
        & crash_df.get("start_date").notna()
        & (crash_df["start_date"].dt.normalize() == d)
    ].copy()

    # compute crash length if possible
    if not multi.empty:
        if "end_date" in multi.columns and multi["end_date"].notna().any():
            sd = multi["start_date"].dt.normalize()
            ed = multi["end_date"].dt.normalize()
            multi["crash_length_calc"] = (ed - sd).dt.days + 1
        else:
            multi["crash_length_calc"] = pd.to_numeric(multi.get("crash_length"), errors="coerce")

        multi["day_in_crash"] = 1

    def pack_rows(frame: pd.DataFrame):
        matches = []
        for _, r in frame.iterrows():
            # length
            cl = r.get("crash_length_calc", None)
            if cl is None or (isinstance(cl, float) and pd.isna(cl)):
                cl = r.get("crash_length", None)

            # day in crash
            di = r.get("day_in_crash", None)

            # first-day drop (negative already)
            first_day = None
            drop = r.get("drop_pct")
            if drop is not None and not pd.isna(drop):
                try:
                    first_day = float(drop)
                except Exception:
                    first_day = None

            # total drop
            total = None
            if TOTAL_COL is not None:
                t = r.get(TOTAL_COL)
                if t is not None and not pd.isna(t):
                    try:
                        total = float(t)
                    except Exception:
                        total = None

            # recovery_date: prefer explicit recovery_date else fallback to end_date
            rd = r.get("recovery_date", None) if "recovery_date" in frame.columns else None
            recovery_date = _fmt_date(rd) or _fmt_date(r.get("end_date"))

            # recovery_days:
            # prefer explicit recovery_days else compute from end-start+1 else fallback to crash_length
            recovery_days = None
            if "recovery_days" in frame.columns:
                rdays = r.get("recovery_days")
                if rdays is not None and not pd.isna(rdays):
                    try:
                        recovery_days = int(float(rdays))
                    except Exception:
                        recovery_days = None

            if recovery_days is None:
                sd = r.get("start_date")
                ed = r.get("end_date")
                if sd is not None and ed is not None and pd.notna(sd) and pd.notna(ed):
                    try:
                        recovery_days = int((ed.normalize() - sd.normalize()).days + 1)
                    except Exception:
                        recovery_days = None

            if recovery_days is None:
                try:
                    if cl is not None and not (isinstance(cl, float) and pd.isna(cl)):
                        recovery_days = int(float(cl))
                except Exception:
                    recovery_days = None

            matches.append({
                "type": r.get("type_norm", r.get("type")),
                "index": r.get("index"),
                "index_name": INDEX_LABELS.get(r.get("index"), r.get("index")),

                "first_day_drop_pct": first_day,

                "start_date": _fmt_date(r.get("start_date")),
                "end_date": _fmt_date(r.get("end_date")),

                "crash_length": None if cl is None or (isinstance(cl, float) and pd.isna(cl)) else int(float(cl)),
                "day_in_crash": None if di is None or (isinstance(di, float) and pd.isna(di)) else int(float(di)),

                "total_drop_percent": total,

                # always included for frontend
                "recovery_date": recovery_date,
                "recovery_days": recovery_days,
            })
        return matches

    out = {
        "date": date_str,
        "worst_first_day_drop": None,
        "single_day": {"has_event": False, "matches": [], **_indexes_payload([])},
        "multi_day": {
            "has_event": False,
            "matches": [],
            "start_date": None,
            "end_date": None,
            "crash_length": None,
            "day_in_crash": None,
            "recovery_date": None,
            "recovery_days": None,
            **_indexes_payload([]),
        },
        "notes": [
            "1-index only (no 2-index/3-index grouping).",
            "same-day is treated as single-day.",
            "drop_pct is treated as first-day drop (forced negative).",
            "multi-day matches are returned when start_date == selected date (crash start day).",
            "recovery_date prefers CSV recovery_date else falls back to end_date.",
            "recovery_days prefers CSV recovery_days else computes from end-start+1 else falls back to crash_length.",
        ],
    }

    if not single.empty:
        out["single_day"]["has_event"] = True
        sidx = single["index"].dropna().unique().tolist() if "index" in single.columns else []
        out["single_day"].update(_indexes_payload(sidx))
        out["single_day"]["matches"] = pack_rows(single)

    if not multi.empty:
        out["multi_day"]["has_event"] = True
        midx = multi["index"].dropna().unique().tolist() if "index" in multi.columns else []
        out["multi_day"].update(_indexes_payload(midx))
        out["multi_day"]["matches"] = pack_rows(multi)

        first = multi.iloc[0]
        out["multi_day"]["start_date"] = _fmt_date(first.get("start_date"))
        out["multi_day"]["end_date"] = _fmt_date(first.get("end_date"))
        if pd.notna(first.get("crash_length_calc", None)):
            out["multi_day"]["crash_length"] = int(float(first["crash_length_calc"]))
        out["multi_day"]["day_in_crash"] = 1

        # summary-level recovery fields
        first_packed = out["multi_day"]["matches"][0] if out["multi_day"]["matches"] else {}
        out["multi_day"]["recovery_date"] = first_packed.get("recovery_date")
        out["multi_day"]["recovery_days"] = first_packed.get("recovery_days")

    # worst first-day drop across both sections
    all_matches = out["single_day"]["matches"] + out["multi_day"]["matches"]
    vals = []
    for m in all_matches:
        v = m.get("first_day_drop_pct")
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            pass
    out["worst_first_day_drop"] = min(vals) if vals else None

    return out


# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("Stocks5.html")


@app.route("/ping")
def ping():
    return "OK"


@app.route("/news/<date_str>", methods=["GET"])
def news(date_str):
    return jsonify(fetch_news(date_str))


@app.route("/bundle/<date_str>", methods=["GET"])
def bundle(date_str):
    return jsonify({
        "date": date_str,
        "news": fetch_news(date_str),
        "crash": crash_metrics_for_date(date_str),
    })


@app.route("/crash-analysis-firstline", methods=["GET"])
def crash_analysis_firstline():
    return jsonify({"analysis": fetch_analysis_first_lines_by_date()})



@app.route("/crash-drops", methods=["GET"])
def crash_drops():
    """
    Per chart date:
      - If multi-day crash starts that day: use worst (most negative) total_drop_percent
      - Else (single-day only): use worst (most negative) drop_pct (first day)

    Also returns recovery_dates aligned with dates:
      - For multi-day dates: take recovery_date from the row that produced the worst_total_drop (prefer recovery_date else end_date)
      - For single-day-only dates: show None (or you can choose end_date if you want)
    """
    if crash_df.empty:
        return jsonify({
            "dates": [], "drops": [], "drop_kind": [],
            "recovery_dates": [], "recovery_days": [],
            "error": f"CSV not loaded: {CSV_PATH}"
        }), 200

    df = crash_df.copy()

    
    df["drop_pct"] = pd.to_numeric(df.get("drop_pct"), errors="coerce")
    df["drop_pct"] = -df["drop_pct"].abs()

    has_total = (TOTAL_COL is not None and TOTAL_COL in df.columns)
    if has_total:
        df[TOTAL_COL] = pd.to_numeric(df.get(TOTAL_COL), errors="coerce")
        df[TOTAL_COL] = -df[TOTAL_COL].abs()

    
    df["chart_date"] = pd.NaT
    single_mask = df["type_norm"].eq("single-day") & df.get("date").notna()
    multi_mask  = df["type_norm"].eq("multi-day")  & df.get("start_date").notna()

    df.loc[single_mask, "chart_date"] = df.loc[single_mask, "date"].dt.normalize()
    df.loc[multi_mask,  "chart_date"] = df.loc[multi_mask,  "start_date"].dt.normalize()

    df = df[df["chart_date"].notna()].copy()
    df["date_str"] = df["chart_date"].dt.strftime("%Y-%m-%d")

    
    def _fmt(dt):
        if dt is None or (isinstance(dt, float) and pd.isna(dt)) or pd.isna(dt):
            return None
        try:
            return pd.to_datetime(dt).strftime("%Y-%m-%d")
        except Exception:
            return None

   
    single_df = df[single_mask & df["drop_pct"].notna()].copy()
    worst_first = single_df.groupby("date_str")["drop_pct"].min()

    
    worst_total = pd.Series(dtype=float)
    multi_winners = {}  

    if has_total:
        multi_df = df[multi_mask & df[TOTAL_COL].notna()].copy()

        if not multi_df.empty:
            
            winner_idx = multi_df.groupby("date_str")[TOTAL_COL].idxmin()
            winners = multi_df.loc[winner_idx].copy()

           
            worst_total = winners.set_index("date_str")[TOTAL_COL]

            # Keep the full row for recovery fields
            for _, row in winners.iterrows():
                multi_winners[row["date_str"]] = row

    # union of dates
    all_dates = sorted(set(worst_first.index.tolist()) | set(worst_total.index.tolist()))

    drops = []
    drop_kind = []
    recovery_dates = []
    recovery_days = []

    for d in all_dates:
        if d in worst_total.index:
            drops.append(float(worst_total.loc[d]))
            drop_kind.append("total")

            row = multi_winners.get(d)
            if row is not None:
                # Prefer recovery_date column if present, else end_date
                rd = row.get("recovery_date", None) if "recovery_date" in row.index else None
                rec_date = _fmt(rd) or _fmt(row.get("end_date"))
                recovery_dates.append(rec_date)

                # Prefer recovery_days if present (numeric), else compute end-start+1, else None
                rdays = row.get("recovery_days", None) if "recovery_days" in row.index else None
                rec_days = None
                if rdays is not None and not pd.isna(rdays):
                    try:
                        rec_days = int(float(rdays))
                    except Exception:
                        rec_days = None
                if rec_days is None:
                    sd = row.get("start_date")
                    ed = row.get("end_date")
                    if sd is not None and ed is not None and pd.notna(sd) and pd.notna(ed):
                        try:
                            rec_days = int((pd.to_datetime(ed).normalize() - pd.to_datetime(sd).normalize()).days + 1)
                        except Exception:
                            rec_days = None

                recovery_days.append(rec_days)
            else:
                recovery_dates.append(None)
                recovery_days.append(None)

        else:
            drops.append(float(worst_first.loc[d]))
            drop_kind.append("first_day")

            
            recovery_dates.append(None)
            recovery_days.append(None)

    return jsonify({
        "dates": all_dates,
        "drops": drops,
        "drop_kind": drop_kind,
        "recovery_dates": recovery_dates,  
        "recovery_days": recovery_days,    
        "has_total": bool(has_total),
    })

if __name__ == "__main__":
    app.run(debug=True)
