"""
COMPLETE CRASH ANALYSIS PIPELINE WITH CHUNKING (WINDOWED + CRASH REPORT CONTEXT)
Loads → Cleans → FinBERT → (Neg ≥ 45%) → Chunking → LLM Batch Analysis → Synthesis (+Indexes Crashed)
→ 5-Line Summary → (NEW: 3–4 word Main Cause Label) → DB

CHANGE REQUESTS APPLIED (YOUR ORIGINAL COMMENTS + CODE):
1) Windowed load around center date (your variables currently use ±2 days; change to 3/3 if desired)
2) Inject crash report "indexes crashed" into FINAL SUMMARY prompt
3) Parallel local LLM calls for batch chunk analysis using ThreadPoolExecutor + shared requests.Session
4) Fix duplicate "Line X: Line X:" issue
5) Save crash_reason as 5 plain lines (NO "Line X:" prefixes)
6) NEW: Ask LLM for 3–4 word main cause label and store in DB column main_cause_short
"""

import sqlite3
import pandas as pd
import torch
import re
import requests
import tiktoken
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
from datetime import datetime, timedelta

# ✅ parallel HTTP
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DB_PATH = "ScrapingLast1_Stocks.db"
LOCAL_LLM_URL = "http://192.168.1.150:8000/v1/chat/completions"
LOCAL_MODEL_NAME = "gpt-oss-20b"
NEG_THRESHOLD = 0.55

MAX_CHUNK_TOKENS = 3500
MAX_CHUNKS_PER_BATCH = 3
OVERLAP_TOKENS = 100

# NOTE: Your header comment said ±3, but your code is ±2. Keep as-is unless you change it.
WINDOW_DAYS_BEFORE = 3
WINDOW_DAYS_AFTER = 3

CRASH_REPORT_PATH = "market_crash_report_1.csv"
CRASH_REPORT_DATE_COL = "date"
CRASH_REPORT_INDEX_COL = "index"

CRASH_REPORT_FLAG_COL = None
CRASH_REPORT_FLAG_TRUE_VALUES = {"1", "true", "yes", "y"}

# ✅ parallel config for local LLM
LLM_MAX_WORKERS = 3  # tune 2–8 depending on server capacity
LLM_TIMEOUT = 300


def make_llm_session():
    s = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        pool_connections=LLM_MAX_WORKERS,
        pool_maxsize=LLM_MAX_WORKERS,
        max_retries=retry
    )
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


LLM_SESSION = make_llm_session()


try:
    llm_encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    llm_encoding = None
    print("Warning: tiktoken not installed. Using character-based chunking.")
    print("Install with: pip install tiktoken")
import transformers
import torch
from transformers import AutoModelForSequenceClassification
import os


cache_path = "C:\\Users\\hamma\\.cache\\huggingface\\hub\\models--ProsusAI--finbert"
if os.path.exists(cache_path):
    import shutil
    shutil.rmtree(cache_path)

# Also clear PyTorch cache if needed
torch_cache = os.path.expanduser("~/.cache/torch")
if os.path.exists(os.path.join(torch_cache, "transformers")):
    shutil.rmtree(os.path.join(torch_cache, "transformers"))

# Now try loading the model again
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")


# ----------------------------
# Crash report helpers
# ----------------------------
def load_crash_report(path: str) -> pd.DataFrame:
    """Load crash report from CSV or Excel into a DataFrame."""
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    raise ValueError("Unsupported crash report format. Use .csv, .xlsx, or .xls")


def normalize_report_date(x) -> str:
    """Convert different date formats into YYYY-MM-DD string."""
    if pd.isna(x):
        return ""
    dt = pd.to_datetime(x, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        return ""
    return dt.strftime("%Y-%m-%d")


def build_crash_index_lookup(report_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Builds: {'YYYY-MM-DD': ['IndexA', 'IndexB', ...]}
    Optionally filters by a crash flag column if provided.
    """
    df = report_df.copy()

    if CRASH_REPORT_DATE_COL not in df.columns:
        raise KeyError(f"Crash report missing date column: {CRASH_REPORT_DATE_COL}")
    if CRASH_REPORT_INDEX_COL not in df.columns:
        raise KeyError(f"Crash report missing index column: {CRASH_REPORT_INDEX_COL}")

    df[CRASH_REPORT_DATE_COL] = df[CRASH_REPORT_DATE_COL].apply(normalize_report_date)
    df = df[df[CRASH_REPORT_DATE_COL].astype(str).str.len() == 10]

    if CRASH_REPORT_FLAG_COL:
        if CRASH_REPORT_FLAG_COL not in df.columns:
            raise KeyError(f"Crash report missing flag column: {CRASH_REPORT_FLAG_COL}")
        flag = df[CRASH_REPORT_FLAG_COL].astype(str).str.strip().str.lower()
        df = df[flag.isin({v.lower() for v in CRASH_REPORT_FLAG_TRUE_VALUES})]

    df[CRASH_REPORT_INDEX_COL] = df[CRASH_REPORT_INDEX_COL].astype(str).str.strip()
    grouped = df.groupby(CRASH_REPORT_DATE_COL)[CRASH_REPORT_INDEX_COL].apply(list).to_dict()

    deduped = {}
    for d, idxs in grouped.items():
        seen = set()
        out = []
        for it in idxs:
            if it and it not in seen:
                seen.add(it)
                out.append(it)
        deduped[d] = out

    return deduped


def get_crash_indexes_for_date(date_string: str, lookup: Dict[str, List[str]]) -> str:
    idxs = lookup.get(date_string, [])
    return ", ".join(idxs) if idxs else "None listed in crash report"


# ----------------------------
# DB schema helpers
# ----------------------------
def ensure_crash_reason_column():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(articles);")
    columns = [col[1] for col in cursor.fetchall()]

    if "crash_reason" not in columns:
        cursor.execute("ALTER TABLE articles ADD COLUMN crash_reason TEXT;")
        print("✓ Added crash_reason column to database")
    else:
        print("✓ crash_reason column already exists")

    conn.commit()
    conn.close()


# ✅ NEW: column for 3–4 word label
def ensure_main_cause_column():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(articles);")
    columns = [col[1] for col in cursor.fetchall()]

    if "main_cause_short" not in columns:
        cursor.execute("ALTER TABLE articles ADD COLUMN main_cause_short TEXT;")
        print("✓ Added main_cause_short column to database")
    else:
        print("✓ main_cause_short column already exists")

    conn.commit()
    conn.close()


# ----------------------------
# Cleaning + DB window load
# ----------------------------
def _clean_text(x: str) -> str:
    x = str(x) if x is not None else ""
    x = re.sub(r"\s+", " ", x)
    x = x.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    return x.strip()


def load_news_from_db_window(center_date: str, days_before: int = 3, days_after: int = 3) -> pd.DataFrame:
    center = datetime.strptime(center_date, "%Y-%m-%d").date()
    start = center - timedelta(days=days_before)
    end = center + timedelta(days=days_after)

    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT date, article_body
        FROM articles
        WHERE date BETWEEN ? AND ?
          AND article_body IS NOT NULL
          AND TRIM(article_body) <> ''
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(start.isoformat(), end.isoformat()))
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=["date", "news"])

    df.rename(columns={"article_body": "news"}, inplace=True)
    df["news"] = df["news"].astype(str).apply(_clean_text)
    df = df[df["news"].str.len() > 0].reset_index(drop=True)
    return df

def finbert_sentiment(text: str) -> Dict[str, float]:
    encoding = finbert_tokenizer.encode(text, add_special_tokens=True)
    max_len = 1500
    chunk_size = 510

    if len(encoding) <= max_len:
        inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = finbert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return {
            "positive": float(probs[0][0].detach()),
            "negative": float(probs[0][1].detach()),
            "neutral": float(probs[0][2].detach()),
        }

    token_ids = encoding[1:-1]
    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        chunk_ids = token_ids[i:i + chunk_size]
        chunk_ids = [finbert_tokenizer.cls_token_id] + chunk_ids + [finbert_tokenizer.sep_token_id]
        chunks.append(chunk_ids)

    sent_scores = []
    for chunk in chunks:
        inputs = {
            "input_ids": torch.tensor([chunk]),
            "attention_mask": torch.ones(1, len(chunk), dtype=torch.long),
        }
        outputs = finbert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        sent_scores.append(probs.detach())

    sent_scores = torch.cat(sent_scores, dim=0)
    avg_scores = torch.mean(sent_scores, dim=0)

    return {
        "positive": float(avg_scores[0]),
        "negative": float(avg_scores[1]),
        "neutral": float(avg_scores[2]),
    }


def process_news_dataframe(df: pd.DataFrame):
    results = []
    for _, row in df.iterrows():
        news_text = row["news"]
        if not isinstance(news_text, str) or not news_text.strip():
            continue

        results.append({
            "date": row["date"],
            "text": news_text,
            "sentiment": finbert_sentiment(news_text)
        })
    return results


def filter_negative(news_list, threshold=NEG_THRESHOLD):
    return [item for item in news_list if item["sentiment"]["negative"] >= threshold]


# ----------------------------
# Chunking
# ----------------------------
def chunk_article_preserving_order(article_text: str, article_idx: int, neg_score: float) -> List[Dict]:
    chunks = []

    if llm_encoding:
        tokens = llm_encoding.encode(article_text)

        if len(tokens) <= MAX_CHUNK_TOKENS:
            chunk_text = llm_encoding.decode(tokens)
            chunks.append({
                "text": chunk_text,
                "article_idx": article_idx,
                "chunk_num": 0,
                "total_chunks": 1,
                "neg_score": neg_score,
                "tokens": len(tokens),
                "is_full": True
            })
        else:
            chunk_size = MAX_CHUNK_TOKENS - OVERLAP_TOKENS
            num_chunks = (len(tokens) + chunk_size - 1) // chunk_size

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = start_idx + MAX_CHUNK_TOKENS
                chunk_tokens = tokens[start_idx:end_idx]
                if chunk_tokens:
                    chunk_text = llm_encoding.decode(chunk_tokens)
                    chunks.append({
                        "text": chunk_text,
                        "article_idx": article_idx,
                        "chunk_num": chunk_idx,
                        "total_chunks": num_chunks,
                        "neg_score": neg_score,
                        "tokens": len(chunk_tokens),
                        "is_full": False
                    })
    else:
        max_chars = MAX_CHUNK_TOKENS * 4
        overlap_chars = OVERLAP_TOKENS * 4

        if len(article_text) <= max_chars:
            chunks.append({
                "text": article_text,
                "article_idx": article_idx,
                "chunk_num": 0,
                "total_chunks": 1,
                "neg_score": neg_score,
                "tokens": len(article_text) // 4,
                "is_full": True
            })
        else:
            paragraphs = [p for p in article_text.split("\n\n") if p.strip()]
            current_chunk = []
            current_length = 0
            chunk_idx = 0

            for para in paragraphs:
                para_length = len(para)

                if current_length + para_length > max_chars and current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "article_idx": article_idx,
                        "chunk_num": chunk_idx,
                        "total_chunks": 0,
                        "neg_score": neg_score,
                        "tokens": current_length // 4,
                        "is_full": False
                    })
                    chunk_idx += 1

                    overlap_para = current_chunk[-1] if current_chunk else ""
                    if overlap_para and len(overlap_para) < overlap_chars:
                        current_chunk = [overlap_para, para]
                    else:
                        current_chunk = [para]
                    current_length = sum(len(p) for p in current_chunk)
                else:
                    current_chunk.append(para)
                    current_length += para_length

            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "article_idx": article_idx,
                    "chunk_num": chunk_idx,
                    "total_chunks": chunk_idx + 1,
                    "neg_score": neg_score,
                    "tokens": current_length // 4,
                    "is_full": False
                })

            for chunk in chunks:
                if chunk["total_chunks"] == 0:
                    chunk["total_chunks"] = chunk_idx + 1

    return chunks


# ----------------------------
# LLM call
# ----------------------------
def call_llm(prompt: str, max_tokens: int = 1500) -> str:
    payload = {
        "model": LOCAL_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": False,
    }

    r = None
    try:
        r = LLM_SESSION.post(LOCAL_LLM_URL, json=payload, timeout=LLM_TIMEOUT)

        # ✅ show server error body
        if r.status_code != 200:
            return (
                f"ERROR: HTTP {r.status_code}\n"
                f"URL: {LOCAL_LLM_URL}\n"
                f"Response: {r.text[:2000]}"
            )

        data = r.json()
        if isinstance(data, dict) and data.get("error"):
            return f"ERROR: LLM error payload: {data['error']}"

        return (data["choices"][0]["message"]["content"] or "").strip()

    except Exception as e:
        extra = f"\nResponse body: {getattr(r, 'text', '')[:2000]}" if r is not None else ""
        return f"ERROR: Exception: {e}{extra}"


def process_chunk_batch(batch_chunks: List[Dict], batch_num: int, total_batches: int) -> str:
    print(f"    Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")

    formatted_chunks = []
    for chunk in batch_chunks:
        article_num = chunk["article_idx"] + 1

        if chunk["total_chunks"] > 1:
            chunk_info = f"[Article {article_num}, Part {chunk['chunk_num'] + 1}/{chunk['total_chunks']}]"
        else:
            chunk_info = f"[Article {article_num}]"

        chunk_info += f" (Negativity: {chunk['neg_score']:.2f})"
        formatted_chunks.append(f"{chunk_info}\n{chunk['text']}")

    news_text_block = "\n\n" + "=" * 60 + "\n\n".join(formatted_chunks) + "\n" + "=" * 60

    prompt = f"""
BATCH ANALYSIS - Batch {batch_num} of {total_batches}

Goal: Extract ONLY explicit facts from these news segments and organize them into:
(1) lead-up drivers BEFORE/ON the crash date,
(2) What can be the Main Causes of the Crash,

NEWS SEGMENTS (original order):
{news_text_block}

SYNTHESIS TASK: Create comprehensive crash analysis with 5-line summary

STRUCTURE YOUR RESPONSE AS FOLLOWS:

I. PRIMARY CAUSES (Prioritize by frequency across batches)
1. [Most frequent issue]: [Evidence from batches X, Y, Z]
2. [Second issue]: [Evidence from batches X, Y]
3. [Third issue]: [Evidence from batches Y, Z]
4. [Additional factors]: [Evidence from specific batches]

II. MARKET MECHANISM
• Initial Trigger: [What started the selloff?]
• Amplification: [How did it spread/accelerate?]
• Sector Impact: [Which sectors were hardest hit?]
• Investor Psychology: [Fear, panic, capitulation patterns]

III. KEY EVIDENCE
• Direct Quotes: "[Important quote]" (Batch X)
• Data Points: [Specific numbers, percentages] (Batch Y)
• Expert Statements: [Analyst/CEO comments] (Batch Z)

IV. 5 LINE SUMMARY (CRITICAL - USE EXACT FORMAT BELOW)

 [Primary catalyst - what directly caused the crash?]
 [Key amplifying factor - what made it worse?]
 [Most affected sectors/companies - who was hit hardest?]
 [Investor/market reaction - how did participants respond?]


FORMATTING REQUIREMENTS:
• Each summary line MUST begin with "Line X: " where X is 1-5
• Each line should be 15-25 words, clear and concise
• Line 1 states the MAIN trigger
• Line 5 includes specific numbers if mentioned in analyses
• Use professional financial reporting language

"""
    return call_llm(prompt, max_tokens=1000)


def synthesize_all_analyses(batch_analyses: List[str], negative_news: List[Dict], date: str, crash_indexes_str: str) -> str:
    print(f"    Synthesizing {len(batch_analyses)} batch analyses...")

    analyses_text = "\n\n" + "═" * 70 + "\n"
    for i, analysis in enumerate(batch_analyses):
        analyses_text += f"\nBATCH {i+1} ANALYSIS:\n{'─'*40}\n{analysis}\n"
    analyses_text += "\n" + "═" * 70

    prompt = f"""
AUTHORITATIVE CRASH CONTEXT (do not add indexes):
• Indexes reported as crashed on {date}: {crash_indexes_str}

WINDOW:
• From {WINDOW_DAYS_BEFORE} days before to {WINDOW_DAYS_AFTER} days after
• Total batch analyses: {len(batch_analyses)}

BATCH FACTS (authoritative for this answer):
{analyses_text}
{'='*80}



OUTPUT REQUIREMENTS:
Return EXACTLY 5 lines.
Each line = 15–25 words, concise, professional, factual.
No bullets, no extra commentary.

Now produce the 5 lines for {date}.
Words should be simpler
"""
    return call_llm(prompt, max_tokens=3000)


def generate_llm_explanation_with_chunking(negative_news: List[Dict], date: str, crash_indexes_str: str) -> str:
    if not negative_news:
        print("    No negative articles found - generating fallback explanation")
        return f"""
Line 1: Declines may reflect technical flows, positioning, or unobserved macro/geopolitical headlines outside this dataset
Line 2: Crash report lists impacted indexes as: {crash_indexes_str}, but text evidence in-window was insufficient to attribute causes
Line 3: Manual review of broader sources is recommended to validate the driver narrative for this center date
""".strip()

    print(f"    Processing {len(negative_news)} negative articles with chunking...")

    all_chunks = []
    total_tokens = 0
    for idx, article in enumerate(negative_news):
        article_chunks = chunk_article_preserving_order(
            article["text"],
            article_idx=idx,
            neg_score=article["sentiment"]["negative"]
        )
        all_chunks.extend(article_chunks)
        total_tokens += sum(chunk["tokens"] for chunk in article_chunks)

    print(f"    Created {len(all_chunks)} chunks ({total_tokens:,} estimated tokens)")

    batches = []
    total_batches = (len(all_chunks) + MAX_CHUNKS_PER_BATCH - 1) // MAX_CHUNKS_PER_BATCH
    for batch_start in range(0, len(all_chunks), MAX_CHUNKS_PER_BATCH):
        batch_end = min(batch_start + MAX_CHUNKS_PER_BATCH, len(all_chunks))
        batches.append(all_chunks[batch_start:batch_end])

    results_by_batch = {}

    with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as ex:
        futures = {}
        for batch_num, batch in enumerate(batches, start=1):
            futures[ex.submit(process_chunk_batch, batch, batch_num, total_batches)] = batch_num

        for fut in as_completed(futures):
            batch_num = futures[fut]
            try:
                results_by_batch[batch_num] = fut.result()
            except Exception as e:
                results_by_batch[batch_num] = f"ERROR: Batch {batch_num} failed - {e}"

    batch_analyses = [results_by_batch[i] for i in range(1, total_batches + 1)]

    if len(batch_analyses) == 0:
        return "Error: No batch analyses generated"
    elif len(batch_analyses) == 1:
        single_batch_text = batch_analyses[0]
        enhancement_prompt = f"""
Convert this single batch analysis into exactly 5 summary lines.
Include the crashed indexes context: {crash_indexes_str}

ANALYSIS:
{single_batch_text}

Return ONLY these 5 lines in order (you may include "Line 1:" etc, it will be stripped later):
1. SPECIFIC EVENTS: What concrete events, announcements, or data are mentioned?
2. MARKET MOVERS: Which companies, sectors, or indices are affected?
3. MAJOR CAUSES: Which factors are described as causing market declines/crashes?
"""
        return call_llm(enhancement_prompt, max_tokens=700)

    return synthesize_all_analyses(batch_analyses, negative_news, date, crash_indexes_str)


# ----------------------------
# Summary parsing + saving
# ----------------------------

def strip_line_prefix(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    while True:
        new_s = re.sub(r'^\s*Line\s*\d+\s*:\s*', '', s, flags=re.IGNORECASE)
        if new_s == s:
            break
        s = new_s.strip()
    return s


def extract_final_summary(explanation_text: str) -> List[str]:
    if not explanation_text:
        return [
            "No explanation returned by analysis system",
            "Technical error in processing pipeline",
            "Unable to determine market crash causes",
            "Please check data availability and parameters",
            "Analysis failed to generate valid summary",
        ]

    lines = [l.strip() for l in explanation_text.split("\n") if l.strip()]
    collected = []

    for l in lines:
        if re.match(r'^\s*Line\s*[1-5]\s*:', l, flags=re.IGNORECASE):
            collected.append(strip_line_prefix(l))

    if len(collected) >= 5:
        return collected[:5]

    non_empty = [strip_line_prefix(l) for l in lines if len(l) > 15]
    if non_empty:
        last_lines = non_empty[-5:] if len(non_empty) >= 5 else non_empty
        while len(last_lines) < 5:
            last_lines.append("[Additional analysis not available]")
        return last_lines[:5]

    return [
        "Analysis completed but summary extraction failed",
        "LLM response format was unexpected",
        "Technical issue in parsing the explanation",
        "Raw data may need manual review",
        "Pipeline requires debugging for this date",
    ]


def save_crash_reason(date: str, crash_reason_lines: List[str]) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    lines = [(x or "").strip() for x in (crash_reason_lines or [])]
    lines = [strip_line_prefix(x) for x in lines if x]

    if len(lines) < 5:
        for _ in range(len(lines), 5):
            lines.append("[Information incomplete]")
    elif len(lines) > 5:
        lines = lines[:5]

    formatted_reason = "\n".join(lines)

    cursor.execute("""
        UPDATE articles
        SET crash_reason = ?
        WHERE date = ?
    """, (formatted_reason, date))

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM articles WHERE date = ? AND crash_reason IS NOT NULL", (date,))
    updated_count = cursor.fetchone()[0]

    conn.close()
    return updated_count


# ✅ NEW: generate 3–4 word label from the final summary (1 extra LLM call per date)
def generate_main_cause_label(summary_lines: List[str], date: str, crash_indexes_str: str) -> str:
    facts = "\n".join([f"- {l}" for l in (summary_lines or [])])

    prompt = f"""
Task: Produce a SHORT label naming the main cause of the crash.

CONTEXT:
Date: {date}
Indexes: {crash_indexes_str}

SOURCE FACTS (only use these):
{facts}

RULES:
- Output ONLY 1 or 2 label, 5 to 6 words maximum.
- No punctuation, no quotes.
- Use simple words.
- If unclear from facts, output exactly: Insufficient evidence

Now output the label:
""".strip()

    out = call_llm(prompt, max_tokens=20)
    out = (out or "").strip()
    if not out:
        return "Insufficient evidence"

    line = out.splitlines()[0].strip()
    if line.lower() == "insufficient evidence":
        return "Insufficient evidence"

    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", line)

    banned = {"crash", "market"}
    words = [w for w in words if w.lower() not in banned]

    if len(words) < 2:
        return "Insufficient evidence"

    return " ".join(words[:4])


def save_main_cause_label(date: str, label: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE articles
        SET main_cause_short = ?
        WHERE date = ?
    """, ((label or "").strip(), date))

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM articles WHERE date = ? AND main_cause_short IS NOT NULL", (date,))
    updated_count = cursor.fetchone()[0]

    conn.close()
    return updated_count


# ----------------------------
# Main pipeline per date
# ----------------------------
def crash_reason_pipeline(date_string: str, crash_lookup: Dict[str, List[str]]):
    print(f"\n📊 Processing {date_string}")
    print("  └─ Loading articles (± window)...")

    crash_indexes_str = get_crash_indexes_for_date(date_string, crash_lookup)

    df = load_news_from_db_window(
        date_string,
        days_before=WINDOW_DAYS_BEFORE,
        days_after=WINDOW_DAYS_AFTER
    )

    if df.empty:
        print("  └─ ⚠️ No usable articles found in window")
        summary_lines = [
            "No usable articles found in the 7-day window for this date",
            "Article body was empty or null in database for the selected range",
            f"Crash report lists impacted indexes as: {crash_indexes_str}",
            "Check data collection and scraping pipeline for this window",
            "Manual analysis required for market movements",
        ]
        save_crash_reason(date_string, summary_lines)

        # save label too (no LLM call here)
        label_rows = save_main_cause_label(date_string, "Insufficient evidence")
        print(f"  └─ ✓ Saved main_cause_short to {label_rows} rows: Insufficient evidence")

        return [], [], summary_lines

    min_d = df["date"].min() if "date" in df.columns and not df.empty else "N/A"
    max_d = df["date"].max() if "date" in df.columns and not df.empty else "N/A"

    print(f"  └─ ✓ Loaded {len(df)} articles from {min_d} to {max_d}")
    print(f"  └─ Crash report indexes on {date_string}: {crash_indexes_str}")
    print("  └─ Analyzing sentiment with FinBERT...")

    all_news = process_news_dataframe(df)
    negative_news = filter_negative(all_news, NEG_THRESHOLD)

    print(f"  └─ ✓ Found {len(negative_news)} negative articles (≥{NEG_THRESHOLD*100}%)")

    if len(negative_news) == 0:
        print("  └─ No negative articles - using fallback explanation")
        explanation = f"""
Line 1: No negative news catalyst identified around {date_string} within the selected 7-day window of articles
Line 2: All window articles were below {NEG_THRESHOLD*100:.0f}% negativity threshold by FinBERT sentiment scoring
Line 3: Market movement may reflect technical flows, positioning, liquidity conditions, or missing headlines not captured in sources
Line 4: Crash report lists impacted indexes as: {crash_indexes_str}, but text evidence in-window was insufficient to attribute causes
Line 5: Manual investigation is recommended to validate the market driver narrative for this center date
""".strip()
    else:
        print("  └─ Generating explanation with chunking (parallel LLM batches)...")
        explanation = generate_llm_explanation_with_chunking(negative_news, date_string, crash_indexes_str)

    print("  └─ Extracting 5-line summary...")
    summary_lines = extract_final_summary(explanation)

    # ✅ NEW: produce 3–4 word label based on summary (1 LLM call)
    print("  └─ Generating 3–4 word main cause label...")
    main_label = generate_main_cause_label(summary_lines, date_string, crash_indexes_str)

    print("  └─ Saving to database (center date only)...")
    updated_rows = save_crash_reason(date_string, summary_lines)

    # ✅ NEW: save to DB
    label_rows = save_main_cause_label(date_string, main_label)

    print(f"  └─ ✓ Saved crash_reason to {updated_rows} rows in database")
    print(f"  └─ ✓ Saved main_cause_short to {label_rows} rows: {main_label}")

    return all_news, negative_news, summary_lines


# ----------------------------
# Run pipeline for date list
# ----------------------------
def run_pipeline_for_dates(date_list):
    print("\n" + "=" * 80)
    print("CRASH ANALYSIS PIPELINE WITH CHUNKING (WINDOWED + CRASH REPORT CONTEXT)")
    print("=" * 80)
    print("Configuration:")
    print(f"  • Database: {DB_PATH}")
    print(f"  • LLM Endpoint: {LOCAL_LLM_URL}")
    print(f"  • Model: {LOCAL_MODEL_NAME}")
    print(f"  • Negativity Threshold: {NEG_THRESHOLD*100}%")
    print(f"  • Max Chunk Tokens: {MAX_CHUNK_TOKENS:,}")
    print(f"  • Chunks per Batch: {MAX_CHUNKS_PER_BATCH}")
    print(f"  • Window: -{WINDOW_DAYS_BEFORE} days / +{WINDOW_DAYS_AFTER} days")
    print(f"  • Crash Report: {CRASH_REPORT_PATH}")
    print(f"  • Parallel LLM Workers: {LLM_MAX_WORKERS}")
    print(f"  • Dates to Process: {len(date_list)}")
    print("=" * 80)

    ensure_crash_reason_column()
    ensure_main_cause_column()  # ✅ NEW

    crash_lookup = {}
    try:
        report_df = load_crash_report(CRASH_REPORT_PATH)
        crash_lookup = build_crash_index_lookup(report_df)
        print(f"✓ Crash report loaded ({len(crash_lookup)} unique dates with index entries)")
    except Exception as e:
        print(f"⚠️ Could not load crash report: {e}")
        crash_lookup = {}

    start_time = datetime.now()
    processed_count = 0
    error_count = 0

    for date_string in date_list:
        try:
            print(f"\n[{processed_count + 1}/{len(date_list)}] Processing {date_string}")

            all_news, negative_news, summary_lines = crash_reason_pipeline(date_string, crash_lookup)

            print(f"\n  📝 SUMMARY FOR {date_string}:")
            print("  " + "-" * 50)
            for line in summary_lines:
                print(f"  {line}")
            print("  " + "-" * 50)

            processed_count += 1

        except Exception as e:
            print(f"\n  ❌ ERROR processing {date_string}: {str(e)}")
            error_count += 1

            try:
                error_lines = [
                    f"Pipeline error processing {date_string}",
                    f"Exception: {str(e)[:120]}",
                    "Analysis incomplete",
                    "Manual review required",
                    "Check logs and retry pipeline",
                ]
                save_crash_reason(date_string, error_lines)
                save_main_cause_label(date_string, "Insufficient evidence")  # ✅ NEW
            except Exception:
                pass

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print("Results:")
    print(f"  • Dates Processed: {processed_count}")
    print(f"  • Errors: {error_count}")
    print(f"  • Total Time: {duration}")
    print(f"  • Avg Time per Date: {duration / processed_count if processed_count > 0 else 'N/A'}")
    print("=" * 80)


if __name__ == "__main__":
    dates = [

        "2020-09-03",

    ]
    run_pipeline_for_dates(dates)
