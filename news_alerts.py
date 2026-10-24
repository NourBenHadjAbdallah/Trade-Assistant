"""
news_alerts.py - follow coins, get alerts when high-impact news from TODAY hits them.

Sources (no API key needed): Google News search per followed coin + crypto RSS feeds
(Cointelegraph, CoinDesk, Decrypt, Bitcoin Magazine, CryptoSlate).
Optional: set CRYPTOCOMPARE_KEY to add CryptoCompare as one more source.
"Today" is the calendar day in NEWS_TZ (default Africa/Tunis).

Each article gets an impact score (0-10) and a bullish/bearish direction from weighted keywords;
only articles at or above the user's threshold are alerted, once per user.
"""
import hashlib
import html
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import requests

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo(os.environ.get("NEWS_TZ", "Africa/Tunis"))
except Exception:                       # tzdata missing (common on Windows)
    TZ = timezone(timedelta(hours=1))

STATE_PATH = os.environ.get("STATE_PATH", "state.json")
DEFAULT_THRESHOLD = 3
MAX_ALERTS_PER_RUN = 8                  # per user, newest first
HEADERS = {"User-Agent": "Mozilla/5.0 (TradeAssistant news reader)"}
LAST = (0, 0)                           # (sources that answered, sources tried) in the last fetch
_lock = threading.Lock()

FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://decrypt.co/feed",
    "https://bitcoinmagazine.com/.rss/full/",
    "https://cryptoslate.com/feed/",
]
GOOGLE = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
CRYPTOCOMPARE = "https://min-api.cryptocompare.com/data/v2/news/"

ALIASES = {
    "BTC": ["bitcoin"], "ETH": ["ethereum", "ether"], "SOL": ["solana"], "BNB": ["binance coin", "bnb chain"],
    "XRP": ["ripple"], "ADA": ["cardano"], "DOGE": ["dogecoin"], "AVAX": ["avalanche"],
    "LINK": ["chainlink"], "DOT": ["polkadot"], "MATIC": ["polygon"], "LTC": ["litecoin"],
    "TRX": ["tron"], "TON": ["toncoin"], "SUI": [], "ARB": ["arbitrum"], "OP": ["optimism"],
    "SHIB": ["shiba inu"], "PEPE": ["pepe coin"], "XLM": ["stellar"], "ATOM": ["cosmos"],
    "NEAR": ["near protocol"], "UNI": ["uniswap"], "BCH": ["bitcoin cash"], "ETC": ["ethereum classic"],
}

# (keyword, weight). Positive = bullish, negative = bearish. |weight| = impact.
KEYWORDS = [
    ("etf approv", 5), ("approves", 2), ("listing on", 3), ("lists ", 2), ("partnership", 2),
    ("upgrade", 2), ("mainnet", 2), ("institutional", 2), ("adoption", 2), ("buyback", 2),
    ("all-time high", 3), ("record high", 3), ("surges", 2), ("rallies", 2), ("accumulat", 2),
    ("hack", -5), ("exploit", -5), ("stolen", -4), ("drained", -5), ("rug pull", -5),
    ("delist", -4), ("lawsuit", -3), ("sues", -3), ("sec charges", -5), ("ban", -3), ("banned", -3),
    ("investigation", -3), ("bankruptcy", -5), ("insolven", -5), ("outage", -2), ("halts", -3),
    ("suspends", -3), ("crash", -3), ("plunges", -3), ("liquidat", -2), ("dumps", -2), ("fraud", -4),
    ("sanction", -3), ("rejects", -3), ("delays", -2),
]
MARKET_WIDE = ["fed ", "fomc", "interest rate", "cpi", "inflation", "sec ", "tariff", "regulation"]


# ---------- time ----------
def is_today(ts):
    return datetime.fromtimestamp(ts, TZ).date() == datetime.now(TZ).date()


def local_hm(ts):
    return datetime.fromtimestamp(ts, TZ).strftime("%H:%M")


# ---------- state ----------
def _load():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {"users": {}, "seen": []}


def _save(st):
    st["seen"] = st["seen"][-3000:]
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, STATE_PATH)


def _user(st, chat_id):
    return st["users"].setdefault(str(chat_id), {"coins": [], "threshold": DEFAULT_THRESHOLD})


def _clean(c):
    return re.sub(r"(USDT|USDC|USD)$", "", str(c).upper().strip())


def follow(chat_id, coins):
    with _lock:
        st = _load()
        u = _user(st, chat_id)
        for c in map(_clean, coins):
            if c and c not in u["coins"]:
                u["coins"].append(c)
        _save(st)
        return list(u["coins"])


def unfollow(chat_id, coins):
    with _lock:
        st = _load()
        u = _user(st, chat_id)
        drop = {_clean(c) for c in coins}
        u["coins"] = [c for c in u["coins"] if c not in drop]
        _save(st)
        return list(u["coins"])


def toggle(chat_id, coin):
    """Follow if not followed, else unfollow. Returns (now_following, coins)."""
    c = _clean(coin)
    if c in get_user(chat_id)["coins"]:
        return False, unfollow(chat_id, [c])
    return True, follow(chat_id, [c])


def get_user(chat_id):
    with _lock:
        u = _user(_load(), chat_id)
        return {"coins": list(u["coins"]), "threshold": u.get("threshold", DEFAULT_THRESHOLD)}


def set_threshold(chat_id, n):
    with _lock:
        st = _load()
        _user(st, chat_id)["threshold"] = max(1, min(10, int(n)))
        _save(st)


# ---------- fetching ----------
def _strip(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def _parse_rss(content, default_source):
    items = []
    root = ET.fromstring(content)
    for it in root.iter("item"):
        title = _strip(it.findtext("title"))
        link = (it.findtext("link") or "").strip()
        pub = it.findtext("pubDate")
        if not (title and link and pub):
            continue
        try:
            dt = parsedate_to_datetime(pub)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        src_el = it.find("source")
        source = (src_el.text if src_el is not None and src_el.text else default_source)
        if source and title.endswith(" - " + source):          # Google News appends the outlet
            title = title[: -len(source) - 3]
        items.append({"id": hashlib.md5(link.encode()).hexdigest()[:16], "title": title,
                      "body": _strip(it.findtext("description"))[:400], "url": link,
                      "published_on": dt.timestamp(), "source": source})
    return items


def _get_feed(url, default_source):
    r = requests.get(url, headers=HEADERS, timeout=12)
    r.raise_for_status()
    return _parse_rss(r.content, default_source)


def _get_cryptocompare():
    key = os.environ.get("CRYPTOCOMPARE_KEY")
    r = requests.get(CRYPTOCOMPARE, params={"lang": "EN"}, headers={"authorization": f"Apikey {key}"}, timeout=12)
    r.raise_for_status()
    return [{"id": "cc" + str(d.get("id")), "title": d.get("title", ""), "body": d.get("body", "")[:400],
             "url": d.get("url", ""), "published_on": d.get("published_on", 0),
             "source": (d.get("source_info") or {}).get("name") or d.get("source", "")}
            for d in r.json().get("Data", [])]


def _google_query(coin):
    names = ALIASES.get(coin, [])
    terms = " OR ".join(['"' + n + '"' for n in names] + [coin])
    return GOOGLE.format(q=quote_plus(f"({terms}) crypto when:1d"))


def fetch_news(coins):
    """Today's articles from all sources, newest first, deduplicated. Raises only if every source fails."""
    global LAST
    jobs = [(lambda u=u: _get_feed(u, u.split("/")[2].replace("www.", ""))) for u in FEEDS]
    jobs += [(lambda u=_google_query(c): _get_feed(u, "Google News")) for c in coins]
    if os.environ.get("CRYPTOCOMPARE_KEY"):
        jobs.append(_get_cryptocompare)

    def safe(fn):
        try:
            return fn()
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(safe, jobs))
    ok = sum(1 for r in results if r is not None)
    LAST = (ok, len(jobs))
    if ok == 0:
        raise RuntimeError("no news source reachable (check your internet connection)")
    seen, out = set(), []
    for items in results:
        for it in items or []:
            key = re.sub(r"\W+", "", it["title"].lower())[:80]   # same story from several outlets
            if it["id"] in seen or key in seen or not is_today(it["published_on"]):
                continue
            seen.update((it["id"], key))
            out.append(it)
    return sorted(out, key=lambda i: -i["published_on"])


# ---------- analysis ----------
def matches(item, coins):
    """Coins from `coins` that this article is really about (title match, or name in headline/body)."""
    title = item.get("title", "")
    text = (title + " " + item.get("body", "")).lower()
    found = []
    for c in coins:
        names = ALIASES.get(c, []) + ([c.lower()] if len(c) >= 4 else [])
        in_title = re.search(rf"\b{re.escape(c)}\b", title) is not None      # symbol, case-sensitive
        by_name = any(re.search(rf"\b{re.escape(a)}\b", text) for a in names)
        if in_title or by_name:
            found.append(c)
    return found


def score(item):
    """Return (impact 0-10, direction -1/0/+1, matched keywords)."""
    title = item.get("title", "").lower()
    text = title + " " + item.get("body", "")[:300].lower()
    total, impact, hits = 0, 0, []
    for kw, w in KEYWORDS:
        if kw in text:
            mult = 1.5 if kw in title else 1.0                                 # headline matters more
            total += w * mult
            impact += abs(w) * mult
            hits.append(kw.strip())
    if any(m in text for m in MARKET_WIDE):
        impact += 1
    direction = 1 if total > 1 else -1 if total < -1 else 0
    return min(10, round(impact)), direction, hits


def format_alert(item, coins, impact, direction, hits):
    icon = {1: "🟢 Likely bullish", -1: "🔴 Likely bearish", 0: "🟡 Unclear direction"}[direction]
    return (f"🚨 {', '.join(coins)} news  |  impact {impact}/10\n"
            f"{icon}\n\n{item.get('title', '')}\n"
            f"Source: {item.get('source') or '?'} · today {local_hm(item.get('published_on', time.time()))}\n"
            f"Triggers: {', '.join(hits[:4]) or 'general'}\n{item.get('url', '')}\n\n"
            f"Check your levels and stops before reacting. Not financial advice.")


def latest_for(coins, limit=6):
    """Today's articles about these coins, newest first (no impact threshold)."""
    out = []
    for it in fetch_news(coins):
        m = matches(it, coins)
        if m:
            imp, d, h = score(it)
            out.append(format_alert(it, m, imp, d, h))
        if len(out) >= limit:
            break
    return out


def run_check(send):
    """Push new high-impact TODAY articles to every user following the coin. send(chat_id, text)."""
    with _lock:
        st = _load()
    all_coins = sorted({c for u in st["users"].values() for c in u["coins"]})
    if not all_coins:
        return 0
    items = fetch_news(all_coins)                       # already today-only, newest first
    sent, new_seen, per_user = 0, [], {}
    for it in items:
        impact, direction, hits = score(it)
        for chat_id, u in st["users"].items():
            key = f"{chat_id}:{it['id']}"
            if key in st["seen"] or key in new_seen:
                continue
            m = matches(it, u["coins"])
            if not m:
                continue
            new_seen.append(key)                         # never re-evaluate this article for this user
            if impact >= u.get("threshold", DEFAULT_THRESHOLD) and per_user.get(chat_id, 0) < MAX_ALERTS_PER_RUN:
                try:
                    send(chat_id, format_alert(it, m, impact, direction, hits))
                    per_user[chat_id] = per_user.get(chat_id, 0) + 1
                    sent += 1
                except Exception as e:
                    print("send failed:", e)
    with _lock:
        st2 = _load()
        st2["seen"] += new_seen
        _save(st2)
    return sent