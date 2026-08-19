"""
scrape_reviews.py
=================
Scrapes Google Play Store reviews and app metadata for every Salah app
listed in the CSV file.

Output structure
----------------
reviews/
  <AppName>/
    metadata.json      <- app-level info (description, installs, rating, ...)
    reviews.csv        <- every review (text, rating, date, author, ...)
scrape_summary.csv     <- one row per app (success/fail, total reviews scraped)

Usage
-----
  python scrape_reviews.py
  python scrape_reviews.py --csv "Salah App Analysis - Sheet1.csv" --out reviews
  python scrape_reviews.py --delay 2.0 --retries 5
"""

import argparse
import csv
import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import pandas as pd
from google_play_scraper import Sort, app as gps_app, reviews_all
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
DEFAULT_CSV = "Salah App Analysis - Sheet1.csv"
DEFAULT_OUT = "reviews"
DEFAULT_DELAY = 1.5   # seconds between apps
DEFAULT_RETRIES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def safe_dirname(name):
    """Convert an app name to a filesystem-safe directory name."""
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"[^\w\s\-]", "", name, flags=re.UNICODE)
    name = re.sub(r"[\s]+", "_", name.strip())
    return name[:80]


def extract_package_name(row):
    """
    Return the package name from the 'package name' column.
    Fall back to parsing it out of the Play Store URL.
    """
    pkg = str(row.get("package name") or "").strip()
    if pkg and pkg.lower() != "nan":
        return pkg

    url = str(row.get("Link") or "").strip()
    if url:
        try:
            qs = parse_qs(urlparse(url).query)
            ids = qs.get("id", [])
            if ids:
                return ids[0].strip()
        except Exception:
            pass

    return None


def fetch_app_metadata(package, retries, log):
    """Fetch app-level metadata with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            result = gps_app(
                package,
                lang="en",
                country="us",
            )
            return result
        except Exception as exc:
            log.warning("  Metadata attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)  # exponential back-off
    return None


def fetch_all_reviews(package, retries, log):
    """Fetch ALL reviews (newest-first) with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            result = reviews_all(
                package,
                lang="en",
                country="us",
                sort=Sort.NEWEST,
                sleep_milliseconds=200,   # built-in polite delay per batch
            )
            return result
        except Exception as exc:
            log.warning("  Reviews attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)
    return []


def save_metadata(meta, out_dir):
    """Persist app metadata as JSON, converting non-serialisable types."""
    def default_serialiser(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    path = os.path.join(out_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=default_serialiser)


def save_reviews(reviews, out_dir):
    """Persist reviews as CSV."""
    if not reviews:
        return

    rows = []
    for r in reviews:
        rows.append({
            "reviewId":              r.get("reviewId", ""),
            "userName":              r.get("userName", ""),
            "userImage":             r.get("userImage", ""),
            "content":               r.get("content", ""),
            "score":                 r.get("score", ""),
            "thumbsUpCount":         r.get("thumbsUpCount", ""),
            "reviewCreatedVersion":  r.get("reviewCreatedVersion", ""),
            "at":                    r.get("at", ""),
            "replyContent":          r.get("replyContent", ""),
            "repliedAt":             r.get("repliedAt", ""),
            "appVersion":            r.get("appVersion", ""),
        })

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "reviews.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig for Excel compat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Scrape Play Store reviews for Salah apps")
    p.add_argument("--csv",     default=DEFAULT_CSV,  help="Input CSV file")
    p.add_argument("--out",     default=DEFAULT_OUT,  help="Output root directory")
    p.add_argument("--delay",   type=float, default=DEFAULT_DELAY,
                   help="Seconds to wait between apps (default 1.5)")
    p.add_argument("--retries", type=int,   default=DEFAULT_RETRIES,
                   help="Max retries per request (default 3)")
    return p.parse_args()


def load_apps(csv_path, log):
    """
    Read the CSV. The actual header row is row 4 (0-indexed: row 3).
    Rows above it are empty / section headers.
    """
    df = pd.read_csv(csv_path, header=3, dtype=str)
    df = df.dropna(subset=["App Name"])
    df = df[df["App Name"].str.strip() != ""]
    log.info("Loaded %d apps from %s", len(df), csv_path)
    return df.to_dict(orient="records")


def main():
    args = parse_args()
    log = setup_logging()

    os.makedirs(args.out, exist_ok=True)

    # ---- Load app list --------------------------------------------------
    apps = load_apps(args.csv, log)

    summary_rows = []
    scrape_time = datetime.now(tz=timezone.utc).isoformat()

    # ---- Scrape each app ------------------------------------------------
    for row in tqdm(apps, desc="Apps", unit="app"):
        app_name = (row.get("App Name") or "").strip()
        package  = extract_package_name(row)

        if not package:
            log.warning("Skipping '%s' -- no package name found.", app_name)
            summary_rows.append({
                "app_name":        app_name,
                "package":         "",
                "status":          "SKIPPED (no package)",
                "reviews_scraped": 0,
                "scrape_time":     scrape_time,
            })
            continue

        log.info("-- %s  [%s]", app_name, package)
        out_dir = os.path.join(args.out, safe_dirname(app_name))
        os.makedirs(out_dir, exist_ok=True)

        # 1. App metadata -------------------------------------------------
        log.info("  Fetching metadata ...")
        meta = fetch_app_metadata(package, args.retries, log)
        if meta:
            save_metadata(meta, out_dir)
            log.info("  OK Metadata saved")
        else:
            log.error("  FAIL Could not fetch metadata for %s", package)

        # 2. Reviews -------------------------------------------------------
        log.info("  Fetching all reviews (newest first) ...")
        reviews = fetch_all_reviews(package, args.retries, log)
        n = len(reviews)
        log.info("  OK %d reviews fetched", n)

        save_reviews(reviews, out_dir)

        summary_rows.append({
            "app_name":         app_name,
            "package":          package,
            "csv_rating":       row.get("Rating", ""),
            "csv_rating_count": row.get("Rating Count", ""),
            "csv_downloads":    row.get("Downloads", ""),
            "store_rating":     meta.get("score", "")        if meta else "",
            "store_reviews":    meta.get("reviews", "")      if meta else "",
            "store_installs":   meta.get("realInstalls", "") if meta else "",
            "reviews_scraped":  n,
            "status":           "OK" if meta else "METADATA_FAILED",
            "scrape_time":      scrape_time,
            "out_dir":          out_dir,
        })

        time.sleep(args.delay)

    # ---- Summary CSV ----------------------------------------------------
    summary_path = os.path.join(args.out, "scrape_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    log.info("Done! Summary -> %s", summary_path)
    log.info("Total apps processed: %d", len(summary_rows))
    total_reviews = sum(r["reviews_scraped"] for r in summary_rows)
    log.info("Total reviews scraped: %d", total_reviews)


if __name__ == "__main__":
    main()
