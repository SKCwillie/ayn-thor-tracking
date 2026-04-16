import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from bs4 import BeautifulSoup


DASHBOARD_URL = "https://www.ayntec.com/pages/shipment-dashboard"
DATE_PATTERN = re.compile(r"(\d{4}/\d{1,2}/\d{1,2})")
LINE_DETAIL_PATTERN = re.compile(r"Thor ([\w\s]+) ([\w]+): (\d{4,5})xx--(\d{4,5})xx")

DEFAULT_INTERVAL_SECONDS = 3600
DEFAULT_STATE_PATH = ".monitor_state.json"
DEFAULT_RESTART_SCRIPT = "/home/ubuntu/restart_service.sh"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_date(date_str: str) -> str:
    parts = date_str.split("/")
    return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"


def fetch_dashboard(url: str, timeout: int, retries: int) -> str:
    backoff = 2
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_error = exc
            print(f"[{utc_now()}] Fetch failed (attempt {attempt}/{retries}): {exc}", flush=True)
            if attempt < retries:
                time.sleep(backoff)
                backoff *= 2

    raise RuntimeError(f"Failed to fetch dashboard after {retries} attempts: {last_error}")


def parse_shipments(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")

    shipments: List[Dict[str, str]] = []
    current_date = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        date_match = DATE_PATTERN.match(line)
        if date_match:
            current_date = normalize_date(date_match.group(1))
            continue

        if "Thor" in line and ":" in line and "xx--" in line:
            detail_match = LINE_DETAIL_PATTERN.search(line)
            if detail_match and current_date:
                color_model = detail_match.group(1).strip()
                model = detail_match.group(2).strip()
                color = color_model.replace(model, "").strip()
                begin = detail_match.group(3)
                end = detail_match.group(4)

                shipments.append(
                    {
                        "date": current_date,
                        "make": "Thor",
                        "model": model,
                        "color": color,
                        "begin": begin,
                        "end": end,
                    }
                )

    return shipments


def load_state(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[{utc_now()}] Warning: could not read state file {path}: {exc}", flush=True)
        return {}


def save_state(path: Path, state: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def row_key(row: Dict[str, str]) -> Tuple[str, str, str, str, str]:
    return row["date"], row["model"], row["color"], row["begin"], row["end"]


def latest_by_variant(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    latest: Dict[str, Dict[str, str]] = {}
    for row in rows:
        key = f"{row['color']}|{row['model']}"
        current = latest.get(key)
        if current is None or int(row["end"]) > int(current["end"]):
            latest[key] = {
                "date": row["date"],
                "end": row["end"],
                "begin": row["begin"],
            }
    return latest


def run_restart_script(script_path: Path, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[{utc_now()}] Dry run: would execute {script_path}", flush=True)
        return

    if not script_path.exists():
        raise FileNotFoundError(f"Restart script not found: {script_path}")

    result = subprocess.run(
        ["bash", str(script_path)],
        text=True,
        capture_output=True,
        check=False,
    )

    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    if result.stderr.strip():
        print(result.stderr.strip(), flush=True)

    if result.returncode != 0:
        raise RuntimeError(f"Restart script failed with exit code {result.returncode}")

    print(f"[{utc_now()}] Restart script completed successfully.", flush=True)


def check_once(args: argparse.Namespace) -> None:
    state_path = Path(args.state_path).expanduser().resolve()
    script_path = Path(args.restart_script).expanduser().resolve()

    html = fetch_dashboard(args.url, args.timeout, args.retries)
    current_rows = parse_shipments(html)
    current_keys = sorted(row_key(row) for row in current_rows)
    current_latest = latest_by_variant(current_rows)

    if not current_rows:
        print(f"[{utc_now()}] No shipment rows parsed from dashboard.", flush=True)

    state = load_state(state_path)
    previous_keys = [tuple(item) for item in state.get("shipment_keys", [])]
    previous_key_set = set(previous_keys)

    new_rows = [row for row in current_rows if row_key(row) not in previous_key_set]

    if previous_keys and new_rows:
        print(f"[{utc_now()}] Detected {len(new_rows)} new shipment row(s).", flush=True)
        run_restart_script(script_path, dry_run=args.dry_run)
    elif not previous_keys:
        print(f"[{utc_now()}] Baseline saved. No restart on first run.", flush=True)
    else:
        print(f"[{utc_now()}] No new shipments detected.", flush=True)

    new_state = {
        "url": args.url,
        "last_checked": utc_now(),
        "shipment_keys": [list(item) for item in current_keys],
        "latest_by_variant": current_latest,
        "total_rows": len(current_rows),
    }
    save_state(state_path, new_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor AYN shipment dashboard and restart services when new shipments appear."
    )
    parser.add_argument("--url", default=DASHBOARD_URL, help="Dashboard URL to check")
    parser.add_argument("--state-path", default=DEFAULT_STATE_PATH, help="Path to state JSON file")
    parser.add_argument(
        "--restart-script",
        default=DEFAULT_RESTART_SCRIPT,
        help="Script to execute when new shipments are detected",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Seconds between checks in monitor mode",
    )
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Fetch retries per check")
    parser.add_argument("--dry-run", action="store_true", help="Do not execute restart script")
    parser.add_argument("--once", action="store_true", help="Run one check then exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.interval_seconds < 1:
        raise ValueError("--interval-seconds must be >= 1")

    if args.once:
        check_once(args)
        return

    print(f"[{utc_now()}] Starting monitor loop. Interval={args.interval_seconds}s", flush=True)
    while True:
        try:
            check_once(args)
        except Exception as exc:
            print(f"[{utc_now()}] Check failed: {exc}", flush=True)

        try:
            time.sleep(args.interval_seconds)
        except KeyboardInterrupt:
            print(f"[{utc_now()}] Stopping monitor.", flush=True)
            break


if __name__ == "__main__":
    print("MONITOR STARTED", flush=True)
    main()