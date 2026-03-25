import requests
from bs4 import BeautifulSoup
import re
import sqlite3
import time

def scrape_shipments(historical=False):
    # List of URL_TIMESTAMPs to scrape
    URL_TIMESTAMPS = [
        20251103045827,
        20251124183057,
        20251126170135,
        20251208032436,
        20251210022244,
        20251227063131,
        20260101205705,
        20260103045114,
        20260103184600,
        20260105202233,
        20260112035709,
        20260112170509,
        20260115125457, #
        20260118182145,
        20260119204153,
        20260121024501,
        20260126184019,
        20260129193535,
        20260201232149,
        20260202060816,
        20260227235947,
        20260301013408, #
        20260301175137,
        20260303023618,
        20260320034019,
        20260322014022,
        20260323133734
    ]

    BASE_URL = 'https://web.archive.org/web/{}/https://www.ayntec.com/pages/shipment-dashboard'
    LIVE_URL = 'https://www.ayntec.com/pages/shipment-dashboard'

    # Regex patterns
    DATE_PATTERN = re.compile(r"(\d{4}/\d{1,2}/\d{1,2})")
    LINE_DETAIL_PATTERN = re.compile(r"Thor ([\w\s]+) ([\w]+): (\d{4,5})xx--(\d{4,5})xx")

    # Set up SQLite database
    conn = sqlite3.connect('shipping_info.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS shipments (
            date DATE,
            make TEXT,
            model TEXT,
            color TEXT,
            begin TEXT,
            end TEXT,
            units_shipped INTEGER,
            UNIQUE(date, make, model, color, begin, end)
        )
    ''')

    total_rows = 0
    urls = []
    if historical:
        urls = [BASE_URL.format(ts) for ts in URL_TIMESTAMPS]
    else:
        urls = [LIVE_URL]

    for url in urls:
        max_retries = 5
        retry_wait = 5
        response = None
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                break  # Success, exit retry loop
            except Exception as e:
                err_str = str(e)
                if ("Failed to establish a new connection" in err_str and
                    "Errno 61" in err_str and
                    attempt < max_retries - 1):
                    print(f"Connection refused for {url}, retrying in {retry_wait} seconds (attempt {attempt+1}/{max_retries})...")
                    time.sleep(retry_wait)
                    retry_wait *= 2
                    continue
                else:
                    print(f"Failed to fetch {url}: {e}")
                    break
        if response is None:
            # All retries failed, skip this URL
            continue
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n')
        current_date = None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            date_match = DATE_PATTERN.match(line)
            if date_match:
                current_date = date_match.group(1)
                continue
            if 'Thor' in line and ':' in line and 'xx--' in line:
                detail_match = LINE_DETAIL_PATTERN.search(line)
                if detail_match and current_date:
                    color_model = detail_match.group(1).strip()
                    model = detail_match.group(2).strip()
                    color = color_model.replace(model, '').strip()
                    begin = detail_match.group(3)
                    end = detail_match.group(4)
                    try:
                        # Convert date from yyyy/mm/dd to yyyy-mm-dd for SQL DATE, zero-padded
                        parts = current_date.split('/')
                        date_sql = f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
                        units_shipped = int(end) - int(begin)
                        c.execute(
                            'INSERT OR IGNORE INTO shipments VALUES (?, ?, ?, ?, ?, ?, ?)',
                            (date_sql, 'Thor', model, color, begin, end, units_shipped)
                        )
                        total_rows += c.rowcount
                    except Exception as db_e:
                        print(f"DB error: {db_e}")
        time.sleep(2)  # Add a 2-second delay between requests
    conn.commit()
    conn.close()
    print(f"Scraping complete. {total_rows} new rows added to shipping_info.db.")

if __name__ == "__main__":
    scrape_shipments(historical=True)
