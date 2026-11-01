"""
Docker / Render / Railway entrypoint:
ensure dirs → wait for DB (MySQL only) → init schema/seed → start uvicorn.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def wait_for_db(retries: int = 40, delay: float = 3.0) -> None:
    from webapp.database import DB_BACKEND, get_connection

    if DB_BACKEND == "sqlite":
        conn = get_connection()
        conn.close()
        print("SQLite ready.")
        return

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            conn = get_connection()
            conn.close()
            print(f"MySQL ready (attempt {attempt}).")
            return
        except Exception as exc:
            last_error = exc
            print(f"Waiting for MySQL ({attempt}/{retries}): {exc}")
            time.sleep(delay)
    raise RuntimeError(f"MySQL not reachable after {retries} tries: {last_error}")


def ensure_dirs() -> None:
    base = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(base, "uploads", "doctor_docs"), exist_ok=True)
    os.makedirs(os.path.join(base, "data"), exist_ok=True)


def main() -> None:
    ensure_dirs()
    wait_for_db()

    from webapp.init_db import init_database

    init_database()

    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    print(f"Starting BrainScan AI on {host}:{port}")
    uvicorn.run("webapp.main:app", host=host, port=port, proxy_headers=True)


if __name__ == "__main__":
    main()
