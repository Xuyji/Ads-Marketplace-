"""Loads simulator output into SQLite and evaluates the governed KPI views in sql/metrics.sql."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .sim import CATEGORIES, SimResult, World, calendar

SQL = Path(__file__).resolve().parent.parent / "sql" / "metrics.sql"


def build_warehouse(res: SimResult, world: World, path: Path) -> sqlite3.Connection:
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    c = res.country.assign(date=res.country.day.map(lambda d: calendar(d).strftime("%Y-%m-%d")))
    a = res.adv.assign(category=CATEGORIES[world.adv_cat[res.adv.adv]])
    c.to_sql("fact_country_day", con, index=False)
    a.to_sql("fact_adv_day", con, index=False)
    con.executescript(SQL.read_text())
    return con


def query(con: sqlite3.Connection, view: str) -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM {view}", con)
