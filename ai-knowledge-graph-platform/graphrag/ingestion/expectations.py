"""Declarative data expectations on the tables a relational build reads.

An expectation is checked against the rows read from the source before
anything is written; a failed expectation blocks the build, so the previous
publication stays in place. Kinds:

* ``schema``     -- the listed columns are present in every row;
* ``not_null``   -- the column has no null or empty value;
* ``unique``     -- the column's values are unique;
* ``freshness``  -- the newest timestamp in the column is at most
  ``max_age_hours`` old;
* ``row_count``  -- the table has between ``min_rows`` and ``max_rows`` rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Expectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")
    table: str = Field(min_length=1, max_length=128)
    kind: Literal["schema", "not_null", "unique", "freshness", "row_count"]
    columns: list[str] = Field(default_factory=list)
    column: str | None = None
    max_age_hours: float | None = Field(default=None, gt=0)
    min_rows: int | None = Field(default=None, ge=0)
    max_rows: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def operands_match_kind(self) -> "Expectation":
        if self.kind == "schema" and not self.columns:
            raise ValueError(f"expectation {self.id}: schema needs columns")
        if self.kind in {"not_null", "unique", "freshness"} and not self.column:
            raise ValueError(f"expectation {self.id}: {self.kind} needs a column")
        if self.kind == "freshness" and self.max_age_hours is None:
            raise ValueError(f"expectation {self.id}: freshness needs max_age_hours")
        if self.kind == "row_count" and self.min_rows is None and self.max_rows is None:
            raise ValueError(f"expectation {self.id}: row_count needs min_rows or max_rows")
        if self.min_rows is not None and self.max_rows is not None and self.min_rows > self.max_rows:
            raise ValueError(f"expectation {self.id}: min_rows exceeds max_rows")
        return self


@dataclass(frozen=True)
class ExpectationResult:
    id: str
    table: str
    kind: str
    passed: bool
    observed: Any
    message: str
    column: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExpectationsFailed(ValueError):
    """At least one expectation failed; nothing was written."""

    def __init__(self, results: list[ExpectationResult], build_id: str = ""):
        self.results = results
        self.build_id = build_id
        failed = [r for r in results if not r.passed]
        super().__init__("data expectations failed: " + "; ".join(f"{r.id}: {r.message}" for r in failed))


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime(value.year, value.month, value.day)
    elif isinstance(value, str) and value.strip():
        try:
            moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def check_expectation(expectation: Expectation, rows: list[dict[str, Any]] | None, *,
                      now: datetime) -> ExpectationResult:
    def result(passed: bool, observed: Any, message: str) -> ExpectationResult:
        return ExpectationResult(expectation.id, expectation.table, expectation.kind, passed, observed,
                                 "ok" if passed else message, expectation.column)

    if rows is None:
        return result(False, None, f"table {expectation.table!r} was not read by this build")
    kind = expectation.kind
    if kind == "row_count":
        count = len(rows)
        low_ok = expectation.min_rows is None or count >= expectation.min_rows
        high_ok = expectation.max_rows is None or count <= expectation.max_rows
        return result(low_ok and high_ok, count,
                      f"{count} rows, expected between {expectation.min_rows} and {expectation.max_rows}")
    if kind == "schema":
        missing = sorted({column for row in rows for column in expectation.columns if column not in row})
        if not rows:
            missing = []
        return result(not missing, missing, f"missing column(s) {missing}")
    column = expectation.column or ""
    if kind == "not_null":
        nulls = sum(1 for row in rows if _empty(row.get(column)))
        return result(nulls == 0, nulls, f"{nulls} null or empty value(s) in {column}")
    if kind == "unique":
        seen: set[str] = set()
        duplicates: set[str] = set()
        for row in rows:
            value = row.get(column)
            if _empty(value):
                continue
            key = repr(value)
            (duplicates if key in seen else seen).add(key)
        return result(not duplicates, len(duplicates), f"{len(duplicates)} duplicated value(s) in {column}")
    stamps = [stamp for stamp in (_timestamp(row.get(column)) for row in rows) if stamp is not None]
    if not stamps:
        return result(False, None, f"no readable timestamp in {column}")
    newest = max(stamps)
    age_hours = (now - newest).total_seconds() / 3600
    return result(age_hours <= (expectation.max_age_hours or 0), round(age_hours, 3),
                  f"newest {column} is {age_hours:.1f} h old, limit {expectation.max_age_hours} h")


def check_expectations(expectations: list[Expectation], tables: dict[str, list[dict[str, Any]]], *,
                       now: datetime | None = None) -> list[ExpectationResult]:
    moment = now or datetime.now(timezone.utc)
    return [check_expectation(e, tables.get(e.table), now=moment) for e in expectations]


__all__ = ["Expectation", "ExpectationResult", "ExpectationsFailed", "check_expectation", "check_expectations"]
