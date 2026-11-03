"""Spreadsheet (CSV/XLSX) exchange of ontology proposals and curator decisions.

Export writes proposals in a stable column order, followed by blank decision
columns (``action``, ``reason``, ``edited_value``, ``merge_target``) that a
curator fills in and uploads again.

Safety rules
------------
* **No formulas, ever.**  Any exported string whose first non-blank character is
  ``= + - @`` (or tab/CR) is prefixed with ``'`` (OWASP CSV-injection guidance);
  XLSX string cells are additionally forced to the string data type so
  openpyxl never stores them as formulas.  Numbers stay numeric.  On import the
  escape is reversed, so export -> import is lossless.
* **Uploads are untrusted.**  Formula cells in an uploaded XLSX are rejected,
  row count / cell length / byte size are bounded, and every field is validated.

Import contract (documented choices)
------------------------------------
* Required columns: ``proposal_id`` and ``action``.  Header matching ignores
  case and treats spaces as underscores.
* **Unknown columns are ignored** (so an exported sheet, which carries context
  columns, can be uploaded as-is).  Identity columns such as ``reviewed_by`` are
  never read: the caller supplies the reviewer from the verified token.
* A row with a blank ``action`` is *skipped* (no decision made), not an error.
* ``edit`` needs ``edited_value``; ``merge`` needs ``merge_target``; ``reject``
  and ``quarantine`` need ``reason``.  ``edited_value`` / ``merge_target`` on any
  other action is an error, to avoid ambiguous intent.
* A ``proposal_id`` appearing on more than one row rejects *all* its rows.
* Row-level problems are collected (``ParsedDecisions.errors``); structural
  problems (bad file, missing columns, too many rows) raise ``ExchangeError``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import re
import zipfile
from typing import Any, Iterable, Iterator

import openpyxl
from openpyxl.utils.exceptions import InvalidFileException

ALLOWED_ACTIONS = ("approve", "reject", "edit", "merge", "defer", "quarantine")
MAX_DECISION_ROWS = 1000
MAX_SCANNED_ROWS = 5000  # physical rows, so blank-row padding cannot stall the parser
MAX_FIELD_CHARS = 2000
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
_PROPOSAL_ID = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")
_FORMULA_TRIGGERS = frozenset("=+-@\t\r")
_REASON_REQUIRED = frozenset({"reject", "quarantine"})
_LIST_SEPARATOR = "; "
_XLSX_SHEET = "proposals"
_ESCAPE_PREFIX = "'"

# (column header, key in the proposal dict)
_PROPOSAL_FIELDS = (
    ("proposal_id", "id"), ("kind", "kind"), ("proposed_value", "proposed_value"),
    ("source_type", "source_type"), ("target_type", "target_type"), ("entity_name", "entity_name"),
    ("proposal_reason", "reason"), ("status", "status"), ("confidence", "confidence"),
    ("conflicts", "conflicts"), ("affected_use_cases", "affected_use_cases"),
    ("business_impact", "business_impact"), ("urgency", "urgency"), ("effort", "effort"),
    ("risk", "risk"), ("dependencies", "dependencies"), ("requesting_team", "requesting_team"),
    ("seen_count", "seen_count"), ("created_at", "created_at"), ("last_seen_at", "last_seen_at"),
)
DECISION_COLUMNS = ("action", "reason", "edited_value", "merge_target")
EXPORT_COLUMNS = tuple(name for name, _ in _PROPOSAL_FIELDS) + DECISION_COLUMNS


class ExchangeError(ValueError):
    """A structural problem with an uploaded sheet (not a single bad row)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Decision:
    row: int
    proposal_id: str
    action: str
    reason: str = ""
    edited_value: str | None = None
    merge_target: str | None = None


@dataclass(frozen=True)
class RowError:
    row: int
    message: str


@dataclass(frozen=True)
class ParsedDecisions:
    decisions: tuple[Decision, ...]
    errors: tuple[RowError, ...]
    skipped: int


# -- escaping -----------------------------------------------------------------

def _needs_escape(text: str) -> bool:
    stripped = text.lstrip(" ")  # only spaces: tab/CR are triggers themselves
    if stripped and stripped[0] in _FORMULA_TRIGGERS:
        return True
    # A literal leading quote in front of a trigger would be mistaken for our escape.
    return text.startswith(_ESCAPE_PREFIX) and _needs_escape(text[1:])


def escape_cell(value: Any) -> Any:
    """Neutralise formula injection in a string cell; other types pass through."""
    if isinstance(value, str) and _needs_escape(value):
        return _ESCAPE_PREFIX + value
    return value


def unescape_cell(value: str) -> str:
    if value.startswith(_ESCAPE_PREFIX) and _needs_escape(value[1:]):
        return value[1:]
    return value


def _plain(value: Any) -> Any:
    """Render a proposal field as a scalar cell value (numbers stay numbers)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return _LIST_SEPARATOR.join(str(item) for item in value)
    return str(value)


def _export_rows(proposals: Iterable[dict]) -> Iterator[list[Any]]:
    for proposal in proposals:
        cells = [escape_cell(_plain(proposal.get(key))) for _, key in _PROPOSAL_FIELDS]
        yield cells + [""] * len(DECISION_COLUMNS)


# -- export -------------------------------------------------------------------

def export_csv(proposals: Iterable[dict]) -> bytes:
    """UTF-8 (with BOM, so Excel detects the encoding) CSV with a header row."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")
    writer.writerow(EXPORT_COLUMNS)
    writer.writerows(_export_rows(proposals))
    return buf.getvalue().encode("utf-8-sig")


def export_xlsx(proposals: Iterable[dict]) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = _XLSX_SHEET
    ws.append(list(EXPORT_COLUMNS))
    for cells in _export_rows(proposals):
        ws.append(cells)
        for cell in ws[ws.max_row]:
            if isinstance(cell.value, str):
                cell.data_type = "s"  # never let openpyxl infer a formula
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


# -- import -------------------------------------------------------------------

def _normalise_header(raw: Any) -> str:
    return re.sub(r"\s+", "_", str(raw or "").strip().lower())


def _read_csv(data: bytes) -> Iterator[list[tuple[str, bool]]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExchangeError("invalid_file", "CSV must be UTF-8 encoded") from exc
    try:
        for index, row in enumerate(csv.reader(io.StringIO(text))):
            if index >= MAX_SCANNED_ROWS:
                raise ExchangeError("too_many_rows", f"sheet exceeds {MAX_SCANNED_ROWS} rows")
            yield [(cell, False) for cell in row]
    except csv.Error as exc:
        raise ExchangeError("invalid_file", f"malformed CSV: {exc}") from exc


def _read_xlsx(data: bytes) -> Iterator[list[tuple[str, bool]]]:
    try:
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
        ws = wb.worksheets[0]
    except (zipfile.BadZipFile, InvalidFileException, KeyError, IndexError, OSError, ValueError) as exc:
        raise ExchangeError("invalid_file", "not a readable XLSX workbook") from exc
    try:
        for index, row in enumerate(ws.iter_rows()):
            if index >= MAX_SCANNED_ROWS:
                raise ExchangeError("too_many_rows", f"sheet exceeds {MAX_SCANNED_ROWS} rows")
            yield [("" if c.value is None else str(c.value), c.data_type == "f") for c in row]
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise ExchangeError("invalid_file", "not a readable XLSX workbook") from exc
    finally:
        wb.close()


_READERS = {"csv": _read_csv, "xlsx": _read_xlsx}


def _row_to_decision(row_no: int, values: dict[str, str]) -> Decision | RowError:
    over = [name for name, value in values.items() if len(value) > MAX_FIELD_CHARS]
    if over:
        return RowError(row_no, f"{over[0]} exceeds {MAX_FIELD_CHARS} characters")
    proposal_id, action = values["proposal_id"], values["action"].lower()
    if not _PROPOSAL_ID.match(proposal_id):
        return RowError(row_no, "proposal_id is missing or malformed")
    if action not in ALLOWED_ACTIONS:
        return RowError(row_no, f"action must be one of {', '.join(ALLOWED_ACTIONS)}")
    reason, edited, target = values["reason"], values["edited_value"], values["merge_target"]
    if action == "edit" and not edited:
        return RowError(row_no, "action 'edit' requires edited_value")
    if action == "merge" and not target:
        return RowError(row_no, "action 'merge' requires merge_target")
    if action in _REASON_REQUIRED and not reason:
        return RowError(row_no, f"action '{action}' requires a reason")
    if edited and action != "edit":
        return RowError(row_no, "edited_value is only valid with action 'edit'")
    if target and action != "merge":
        return RowError(row_no, "merge_target is only valid with action 'merge'")
    return Decision(row_no, proposal_id, action, reason, edited or None, target or None)


def _flag_duplicates(decisions: list[Decision]) -> tuple[list[Decision], list[RowError]]:
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.proposal_id] = counts.get(decision.proposal_id, 0) + 1
    kept = [d for d in decisions if counts[d.proposal_id] == 1]
    dupes = [RowError(d.row, f"duplicate proposal_id '{d.proposal_id}'; every duplicate row is rejected")
             for d in decisions if counts[d.proposal_id] > 1]
    return kept, dupes


def _read_header(rows: Iterator[list[tuple[str, bool]]]) -> list[str]:
    header_row = next(rows, None)
    if header_row is None:
        raise ExchangeError("invalid_file", "sheet is empty")
    headers = [_normalise_header(cell) for cell, _ in header_row]
    missing = [name for name in ("proposal_id", "action") if name not in headers]
    if missing:
        raise ExchangeError("missing_columns", f"missing required column(s): {', '.join(missing)}")
    named = [h for h in headers if h]
    if len(set(named)) != len(named):
        raise ExchangeError("invalid_file", "duplicate column headers")
    return headers


def parse_decisions(data: bytes, fmt: str) -> ParsedDecisions:
    """Validate a decisions sheet.  ``fmt`` is ``"csv"`` or ``"xlsx"``."""
    reader = _READERS.get(fmt)
    if reader is None:
        raise ExchangeError("unsupported_format", f"unsupported sheet format '{fmt}'")
    rows = reader(data)
    headers = _read_header(rows)

    decisions: list[Decision] = []
    errors: list[RowError] = []
    skipped = data_rows = 0
    for row_no, row in enumerate(rows, start=2):
        if not any(cell.strip() for cell, _ in row):
            continue
        data_rows += 1
        if data_rows > MAX_DECISION_ROWS:
            raise ExchangeError("too_many_rows", f"more than {MAX_DECISION_ROWS} data rows")
        if any(is_formula for _, is_formula in row):
            errors.append(RowError(row_no, "formula cells are not accepted"))
            continue
        cells = {h: unescape_cell(cell.strip()) for h, (cell, _) in zip(headers, row) if h}
        values = {k: cells.get(k, "") for k in ("proposal_id", *DECISION_COLUMNS)}
        if not values["action"]:
            skipped += 1
            continue
        result = _row_to_decision(row_no, values)
        (errors if isinstance(result, RowError) else decisions).append(result)

    kept, dupes = _flag_duplicates(decisions)
    return ParsedDecisions(tuple(kept), tuple(sorted(errors + dupes, key=lambda e: e.row)), skipped)
