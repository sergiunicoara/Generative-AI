"""CSV/XLSX export and decision-sheet parsing for curator operations (offline)."""
from __future__ import annotations

import csv
import io

import openpyxl
import pytest

from graphrag.curation.exchange import (
    ALLOWED_ACTIONS,
    EXPORT_COLUMNS,
    MAX_DECISION_ROWS,
    ExchangeError,
    export_csv,
    export_xlsx,
    parse_decisions,
)

PROPOSAL = {
    "id": "p-1", "kind": "entity_type", "proposed_value": "VENDOR", "source_type": "",
    "target_type": "", "entity_name": "Beta", "reason": "unknown_entity_type", "status": "pending",
    "confidence": 0.8, "conflicts": ["relation"], "affected_use_cases": ["uc1", "uc2"],
    "business_impact": "high", "urgency": "medium", "effort": "low", "risk": "low",
    "dependencies": [], "requesting_team": "ops", "seen_count": 3,
    "created_at": "2026-01-01", "last_seen_at": "2026-01-02",
}


def _decision_csv(rows: list[dict], header=("proposal_id", "action", "reason", "edited_value", "merge_target")) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    for row in rows:
        writer.writerow([row.get(col, "") for col in header])
    return buf.getvalue().encode("utf-8")


def _decision_xlsx(rows: list[list]) -> bytes:
    wb = openpyxl.Workbook()
    for row in rows:
        wb.active.append(row)
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


class TestExport:
    def test_csv_has_stable_header_order(self):
        text = export_csv([PROPOSAL]).decode("utf-8-sig")
        assert next(csv.reader(io.StringIO(text))) == list(EXPORT_COLUMNS)

    def test_csv_row_uses_proposal_id_and_joins_lists(self):
        text = export_csv([PROPOSAL]).decode("utf-8-sig")
        row = dict(zip(EXPORT_COLUMNS, list(csv.reader(io.StringIO(text)))[1]))
        assert (row["proposal_id"], row["affected_use_cases"]) == ("p-1", "uc1; uc2")

    def test_csv_escapes_formula_leading_cells(self):
        evil = {**PROPOSAL, "proposed_value": "=HYPERLINK(\"http://x\")", "entity_name": "@SUM(A1)"}
        text = export_csv([evil]).decode("utf-8-sig")
        row = dict(zip(EXPORT_COLUMNS, list(csv.reader(io.StringIO(text)))[1]))
        assert (row["proposed_value"][0], row["entity_name"][0]) == ("'", "'")

    @pytest.mark.parametrize("lead", ["=", "+", "-", "@", "\t", "\r"])
    def test_every_formula_trigger_is_escaped(self, lead):
        text = export_csv([{**PROPOSAL, "entity_name": f"{lead}1+1"}]).decode("utf-8-sig")
        row = dict(zip(EXPORT_COLUMNS, list(csv.reader(io.StringIO(text)))[1]))
        assert row["entity_name"].startswith("'")

    def test_xlsx_never_contains_formula_cells(self):
        evil = {**PROPOSAL, "proposed_value": "=1+1", "entity_name": "-2+3"}
        ws = openpyxl.load_workbook(io.BytesIO(export_xlsx([evil]))).active
        assert [c.data_type for row in ws.iter_rows() for c in row if c.data_type == "f"] == []

    def test_xlsx_formula_text_survives_as_inert_string(self):
        ws = openpyxl.load_workbook(io.BytesIO(export_xlsx([{**PROPOSAL, "proposed_value": "=1+1"}]))).active
        assert ws.cell(row=2, column=EXPORT_COLUMNS.index("proposed_value") + 1).value == "'=1+1"

    def test_xlsx_keeps_numbers_numeric(self):
        ws = openpyxl.load_workbook(io.BytesIO(export_xlsx([PROPOSAL]))).active
        assert ws.cell(row=2, column=EXPORT_COLUMNS.index("confidence") + 1).value == 0.8

    def test_negative_numbers_are_not_mangled(self):
        text = export_csv([{**PROPOSAL, "confidence": -0.5}]).decode("utf-8-sig")
        row = dict(zip(EXPORT_COLUMNS, list(csv.reader(io.StringIO(text)))[1]))
        assert row["confidence"] == "-0.5"

    def test_exported_sheet_has_decision_columns_blank(self):
        text = export_csv([PROPOSAL]).decode("utf-8-sig")
        row = dict(zip(EXPORT_COLUMNS, list(csv.reader(io.StringIO(text)))[1]))
        assert (row["action"], row["reason"], row["edited_value"], row["merge_target"]) == ("", "", "", "")


class TestRoundTrip:
    def test_exported_csv_with_filled_action_parses_back(self):
        text = export_csv([PROPOSAL]).decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(text)))
        rows[1][EXPORT_COLUMNS.index("action")] = "approve"
        buf = io.StringIO()
        csv.writer(buf).writerows(rows)
        result = parse_decisions(buf.getvalue().encode(), "csv")
        assert [(d.proposal_id, d.action) for d in result.decisions] == [("p-1", "approve")]

    def test_exported_xlsx_with_filled_action_parses_back(self):
        wb = openpyxl.load_workbook(io.BytesIO(export_xlsx([PROPOSAL])))
        wb.active.cell(row=2, column=EXPORT_COLUMNS.index("action") + 1, value="reject")
        wb.active.cell(row=2, column=EXPORT_COLUMNS.index("reason") + 1, value="not needed")
        out = io.BytesIO()
        wb.save(out)
        result = parse_decisions(out.getvalue(), "xlsx")
        assert [(d.proposal_id, d.action, d.reason) for d in result.decisions] == [("p-1", "reject", "not needed")]

    def test_escaped_edited_value_is_unescaped_on_import(self):
        csv_bytes = _decision_csv([{"proposal_id": "p-1", "action": "edit", "edited_value": "'=VENDOR"}])
        assert parse_decisions(csv_bytes, "csv").decisions[0].edited_value == "=VENDOR"

    def test_escape_then_import_is_identity_for_formula_text(self):
        text = export_csv([{**PROPOSAL, "proposed_value": "=A1"}]).decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(text)))
        rows[1][EXPORT_COLUMNS.index("action")] = "edit"
        rows[1][EXPORT_COLUMNS.index("edited_value")] = rows[1][EXPORT_COLUMNS.index("proposed_value")]
        buf = io.StringIO()
        csv.writer(buf).writerows(rows)
        assert parse_decisions(buf.getvalue().encode(), "csv").decisions[0].edited_value == "=A1"


class TestParseValidation:
    def test_blank_action_rows_are_skipped_not_errors(self):
        result = parse_decisions(_decision_csv([{"proposal_id": "p-1"}]), "csv")
        assert (len(result.decisions), len(result.errors), result.skipped) == (0, 0, 1)

    def test_action_is_case_insensitive(self):
        result = parse_decisions(_decision_csv([{"proposal_id": "p-1", "action": " Approve "}]), "csv")
        assert result.decisions[0].action == "approve"

    def test_unknown_action_is_a_row_error(self):
        result = parse_decisions(_decision_csv([{"proposal_id": "p-1", "action": "delete"}]), "csv")
        assert result.errors[0].row == 2 and "action" in result.errors[0].message

    def test_edit_requires_edited_value(self):
        result = parse_decisions(_decision_csv([{"proposal_id": "p-1", "action": "edit"}]), "csv")
        assert "edited_value" in result.errors[0].message

    def test_merge_requires_merge_target(self):
        result = parse_decisions(_decision_csv([{"proposal_id": "p-1", "action": "merge"}]), "csv")
        assert "merge_target" in result.errors[0].message

    @pytest.mark.parametrize("action", ["reject", "quarantine"])
    def test_reject_and_quarantine_require_a_reason(self, action):
        result = parse_decisions(_decision_csv([{"proposal_id": "p-1", "action": action}]), "csv")
        assert "reason" in result.errors[0].message

    def test_approve_with_stray_merge_target_is_rejected(self):
        rows = [{"proposal_id": "p-1", "action": "approve", "merge_target": "X"}]
        assert "merge_target" in parse_decisions(_decision_csv(rows), "csv").errors[0].message

    def test_duplicate_proposal_ids_reject_every_duplicate_row(self):
        rows = [{"proposal_id": "p-1", "action": "approve"}, {"proposal_id": "p-1", "action": "defer"}]
        result = parse_decisions(_decision_csv(rows), "csv")
        assert (result.decisions, [e.row for e in result.errors]) == ((), [2, 3])

    def test_invalid_proposal_id_is_a_row_error(self):
        result = parse_decisions(_decision_csv([{"proposal_id": "p 1; DROP", "action": "approve"}]), "csv")
        assert "proposal_id" in result.errors[0].message

    def test_valid_rows_are_kept_alongside_row_errors(self):
        rows = [{"proposal_id": "p-1", "action": "approve"}, {"proposal_id": "p-2", "action": "nope"}]
        result = parse_decisions(_decision_csv(rows), "csv")
        assert ([d.proposal_id for d in result.decisions], [e.row for e in result.errors]) == (["p-1"], [3])

    def test_unknown_columns_are_ignored(self):
        header = ("proposal_id", "action", "reviewed_by", "kind")
        rows = [{"proposal_id": "p-1", "action": "approve", "reviewed_by": "mallory"}]
        decision = parse_decisions(_decision_csv(rows, header), "csv").decisions[0]
        assert not hasattr(decision, "reviewed_by")

    def test_missing_required_column_is_fatal(self):
        with pytest.raises(ExchangeError) as err:
            parse_decisions(_decision_csv([], header=("proposal_id", "reason")), "csv")
        assert err.value.code == "missing_columns"

    def test_header_matching_is_case_and_space_insensitive(self):
        header = ("Proposal ID", "ACTION")
        result = parse_decisions(_decision_csv([{"Proposal ID": "p-1", "ACTION": "approve"}], header), "csv")
        assert result.decisions[0].proposal_id == "p-1"

    def test_too_many_rows_is_fatal(self):
        rows = [{"proposal_id": f"p-{i}", "action": "approve"} for i in range(MAX_DECISION_ROWS + 1)]
        with pytest.raises(ExchangeError) as err:
            parse_decisions(_decision_csv(rows), "csv")
        assert err.value.code == "too_many_rows"

    def test_overlong_cell_is_a_row_error(self):
        rows = [{"proposal_id": "p-1", "action": "defer", "reason": "x" * 5000}]
        assert "reason" in parse_decisions(_decision_csv(rows), "csv").errors[0].message

    def test_non_utf8_csv_is_fatal(self):
        with pytest.raises(ExchangeError) as err:
            parse_decisions(b"\xff\xfe\x00bad", "csv")
        assert err.value.code == "invalid_file"

    def test_garbage_xlsx_is_fatal(self):
        with pytest.raises(ExchangeError) as err:
            parse_decisions(b"not a zip", "xlsx")
        assert err.value.code == "invalid_file"

    def test_xlsx_rows_parse(self):
        data = _decision_xlsx([["proposal_id", "action", "merge_target"], ["p-1", "merge", "Supplier"]])
        decision = parse_decisions(data, "xlsx").decisions[0]
        assert (decision.action, decision.merge_target) == ("merge", "Supplier")

    def test_xlsx_formula_cell_is_a_row_error(self):
        data = _decision_xlsx([["proposal_id", "action", "reason"], ["p-1", "defer", "=1+1"]])
        result = parse_decisions(data, "xlsx")
        assert (result.decisions, "formula" in result.errors[0].message) == ((), True)

    def test_unsupported_format_is_fatal(self):
        with pytest.raises(ExchangeError) as err:
            parse_decisions(b"x", "pdf")
        assert err.value.code == "unsupported_format"

    def test_allowed_actions_match_service(self):
        from graphrag.graph.ontology_proposals import OntologyProposalService
        assert set(ALLOWED_ACTIONS) == set(OntologyProposalService._ACTION_STATUS)
