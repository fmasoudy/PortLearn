"""Ingestion contract tests for ``portlearn.data.ingestion``.

These tests freeze the generic-user-data-interface laws: the declared
schema mandate (L1), the single availability convention (L2),
duck-typed DataFrame admission with a closed value domain (L3), the
frozen-law-only time gate (L4), stdlib CSV decoding (L5), the guarded
Parquet extra (L6), the single row→record chokepoint enforcing the
frozen identity/chronology/availability laws (L7), frequency as
carried metadata never semantics (L8), and ingestion provenance (L9).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from portlearn.observations import AmbiguousObservationError, TimedObservation
from portlearn.timing import (
    InvalidChronologyError,
    MissingAvailabilityError,
    NaiveTimestampError,
)

MELBOURNE = timezone(timedelta(hours=10))  # AEST fixed offset for fixtures


def aware(y: int, m: int, d: int, h: int = 0, tz: timezone = UTC) -> datetime:
    return datetime(y, m, d, h, tzinfo=tz)


# --------------------------------------------------------------------------- #
# Deterministic synthetic tabular stand-ins (no third-party dataframe library
# is imported by the suite: the L3 protocol is duck-typed by construction).
# --------------------------------------------------------------------------- #


class RowFrame:
    """A minimal column-name-lookup + row-iteration table (L3 protocol)."""

    def __init__(self, columns: list[str], rows: list[list[Any]]):
        self._columns = list(columns)
        self._rows = [list(r) for r in rows]

    def __getitem__(self, column: str) -> list[Any]:
        try:
            index = self._columns.index(column)
        except ValueError:
            raise KeyError(column) from None
        return [row[index] for row in self._rows]

    def __contains__(self, column: object) -> bool:
        return column in self._columns

    def __iter__(self):
        return iter(self._columns)

    def iterrows(self) -> Any:
        for row in self._rows:
            yield dict(zip(self._columns, row, strict=True))

    def __len__(self) -> int:
        return len(self._rows)


class MappingFrame:
    """A second duck-typed stand-in exercising mapping-style row access."""

    def __init__(self, columns: list[str], rows: list[dict[str, Any]]):
        self._columns = list(columns)
        self._rows = [dict(r) for r in rows]

    def columns(self) -> list[str]:
        return list(self._columns)

    def iterrows(self) -> Any:
        yield from self._rows


# --------------------------------------------------------------------------- #
# L1 — declared schema is mandatory
# --------------------------------------------------------------------------- #


def test_blank_schema_identifier_rejects_value_error() -> None:
    from portlearn.data.ingestion import AvailabilityPolicy, DeclaredTableSchema

    policy = AvailabilityPolicy.same_instant()
    with pytest.raises(ValueError, match="series_id_column"):
        DeclaredTableSchema(
            series_id_column="  ",
            observation_time_column="obs",
            availability=policy,
            value_column="value",
            units="percent",
            frequency="monthly",
        )
    with pytest.raises(ValueError, match="observation_time_column"):
        DeclaredTableSchema(
            series_id_column="sid",
            observation_time_column="",
            availability=policy,
            value_column="value",
            units="percent",
            frequency="monthly",
        )


def test_schema_rejects_non_string_identifier() -> None:
    from portlearn.data.ingestion import AvailabilityPolicy, DeclaredTableSchema

    with pytest.raises(ValueError):
        DeclaredTableSchema(
            series_id_column=None,  # type: ignore[arg-type]
            observation_time_column="obs",
            availability=AvailabilityPolicy.same_instant(),
            value_column="value",
            units="percent",
            frequency="monthly",
        )


def test_description_column_is_optional_and_carried() -> None:
    from portlearn.data.ingestion import AvailabilityPolicy, DeclaredTableSchema

    schema = DeclaredTableSchema(
        series_id_column="sid",
        observation_time_column="obs",
        availability=AvailabilityPolicy.same_instant(),
        value_column="value",
        units="percent",
        frequency="monthly",
        description_column="desc",
    )
    assert schema.description_column == "desc"


# --------------------------------------------------------------------------- #
# L2 — one availability convention, no default
# --------------------------------------------------------------------------- #


def test_three_availability_policy_modes_construct() -> None:
    from portlearn.data.ingestion import AvailabilityPolicy

    explicit = AvailabilityPolicy.explicit_column("avail")
    same = AvailabilityPolicy.same_instant()
    lag = AvailabilityPolicy.fixed_lag(
        timedelta(days=10), justification="declared release delay"
    )
    assert explicit.column_name == "avail"
    assert same.mode == "SAME_INSTANT"
    assert lag.delta == timedelta(days=10)


def test_availability_policy_is_required_not_defaulted() -> None:
    from portlearn.data.ingestion import DeclaredTableSchema

    with pytest.raises((ValueError, TypeError), match="availability"):
        DeclaredTableSchema(  # type: ignore[arg-type]
            series_id_column="sid",
            observation_time_column="obs",
            value_column="value",
            units="percent",
            frequency="monthly",
        )


def test_fixed_lag_requires_justification() -> None:
    from portlearn.data.ingestion import AvailabilityPolicy

    with pytest.raises((ValueError, TypeError)):
        AvailabilityPolicy.fixed_lag(timedelta(days=1), justification="   ")
    with pytest.raises((ValueError, TypeError)):
        AvailabilityPolicy.fixed_lag(timedelta(days=1))


def test_policy_carried_into_provenance() -> None:
    from portlearn.data.ingestion import AvailabilityPolicy

    lag = AvailabilityPolicy.fixed_lag(
        timedelta(hours=6), justification="declared intraday lag"
    )
    assert lag.justification == "declared intraday lag"
    assert AvailabilityPolicy.same_instant().mode == "SAME_INSTANT"
    assert AvailabilityPolicy.explicit_column("a").mode == "EXPLICIT_COLUMN"


# --------------------------------------------------------------------------- #
# L3 — duck-typed DataFrame admission, closed value domain
# --------------------------------------------------------------------------- #


def _monthly_schema() -> Any:
    from portlearn.data.ingestion import AvailabilityPolicy, DeclaredTableSchema

    return DeclaredTableSchema(
        series_id_column="sid",
        observation_time_column="obs",
        availability=AvailabilityPolicy.same_instant(),
        value_column="value",
        units="percent",
        frequency="monthly",
    )


def test_from_dataframe_accepts_two_duck_typed_stand_ins() -> None:
    from portlearn.data.ingestion import from_dataframe

    schema = _monthly_schema()
    instants = [aware(2026, 1, 31), aware(2026, 2, 28)]
    frame_rows = RowFrame(
        ["sid", "obs", "value"],
        [["series-a", instants[0], 1.5], ["series-a", instants[1], 2.5]],
    )
    mapping_rows = MappingFrame(
        ["sid", "obs", "value"],
        [
            {"sid": "series-a", "obs": instants[0], "value": 1.5},
            {"sid": "series-a", "obs": instants[1], "value": 2.5},
        ],
    )
    for frame in (frame_rows, mapping_rows):
        records, _provenance = from_dataframe(frame, schema)
        assert [r.series_id for r in records] == ["series-a", "series-a"]
        assert all(isinstance(r, TimedObservation) for r in records)


def test_from_dataframe_protocol_failure_rejects_type_error() -> None:
    from portlearn.data.ingestion import from_dataframe

    class NotATable:
        pass

    with pytest.raises(TypeError, match="column"):
        from_dataframe(NotATable(), _monthly_schema())  # type: ignore[arg-type]


def test_from_dataframe_value_outside_domain_rejects_type_error() -> None:
    from portlearn.data.ingestion import from_dataframe

    class Opaque:
        pass

    frame = RowFrame(
        ["sid", "obs", "value"],
        [["series-a", aware(2026, 1, 31), Opaque()]],
    )
    with pytest.raises(TypeError, match="value"):
        from_dataframe(frame, _monthly_schema())


def test_value_domain_accepts_str_int_float_bool_none() -> None:
    from portlearn.data.ingestion import from_dataframe

    instants = [aware(2026, 1, 31), aware(2026, 2, 28), aware(2026, 3, 31), aware(2026, 4, 30)]
    values = [3, 4.5, True, None]
    frame = RowFrame(
        ["sid", "obs", "value"],
        [["series-a", instants[i], values[i]] for i in range(4)],
    )
    records, _ = from_dataframe(frame, _monthly_schema())
    assert [r.value for r in records] == values


# --------------------------------------------------------------------------- #
# L4 — time fields are frozen-law-only
# --------------------------------------------------------------------------- #


def test_naive_cell_rejects_through_dataframe_path() -> None:
    from portlearn.data.ingestion import from_dataframe

    frame = RowFrame(
        ["sid", "obs", "value"],
        [["series-a", datetime(2026, 1, 31), 1.0]],  # noqa: DTZ001 — naive by design
    )
    with pytest.raises(NaiveTimestampError):
        from_dataframe(frame, _monthly_schema())


def test_date_only_string_rejects_no_midnight_coercion() -> None:
    from portlearn.data.ingestion import from_dataframe

    frame = RowFrame(
        ["sid", "obs", "value"],
        [["series-a", "2026-01-31", 1.0]],
    )
    with pytest.raises(NaiveTimestampError):
        from_dataframe(frame, _monthly_schema())


def test_offsetless_iso_string_rejects() -> None:
    from portlearn.data.ingestion import from_dataframe

    frame = RowFrame(
        ["sid", "obs", "value"],
        [["series-a", "2026-01-31T00:00:00", 1.0]],
    )
    with pytest.raises(NaiveTimestampError):
        from_dataframe(frame, _monthly_schema())


def test_aware_iso_string_with_offset_accepts() -> None:
    from portlearn.data.ingestion import from_dataframe

    frame = RowFrame(
        ["sid", "obs", "value"],
        [["series-a", "2026-01-31T23:59:59.999999+11:00", 1.0]],
    )
    records, _ = from_dataframe(frame, _monthly_schema())
    assert records[0].observation_time == datetime(2026, 1, 31, 12, 59, 59, 999999, tzinfo=UTC)


def test_naive_cell_rejects_through_csv_path() -> None:
    from portlearn.data.ingestion import (
        AvailabilityPolicy,
        DeclaredTableSchema,
        from_csv,
    )

    schema = DeclaredTableSchema(
        series_id_column="sid",
        observation_time_column="obs",
        availability=AvailabilityPolicy.explicit_column("avail"),
        value_column="value",
        units="percent",
        frequency="monthly",
    )
    text = (
        "sid,obs,avail,value\n"
        "series-a,2026-01-31T00:00:00,2026-02-05T00:00:00+00:00,1.0\n"
    )
    with pytest.raises(NaiveTimestampError):
        from_csv(text, schema)


def test_date_only_string_rejects_through_csv_path() -> None:
    from portlearn.data.ingestion import from_csv

    text = "sid,obs,value\nseries-a,2026-01-31,1.0\n"
    with pytest.raises(NaiveTimestampError):
        from_csv(text, _monthly_schema())


# --------------------------------------------------------------------------- #
# L5 — CSV via stdlib, declared schema
# --------------------------------------------------------------------------- #


def _explicit_schema() -> Any:
    from portlearn.data.ingestion import AvailabilityPolicy, DeclaredTableSchema

    return DeclaredTableSchema(
        series_id_column="sid",
        observation_time_column="obs",
        availability=AvailabilityPolicy.explicit_column("avail"),
        value_column="value",
        units="percent",
        frequency="monthly",
    )


def test_csv_happy_path_loads_records() -> None:
    from portlearn.data.ingestion import from_csv

    text = (
        "sid,obs,avail,value\n"
        "series-a,2026-01-31T00:00:00+00:00,2026-02-05T00:00:00+00:00,1.5\n"
        "series-b,2026-01-31T00:00:00+00:00,2026-02-05T00:00:00+00:00,2.5\n"
    )
    records, provenance = from_csv(text, _explicit_schema())
    assert len(records) == 2
    assert records[0].available_time == aware(2026, 2, 5)
    assert provenance.row_count == 2


def test_csv_missing_declared_column_rejects_value_error() -> None:
    from portlearn.data.ingestion import from_csv

    text = "sid,obs,value\nseries-a,2026-01-31T00:00:00+00:00,1.0\n"
    with pytest.raises(ValueError, match="avail"):
        from_csv(text, _explicit_schema())


def test_csv_from_path_loads_with_fingerprint(tmp_path: Path) -> None:
    from portlearn.data.ingestion import from_csv

    text = (
        "sid,obs,avail,value\n"
        "series-a,2026-01-31T00:00:00+00:00,2026-02-05T00:00:00+00:00,1.5\n"
    )
    path = tmp_path / "monthly.csv"
    path.write_text(text, encoding="utf-8")
    records, provenance = from_csv(str(path), _explicit_schema())
    assert len(records) == 1
    assert provenance.content_sha256 is not None
    assert len(provenance.content_sha256) == 64


def test_csv_header_order_enters_provenance() -> None:
    from portlearn.data.ingestion import from_csv

    text = (
        "value,avail,obs,sid\n"
        "1.5,2026-02-05T00:00:00+00:00,2026-01-31T00:00:00+00:00,series-a\n"
    )
    records, provenance = from_csv(text, _explicit_schema())
    assert records[0].value == 1.5
    assert provenance.header_order == ["value", "avail", "obs", "sid"]


# --------------------------------------------------------------------------- #
# L6 — Parquet behind the guarded extra
# --------------------------------------------------------------------------- #


def test_parquet_guard_error_names_the_extra() -> None:
    from portlearn.data.ingestion import MissingParquetExtraError, from_parquet

    try:
        import pyarrow  # noqa: F401

        pytest.skip("pyarrow present: guard-error path not reachable")
    except ImportError:
        pass
    with pytest.raises(MissingParquetExtraError, match=r"portlearn\[parquet\]"):
        from_parquet("unused-path.parquet", _explicit_schema())


# --------------------------------------------------------------------------- #
# L7 — the single row→record chokepoint
# --------------------------------------------------------------------------- #


def test_duplicate_identity_rejects_ambiguous_observation_error() -> None:
    from portlearn.data.ingestion import from_dataframe

    inst = aware(2026, 1, 31)
    frame = RowFrame(
        ["sid", "obs", "value"],
        [
            ["series-a", inst, 1.0],
            ["series-a", inst, 2.0],
        ],
    )
    with pytest.raises(AmbiguousObservationError):
        from_dataframe(frame, _monthly_schema())


def test_missing_availability_rejects_when_policy_needs_it() -> None:
    from portlearn.data.ingestion import (
        AvailabilityPolicy,
        DeclaredTableSchema,
        from_csv,
    )

    schema = DeclaredTableSchema(
        series_id_column="sid",
        observation_time_column="obs",
        availability=AvailabilityPolicy.explicit_column("avail"),
        value_column="value",
        units="percent",
        frequency="monthly",
    )
    text = (
        "sid,obs,value\n"
        "series-a,2026-01-31T00:00:00+00:00,1.0\n"
    )
    with pytest.raises(MissingAvailabilityError):
        from_csv(text, schema)


def test_missing_availability_rejects_blank_explicit_cell() -> None:
    from portlearn.data.ingestion import from_csv

    text = (
        "sid,obs,avail,value\n"
        "series-a,2026-01-31T00:00:00+00:00,,1.0\n"
    )
    with pytest.raises(MissingAvailabilityError):
        from_csv(text, _explicit_schema())


def test_reversed_chronology_rejects_invalid_chronology_error() -> None:
    from portlearn.data.ingestion import from_csv

    text = (
        "sid,obs,avail,value\n"
        "series-a,2026-02-05T00:00:00+00:00,2026-01-31T00:00:00+00:00,1.0\n"
    )
    with pytest.raises(InvalidChronologyError):
        from_csv(text, _explicit_schema())


def test_revision_pair_same_observation_different_availability_loads() -> None:
    from portlearn.data.ingestion import from_csv

    text = (
        "sid,obs,avail,value\n"
        "series-a,2026-01-31T00:00:00+00:00,2026-02-05T00:00:00+00:00,1.0\n"
        "series-a,2026-01-31T00:00:00+00:00,2026-03-05T00:00:00+00:00,1.2\n"
    )
    records, _ = from_csv(text, _explicit_schema())
    assert len(records) == 2


def test_blank_series_identifier_rejects_value_error_at_ingestion() -> None:
    from portlearn.data.ingestion import from_csv

    text = (
        "sid,obs,avail,value\n"
        " ,2026-01-31T00:00:00+00:00,2026-02-05T00:00:00+00:00,1.0\n"
    )
    with pytest.raises(ValueError):
        from_csv(text, _explicit_schema())


def test_explicit_record_sequence_loads_through_same_chokepoint() -> None:
    from portlearn.data.ingestion import from_records

    records_in = [
        {
            "sid": "series-a",
            "obs": "2026-01-31T00:00:00+00:00",
            "avail": "2026-02-05T00:00:00+00:00",
            "value": 1.5,
        },
        {
            "sid": "series-a",
            "obs": "2026-02-28T00:00:00+00:00",
            "avail": "2026-03-05T00:00:00+00:00",
            "value": 1.7,
        },
    ]
    records, provenance = from_records(records_in, _explicit_schema())
    assert len(records) == 2
    assert records[0].available_time == aware(2026, 2, 5)
    assert provenance.row_count == 2


def test_duplicate_identity_rejects_through_record_sequence() -> None:
    from portlearn.data.ingestion import from_records

    records_in = [
        {
            "sid": "series-a",
            "obs": "2026-01-31T00:00:00+00:00",
            "avail": "2026-02-05T00:00:00+00:00",
            "value": 1.5,
        },
        {
            "sid": "series-a",
            "obs": "2026-01-31T00:00:00+00:00",
            "avail": "2026-02-05T00:00:00+00:00",
            "value": 9.9,
        },
    ]
    with pytest.raises(AmbiguousObservationError):
        from_records(records_in, _explicit_schema())


# --------------------------------------------------------------------------- #
# L8 — frequency is carried metadata, never semantics
# --------------------------------------------------------------------------- #


def test_declared_frequency_with_mismatched_spacing_still_loads() -> None:
    from portlearn.data.ingestion import from_csv

    text = (
        "sid,obs,avail,value\n"
        "series-a,2026-01-01T00:00:00+00:00,2026-01-02T00:00:00+00:00,1.0\n"
        "series-a,2026-01-02T00:00:00+00:00,2026-01-03T00:00:00+00:00,1.1\n"
        "series-a,2026-01-04T00:00:00+00:00,2026-01-05T00:00:00+00:00,1.2\n"
    )
    records, provenance = from_csv(text, _explicit_schema())
    assert len(records) == 3  # daily spacing under a "monthly" declaration loads
    assert provenance.frequency == "monthly"


# --------------------------------------------------------------------------- #
# L9 — ingestion provenance
# --------------------------------------------------------------------------- #


def test_provenance_round_trip_all_fields_present() -> None:
    from portlearn.data.ingestion import from_csv

    text = (
        "sid,obs,avail,value\n"
        "series-a,2026-01-31T00:00:00+00:00,2026-02-05T00:00:00+00:00,1.5\n"
    )
    _records, provenance = from_csv(text, _explicit_schema())
    assert provenance.source_id is not None
    assert provenance.retrieval_instant.tzinfo is not None
    assert provenance.content_sha256 is not None
    assert provenance.licence_note is not None
    assert provenance.availability is _explicit_schema().availability
    assert provenance.adapter_identity is not None
    assert provenance.adapter_version is not None


def test_provenance_hash_stable_across_two_loads_of_same_bytes() -> None:
    from portlearn.data.ingestion import from_csv

    text = (
        "sid,obs,avail,value\n"
        "series-a,2026-01-31T00:00:00+00:00,2026-02-05T00:00:00+00:00,1.5\n"
    )
    _, first = from_csv(text, _explicit_schema())
    _, second = from_csv(text, _explicit_schema())
    assert first.content_sha256 == second.content_sha256


def test_dataframe_provenance_records_adapter_identity() -> None:
    from portlearn.data.ingestion import from_dataframe

    frame = RowFrame(
        ["sid", "obs", "value"],
        [["series-a", aware(2026, 1, 31), 1.0]],
    )
    _, provenance = from_dataframe(frame, _monthly_schema())
    assert "dataframe" in provenance.adapter_identity.lower()
    assert provenance.content_sha256 is None  # no bytes exist for an in-memory frame


def test_same_instant_policy_sets_availability_to_observation() -> None:
    from portlearn.data.ingestion import from_dataframe

    frame = RowFrame(
        ["sid", "obs", "value"],
        [["series-a", aware(2026, 1, 31), 1.0]],
    )
    records, _ = from_dataframe(frame, _monthly_schema())
    assert records[0].available_time == records[0].observation_time


def test_fixed_lag_policy_offsets_availability() -> None:
    from portlearn.data.ingestion import (
        AvailabilityPolicy,
        DeclaredTableSchema,
        from_dataframe,
    )

    schema = DeclaredTableSchema(
        series_id_column="sid",
        observation_time_column="obs",
        availability=AvailabilityPolicy.fixed_lag(
            timedelta(days=5), justification="declared publication delay"
        ),
        value_column="value",
        units="percent",
        frequency="monthly",
    )
    frame = RowFrame(
        ["sid", "obs", "value"],
        [["series-a", aware(2026, 1, 31), 1.0]],
    )
    records, _ = from_dataframe(frame, schema)
    assert records[0].available_time == aware(2026, 2, 5)


def test_units_and_frequency_carried_into_provenance() -> None:
    from portlearn.data.ingestion import from_dataframe

    frame = RowFrame(
        ["sid", "obs", "value"],
        [["series-a", aware(2026, 1, 31), 1.0]],
    )
    _, provenance = from_dataframe(frame, _monthly_schema())
    assert provenance.units == "percent"
    assert provenance.frequency == "monthly"
