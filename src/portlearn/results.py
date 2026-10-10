"""The universal result and provenance schema (accepted results-schema decision).

This module is the single result surface of a run: eight public value
objects (``ResultRecord``, ``PathResult``, ``MetricsBlock``,
``ForecastLink``, ``InformationSetAnchor``, ``ConformanceRecord``,
``SanctionedDeviation``, ``ParameterRelation``), the canonical JSON
serializer/parser pair (``ResultRecord.to_json`` /
``ResultRecord.from_json``), and the one public digest helper
``ledger_digest``.

Design rules implemented here (the results design):

* **Canonical form** — every serialization is sorted-key,
  compact, ASCII, ``allow_nan=False`` JSON; float text is the
  interpreter's shortest round-trip repr.
* **Instants** — aware-only, offset-preserving ISO-8601 text;
  naive instants reject with the timing module's own
  ``NaiveTimestampError`` (this module owns no second timezone
  policy and never shadows that error).
* **Ledger digest** — sha256 over the canonical JSON of the
  bare row array (twelve-key row objects), no wrapper.
* **Horizon grammar** — the CPython ``str(timedelta)`` wire
  form, strictly positive; a pinned parser owns the read side.
* **Verdict** — computed from the declared/effective pair with
  signed-zero normalization and structured sanctions; recomputed at
  parse (tamper-evidence).
* **Parse** — exact key-set equality at every fixed-shape
  object, duplicate-key rejection at every depth, unknown-version,
  nonfinite-literal, and naive-instant rejection.
* **γ/λ relations** — every distinct pairing of a block's
  ``gamma`` with the top-level effective ``risk_aversion`` requires
  a covering ``ParameterRelation``.
* **Sequences** — forecasts, per-path metric blocks,
  sanctions, and relations store canonically (sorted by each
  element's own canonical JSON bytes); caller submission order can
  never alter the record's bytes, and duplicates reject.

Error ownership: ``ResultStructureError`` **is**
``ValueError`` — the module-owned rejection path is the stdlib error the
fail-closed rules already raise (``allow_nan=False`` serialization
rejection, and every structural admission in this module raises it).
``NaiveTimestampError`` remains the timing module's own class; this
module raises it via the public ``to_instant`` validator and never
re-defines it.

Stdlib-only: this module imports no third-party package, directly
or transitively beyond the PortLearn public interfaces it consumes
(``manifest``, ``ledger``, ``metrics``, ``forecasting``,
``interfaces``, ``timing`` — all public constructors and types).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from itertools import pairwise
from types import MappingProxyType
from typing import Any

from portlearn.forecasting import DeterminismClass
from portlearn.interfaces import InformationSet
from portlearn.ledger import LedgerPath
from portlearn.manifest import RunManifest
from portlearn.metrics import MetricsSummary
from portlearn.timing import to_instant

__all__ = [  # alphabetical — the results §4 D1 pinned order
    "ConformanceRecord",
    "ForecastLink",
    "InformationSetAnchor",
    "MetricsBlock",
    "ParameterRelation",
    "PathResult",
    "ResultRecord",
    "ResultStructureError",
    "SanctionedDeviation",
    "ledger_digest",
]


# The module-owned structural error arm (§4 D9): a subclass of the
# stdlib ValueError, so existing ``except ValueError`` surfaces keep
# working while this module owns its error class outright. The
# fail-closed laws that surface through stdlib guards —
# ``json.dumps(allow_nan=False)`` among them — reject with exactly
# this class everywhere: construction, serialization, and parse alike
# (the stdlib rejection is wrapped below, never allowed to leak as a
# bare ``ValueError``).
class ResultStructureError(ValueError):
    """A result-surface structural rejection.

    Every structural admission on this module — the record, block,
    link, anchor, settings, relation, and parse rules — rejects with
    this class; it derives from ``ValueError`` so stdlib error
    handling continues to catch it, but it is owned here, never
    aliased to the stdlib class.
    """


_SCHEMA_VERSION = "1.0.0"

# The two §5.10 role names of the decision-layer decision/evaluation pairing.
_GAMMA_ROLE = "evaluation_gamma"
_RISK_AVERSION_ROLE = "decision_risk_aversion"

_UNSET = object()
_MISSING = object()

_METRIC_NAMES = (
    "cumulative_return",
    "annualized_mean_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "certainty_equivalent_return",
)


# --------------------------------------------------------------------------- #
# Canonical JSON (§5.1–§5.4) and the shared admission validators               #
# --------------------------------------------------------------------------- #


def _dumps(value: Any) -> str:
    """The canonical serialization rule: sorted keys, compact, ASCII, no nonfinite."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except ValueError as nonserializable:
        # The stdlib guard (allow_nan=False among its rejections) is an
        # results-owned structural failure: it rejects as this module's
        # error class, never as a bare stdlib ValueError.
        raise ResultStructureError(
            f"the value is not representable in canonical JSON "
            f"(sorted, compact, ASCII, nonfinite-free): {nonserializable}."
        ) from None


def _require_nonblank_str(value: Any, field: str) -> str:
    """Admit a non-blank string or reject with ``ResultStructureError``."""
    if not isinstance(value, str) or not value.strip():
        raise ResultStructureError(
            f"{field} must be a non-blank string; got {value!r}. Blank or "
            "non-string values name nothing on the result surface, so the "
            "value is rejected unconditionally."
        )
    return value


def _require_aware(value: Any, field: str) -> datetime:
    """Admit an aware instant via the timing module's public validator.

    ``to_instant`` raises timing's own ``NaiveTimestampError`` on naive
    datetimes, dates, and non-datetime input. The normalized UTC
    instant it returns serves comparison only — stored datetimes keep
    their original offsets, so the validated original is what
    flows through.
    """
    to_instant(value, field)
    return value


def _require_finite_number(value: Any, field: str) -> float:
    """Admit a finite exact ``int``/``float`` (never ``bool``), zero-signed."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultStructureError(
            f"{field} must be a finite int or float; got {value!r}. "
            "Booleans are not numbers on this surface."
        )
    number = float(value)
    if not math.isfinite(number):
        raise ResultStructureError(
            f"{field} must be finite; got {value!r}. Every float on the "
            "result surface is finite — fail-closed in both directions."
        )
    return number + 0.0


def _require_exact_keys(node: Any, required: frozenset[str], what: str) -> None:
    """Exact key-set equality at one fixed-shape JSON object."""
    if not isinstance(node, dict):
        raise ResultStructureError(
            f"{what} must be a JSON object; got {type(node).__name__}. The "
            "schema admits no silent forward-compatibility."
        )
    extra = sorted(set(node) - required)
    missing = sorted(required - set(node))
    if extra or missing:
        raise ResultStructureError(
            f"{what} carries an invalid key set: "
            f"unexpected keys {extra}, missing keys {missing}. Exact "
            "key-set equality holds at every object on the wire."
        )


def _require_wire_sequence(node: Any, what: str) -> list[Any]:
    """Admit a JSON array (never an object or scalar) as a sequence."""
    if not isinstance(node, list):
        raise ResultStructureError(
            f"{what} must be a JSON array; got {type(node).__name__}."
        )
    return node


def _require_wire_order(elements: list[Any], what: str) -> None:
    """Reject any sequence not in strict canonical-byte order."""
    keys = [_dumps(element) for element in elements]
    for previous, current in pairwise(keys):
        if not previous < current:
            raise ResultStructureError(
                f"{what} is not in canonical order (or carries a duplicate): "
                "the canonical form is total — sorted by each element's own "
                "canonical JSON bytes — and the parser accepts no other "
                "order and no repeated element."
            )


def _ordered_and_unique(
    items: tuple[Any, ...], wire_of: Any, what: str
) -> tuple[Any, ...]:
    """Store a sequence canonically, blind to submission order."""
    decorated = sorted(
        ((_dumps(wire_of(item)), item) for item in items),
        key=lambda pair: pair[0],
    )
    for previous, current in pairwise(decorated):
        if not previous[0] < current[0]:
            raise ResultStructureError(
                f"{what} carries two canonically identical elements: the "
                "sequence holds exactly one element per distinct value — "
                "no duplicates, at construction and at parse."
            )
    return tuple(item for _, item in decorated)


# --------------------------------------------------------------------------- #
# The §5.6a ledger-digest payload                                              #
# --------------------------------------------------------------------------- #


def _weight_map(weights: Mapping[str, Any]) -> dict[str, float]:
    """A canonical weight map: exact asset keys, floats, signed zero unified."""
    return {asset: float(weights[asset]) + 0.0 for asset in weights}


def _trade_wire(trade: Any) -> dict[str, Any]:
    """The canonical trade projection: exactly source/destination/delta."""
    return {
        "source": {
            "state": trade.source.state.value,
            "weights": _weight_map(trade.source.weights),
        },
        "destination": {
            "state": trade.destination.state.value,
            "weights": _weight_map(trade.destination.weights),
        },
        "delta": _weight_map(trade.delta),
    }


def _execution_wire(execution: Any) -> dict[str, Any]:
    """The canonical execution projection: exactly the eight Execution keys."""
    return {
        "decision_time": execution.decision_time.isoformat(),
        "execution_time": execution.execution_time.isoformat(),
        "target_weights": _weight_map(execution.target_weights),
        "pre_trade_weights": _weight_map(execution.pre_trade_weights),
        "trade": _trade_wire(execution.trade),
        "turnover": float(execution.turnover) + 0.0,
        "transaction_cost": float(execution.transaction_cost) + 0.0,
        "post_trade_weights": _weight_map(execution.post_trade_weights),
    }


def _row_wire(row: Any) -> dict[str, Any]:
    """The canonical row projection: exactly the twelve LedgerRow keys."""
    return {
        "period_start": row.period_start.isoformat(),
        "period_end": row.period_end.isoformat(),
        "opening_weights": _weight_map(row.opening_weights),
        "executions": [_execution_wire(e) for e in row.executions],
        "closing_weights": _weight_map(row.closing_weights),
        "gross_return": float(row.gross_return) + 0.0,
        "net_return": float(row.net_return) + 0.0,
        "turnover": float(row.turnover) + 0.0,
        "transaction_cost": float(row.transaction_cost) + 0.0,
        "wealth_open": float(row.wealth_open) + 0.0,
        "wealth_close": float(row.wealth_close) + 0.0,
        "universe": sorted(row.universe),
    }


def _rows_digest(path: LedgerPath) -> str:
    """sha256 over the canonical JSON of the bare row array — no wrapper."""
    payload = _dumps([_row_wire(row) for row in path.rows])
    return sha256(payload.encode("utf-8")).hexdigest()


def ledger_digest(path: LedgerPath) -> str:
    """The content digest of one complete ``LedgerPath``.

    The payload is the canonical JSON of the bare array of
    row projections — twelve keys per row, eight per execution, three
    per trade, weight maps keyed by asset — hashed with sha256 and
    hex-digested. No wrapper object, no path-level summary fields:
    the digest covers exactly the rows, so row order and execution
    order are content, while weight-map insertion order never is
    (canonical JSON sorts keys at every level).
    """
    if not isinstance(path, LedgerPath):
        raise ResultStructureError(
            "ledger_digest consumes a LedgerPath; got "
            f"{type(path).__name__}. The digest is derived from the "
            "consumed path, never accepted as a caller-supplied value."
        )
    return _rows_digest(path)


# --------------------------------------------------------------------------- #
# The §5.8a recursive settings grammar                                         #
# --------------------------------------------------------------------------- #


def _freeze_settings(value: Any, field: str) -> Any:
    """Validate one settings value and return its deep immutable snapshot.

    Valid kinds: ``None``, ``bool``, ``int``, finite ``float``
    (signed zero unified to ``+0.0``), ``str``, sequences (stored as
    tuples), and mappings with non-blank string keys (stored as
    mapping proxies). Everything else — bytes, enums, datetimes,
    sets, nonfinite floats, arbitrary objects — rejects with
    ``ResultStructureError``. The snapshot is deep: later mutation of
    a caller's submitted container can never reach the record.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResultStructureError(
                f"{field} must be finite; got {value!r}. Settings values "
                "are fail-closed in both directions."
            )
        return value + 0.0
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        raise ResultStructureError(
            f"{field} must be a valid settings value; bytes are not on "
            f"the grammar. Got {value!r}."
        )
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ResultStructureError(
                    f"{field} carries an invalid mapping key {key!r}: "
                    "keys must be non-blank strings."
                )
            frozen[key] = _freeze_settings(item, f"{field}[{key!r}]")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence):
        return tuple(
            _freeze_settings(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    raise ResultStructureError(
        f"{field} must be a valid settings value (None, bool, int, "
        f"finite float, str, sequence, or mapping); got {value!r} of "
        f"type {type(value).__name__}."
    )


def _settings_wire(value: Any) -> Any:
    """Project an immutable settings snapshot onto its JSON wire form."""
    if isinstance(value, Mapping):
        return {key: _settings_wire(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_settings_wire(item) for item in value]
    if isinstance(value, float):
        return value + 0.0
    return value


def _settings_equal(left: Any, right: Any) -> bool:
    """Canonical-content equality under the settings grammar."""
    if left is _MISSING or right is _MISSING:
        return left is right
    return _dumps(_settings_wire(left)) == _dumps(_settings_wire(right))


def _derive_verdict(
    declared: Mapping[str, Any],
    effective: Mapping[str, Any],
    deviations: tuple[SanctionedDeviation, ...],
) -> str:
    """Compute the verdict from the declared/effective pair.

    ``consistent`` iff the two sides are canonically equal and the
    deviation ledger is empty; ``sanctioned_deviation`` iff every
    top-level difference is covered by exactly one complete sanction
    and no sanction names a field that does not differ. Anything else
    rejects — no result proceeds on an unrecorded deviation.
    """
    sanctioned_fields = [deviation.field for deviation in deviations]
    if len(set(sanctioned_fields)) != len(sanctioned_fields):
        raise ResultStructureError(
            "the deviation ledger sanctions one field twice: a differing "
            "field is covered by exactly one sanction, never duplicated."
        )
    differing = {
        key
        for key in set(declared) | set(effective)
        if not _settings_equal(
            declared.get(key, _MISSING), effective.get(key, _MISSING)
        )
    }
    if not differing:
        if deviations:
            raise ResultStructureError(
                "the deviation ledger sanctions fields that do not differ: "
                "declared and effective settings are canonically equal, so "
                "no sanction is valid (signed zero is one value, not a "
                "difference)."
            )
        return "consistent"
    phantoms = sorted(set(sanctioned_fields) - differing)
    if phantoms:
        raise ResultStructureError(
            f"the deviation ledger sanctions fields that do not differ: "
            f"{phantoms}. A sanction covers an actual top-level difference."
        )
    uncovered = sorted(differing - set(sanctioned_fields))
    if uncovered:
        raise ResultStructureError(
            f"declared/effective settings differ at {sorted(differing)} "
            f"without sanction coverage for {uncovered}: no result "
            "proceeds on an unrecorded deviation."
        )
    return "sanctioned_deviation"


# --------------------------------------------------------------------------- #
# The §5.7 horizon grammar                                                    #
# --------------------------------------------------------------------------- #

_HORIZON_RE = re.compile(r"^(?:(\d+) days?, )?(\d+):([0-5]\d):([0-5]\d(?:\.\d{1,6})?)$")


def _require_positive_horizon(value: Any, field: str) -> timedelta:
    """Admit a strictly positive ``timedelta``."""
    if not isinstance(value, timedelta):
        raise ResultStructureError(
            f"{field} must be a datetime.timedelta; got {value!r}."
        )
    if value == timedelta(0):
        raise ResultStructureError(
            f"{field} must be strictly positive: a zero horizon is a "
            "nowcast of the present, not a forecast."
        )
    if value < timedelta(0):
        raise ResultStructureError(
            f"{field} must be strictly positive; got {value!r}. Negative "
            "durations reject fail-closed."
        )
    return value


def _horizon_from_wire(text: Any, field: str) -> timedelta:
    """Parse the pinned ``str(timedelta)`` grammar."""
    if not isinstance(text, str):
        raise ResultStructureError(
            f"{field} must be the documented str(timedelta) text; got {text!r}."
        )
    match = _HORIZON_RE.match(text)
    if match is None:
        raise ResultStructureError(
            f"{field} is not the documented str(timedelta) grammar "
            f'("1 day, 2:30:00", "21 days, 0:00:00", "6:00:00"); got '
            f"{text!r}."
        )
    days, hours, minutes, seconds = match.groups()
    micros = 0
    whole_seconds = seconds
    if "." in seconds:
        whole_seconds, fraction = seconds.split(".")
        micros = int(fraction.ljust(6, "0"))
    horizon = timedelta(
        days=int(days) if days else 0,
        hours=int(hours),
        minutes=int(minutes),
        seconds=int(whole_seconds),
        microseconds=micros,
    )
    if horizon <= timedelta(0):
        raise ResultStructureError(
            f"{field} must be strictly positive; the zero wire form "
            f"{text!r} is exactly the boundary the record rejects."
        )
    return horizon


# --------------------------------------------------------------------------- #
# D4 — MetricsBlock                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, init=False)
class MetricsBlock:
    """The declaration-complete metric block of one path under one basis.

    Exactly ten fields: the six ``MetricsSummary`` values with their
    public names and ``float`` types, plus the four declarations that
    produced them — ``basis`` (``"net"``/``"gross"``),
    ``periods_per_year`` (strictly positive ``int``), ``gamma``
    (strictly positive finite number), and ``risk_free`` (a finite
    scalar or a nonempty sequence of finite floats, stored as a
    tuple). Construction consumes a ``MetricsSummary`` positionally
    and derives nothing else; unknown keyword arguments reject with
    ``TypeError``; nonfinite metric content rejects fail-closed.
    """

    cumulative_return: float
    annualized_mean_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    certainty_equivalent_return: float
    basis: str
    periods_per_year: int
    gamma: float
    risk_free: float | tuple[float, ...]

    def __init__(
        self,
        summary: MetricsSummary,
        *,
        basis: str,
        periods_per_year: int,
        gamma: float,
        risk_free: float | Sequence[float],
        **unknown: Any,
    ) -> None:
        """Admit one summary under its four producing declarations."""
        if unknown:
            raise TypeError(
                f"MetricsBlock got unexpected keyword argument(s) "
                f"{sorted(unknown)}: the block is exactly ten fields."
            )
        if not isinstance(summary, MetricsSummary):
            raise ResultStructureError(
                "MetricsBlock consumes a MetricsSummary positionally; got "
                f"{type(summary).__name__}."
            )
        if basis not in ("net", "gross"):
            raise ResultStructureError(
                f"MetricsBlock.basis must be 'net' or 'gross'; got {basis!r}."
            )
        if (
            isinstance(periods_per_year, bool)
            or not isinstance(periods_per_year, int)
            or periods_per_year <= 0
        ):
            raise ResultStructureError(
                "MetricsBlock.periods_per_year must be a strictly positive "
                f"int; got {periods_per_year!r}."
            )
        gamma_value = _require_finite_number(gamma, "MetricsBlock.gamma")
        if gamma_value <= 0.0:
            raise ResultStructureError(
                f"MetricsBlock.gamma must be strictly positive; got {gamma!r}."
            )
        if isinstance(risk_free, (str, bytes)) or not isinstance(
            risk_free, (Sequence,)
        ):
            risk_free_value: float | tuple[float, ...] = _require_finite_number(
                risk_free, "MetricsBlock.risk_free"
            )
        else:
            series = tuple(risk_free)
            if not series:
                raise ResultStructureError(
                    "MetricsBlock.risk_free must be a finite scalar or a "
                    "nonempty sequence of finite floats; got an empty "
                    "sequence."
                )
            risk_free_value = tuple(
                _require_finite_number(item, "MetricsBlock.risk_free")
                for item in series
            )
        for name in _METRIC_NAMES:
            _require_finite_number(getattr(summary, name), f"MetricsBlock.{name}")
        object.__setattr__(
            self, "cumulative_return", float(summary.cumulative_return) + 0.0
        )
        object.__setattr__(
            self, "annualized_mean_return", float(summary.annualized_mean_return) + 0.0
        )
        object.__setattr__(
            self, "annualized_volatility", float(summary.annualized_volatility) + 0.0
        )
        object.__setattr__(self, "sharpe_ratio", float(summary.sharpe_ratio) + 0.0)
        object.__setattr__(self, "max_drawdown", float(summary.max_drawdown) + 0.0)
        object.__setattr__(
            self,
            "certainty_equivalent_return",
            float(summary.certainty_equivalent_return) + 0.0,
        )
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "periods_per_year", periods_per_year)
        object.__setattr__(self, "gamma", gamma_value)
        object.__setattr__(self, "risk_free", risk_free_value)


def _block_wire(block: MetricsBlock) -> dict[str, Any]:
    """The ten-key wire image of one metric block."""
    wire: dict[str, Any] = {
        "basis": block.basis,
        "periods_per_year": block.periods_per_year,
        "gamma": block.gamma,
    }
    for name in _METRIC_NAMES:
        wire[name] = getattr(block, name)
    risk_free = block.risk_free
    wire["risk_free"] = list(risk_free) if isinstance(risk_free, tuple) else risk_free
    return wire


_BLOCK_KEYS = frozenset(
    {
        "basis",
        "periods_per_year",
        "gamma",
        "risk_free",
        *_METRIC_NAMES,
    }
)


def _block_from_wire(node: Any) -> MetricsBlock:
    """Validate one parsed block through the constructor's own rules."""
    _require_exact_keys(node, _BLOCK_KEYS, "a metrics block")
    summary = MetricsSummary(
        cumulative_return=node["cumulative_return"],
        annualized_mean_return=node["annualized_mean_return"],
        annualized_volatility=node["annualized_volatility"],
        sharpe_ratio=node["sharpe_ratio"],
        max_drawdown=node["max_drawdown"],
        certainty_equivalent_return=node["certainty_equivalent_return"],
    )
    risk_free_node = node["risk_free"]
    if isinstance(risk_free_node, list):
        risk_free: float | Sequence[float] = tuple(risk_free_node)
    else:
        risk_free = risk_free_node
    return MetricsBlock(
        summary,
        basis=node["basis"],
        periods_per_year=node["periods_per_year"],
        gamma=node["gamma"],
        risk_free=risk_free,
    )


# --------------------------------------------------------------------------- #
# D7 — InformationSetAnchor                                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, init=False)
class InformationSetAnchor:
    """The digest link to one admitted information set.

    Exactly two fields, both derived from the consumed
    ``InformationSet``: ``as_of`` (the set's own decision instant)
    and ``membership_digest`` — sha256 over the canonical JSON array
    of the admitted records' sorted identity triples
    ``(series_id, observation_time, available_time)``. The records
    themselves are never embedded; submission order never matters
    (the triple array is sorted). No public constructor accepts
    either field directly.
    """

    as_of: datetime
    membership_digest: str

    def __init__(
        self,
        information_set: InformationSet,
        *,
        as_of: datetime | None = None,
        membership_digest: str | None = None,
        **unknown: Any,
    ) -> None:
        """Derive both fields from the consumed information set."""
        if unknown:
            raise TypeError(
                f"InformationSetAnchor got unexpected keyword "
                f"argument(s) {sorted(unknown)}: the anchor is exactly "
                "two fields."
            )
        if as_of is not None or membership_digest is not None:
            raise ResultStructureError(
                "InformationSetAnchor.as_of and .membership_digest are "
                "derived from the consumed InformationSet — never "
                "accepted from a caller."
            )
        if not isinstance(information_set, InformationSet):
            raise ResultStructureError(
                "InformationSetAnchor consumes an InformationSet; got "
                f"{type(information_set).__name__}."
            )
        moment = _require_aware(information_set.as_of, "InformationSetAnchor.as_of")
        triples = [
            _dumps(
                [
                    record.series_id,
                    record.observation_time.isoformat(),
                    record.available_time.isoformat(),
                ]
            )
            for record in information_set
        ]
        payload = "[" + ",".join(sorted(triples)) + "]"
        digest = sha256(payload.encode("utf-8")).hexdigest()
        object.__setattr__(self, "as_of", moment)
        object.__setattr__(self, "membership_digest", digest)


_ANCHOR_KEYS = frozenset({"as_of", "membership_digest"})


def _anchor_wire(anchor: InformationSetAnchor) -> dict[str, Any]:
    """The two-key wire image of one anchor."""
    return {
        "as_of": anchor.as_of.isoformat(),
        "membership_digest": anchor.membership_digest,
    }


def _anchor_from_wire(node: Any) -> InformationSetAnchor:
    """Validate one parsed anchor in wire form (never re-derived)."""
    _require_exact_keys(node, _ANCHOR_KEYS, "an information-set anchor")
    anchor = InformationSetAnchor.__new__(InformationSetAnchor)
    object.__setattr__(anchor, "as_of", _parse_instant(node["as_of"], "anchor as_of"))
    object.__setattr__(
        anchor,
        "membership_digest",
        _require_nonblank_str(node["membership_digest"], "anchor membership_digest"),
    )
    return anchor


# --------------------------------------------------------------------------- #
# D5 — ForecastLink                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ForecastLink:
    """The schema-owned image of one fit's provenance.

    Exactly ten fields: the ``ForecastProvenance`` image (``model``,
    ``config_hash``, ``cutoff``, ``seed``, ``determinism``,
    ``fit_window_start``) plus the forecast-declaration fields
    (``refit_instants``, ``forecast_target``, ``forecast_horizon``,
    ``information``). Every instant is aware (naive rejects with the
    timing module's ``NaiveTimestampError``); the horizon is a
    strictly positive ``timedelta`` carrying the ``str(timedelta)``
    wire form; ``information`` is a possibly-empty sequence of
    ``InformationSetAnchor`` values. The seed lives here and only
    here on the record.
    """

    model: str
    config_hash: str
    cutoff: datetime
    seed: int | None
    determinism: DeterminismClass
    fit_window_start: datetime
    refit_instants: tuple[datetime, ...]
    forecast_target: str
    forecast_horizon: timedelta
    information: tuple[InformationSetAnchor, ...]

    def __post_init__(self) -> None:
        """Validate the ten-field image fail-closed."""
        _require_nonblank_str(self.model, "ForecastLink.model")
        _require_nonblank_str(self.config_hash, "ForecastLink.config_hash")
        _require_nonblank_str(self.forecast_target, "ForecastLink.forecast_target")
        _require_aware(self.cutoff, "ForecastLink.cutoff")
        _require_aware(self.fit_window_start, "ForecastLink.fit_window_start")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ResultStructureError(
                f"ForecastLink.seed must be an int or None; got {self.seed!r}."
            )
        if not isinstance(self.determinism, DeterminismClass):
            raise ResultStructureError(
                "ForecastLink.determinism must be a DeterminismClass "
                f"member; got {self.determinism!r}."
            )
        raw_refits = self.refit_instants
        if isinstance(raw_refits, (str, bytes)) or not isinstance(raw_refits, Sequence):
            raise ResultStructureError(
                "ForecastLink.refit_instants must be a sequence of aware "
                f"instants; got {raw_refits!r}."
            )
        refits = tuple(
            _require_aware(instant, "ForecastLink.refit_instants")
            for instant in raw_refits
        )
        raw_information = self.information
        if isinstance(raw_information, (str, bytes)) or not isinstance(
            raw_information, Sequence
        ):
            raise ResultStructureError(
                "ForecastLink.information must be a sequence of "
                f"InformationSetAnchor values; got {raw_information!r}."
            )
        for anchor in raw_information:
            if not isinstance(anchor, InformationSetAnchor):
                raise ResultStructureError(
                    "ForecastLink.information carries a non-anchor element "
                    f"{anchor!r}: every element is an InformationSetAnchor."
                )
        horizon = _require_positive_horizon(
            self.forecast_horizon, "ForecastLink.forecast_horizon"
        )
        object.__setattr__(self, "refit_instants", refits)
        object.__setattr__(self, "information", tuple(raw_information))
        object.__setattr__(self, "forecast_horizon", horizon)


_LINK_KEYS = frozenset(
    {
        "model",
        "config_hash",
        "cutoff",
        "seed",
        "determinism",
        "fit_window_start",
        "refit_instants",
        "forecast_target",
        "forecast_horizon",
        "information",
    }
)


def _link_wire(link: ForecastLink) -> dict[str, Any]:
    """The ten-key wire image of one forecast link."""
    return {
        "model": link.model,
        "config_hash": link.config_hash,
        "cutoff": link.cutoff.isoformat(),
        "seed": link.seed,
        "determinism": link.determinism.value,
        "fit_window_start": link.fit_window_start.isoformat(),
        "refit_instants": [instant.isoformat() for instant in link.refit_instants],
        "forecast_target": link.forecast_target,
        "forecast_horizon": str(link.forecast_horizon),
        "information": [_anchor_wire(anchor) for anchor in link.information],
    }


def _link_from_wire(node: Any) -> ForecastLink:
    """Validate one parsed link through the constructor's own rules."""
    _require_exact_keys(node, _LINK_KEYS, "a forecast link")
    determinism_token = node["determinism"]
    if not isinstance(determinism_token, str):
        raise ResultStructureError(
            "a forecast link determinism must be the declared class's "
            f".value text; got {determinism_token!r}."
        )
    try:
        determinism = DeterminismClass(determinism_token)
    except ValueError:
        raise ResultStructureError(
            "a forecast link determinism token names no declared class: "
            f"{determinism_token!r}."
        ) from None
    refit_nodes = _require_wire_sequence(
        node["refit_instants"], "a forecast link refit_instants"
    )
    information_nodes = _require_wire_sequence(
        node["information"], "a forecast link information"
    )
    return ForecastLink(
        model=node["model"],
        config_hash=node["config_hash"],
        cutoff=_parse_instant(node["cutoff"], "a forecast link cutoff"),
        seed=node["seed"],
        determinism=determinism,
        fit_window_start=_parse_instant(
            node["fit_window_start"], "a forecast link fit_window_start"
        ),
        refit_instants=tuple(
            _parse_instant(instant, "a forecast link refit_instant")
            for instant in refit_nodes
        ),
        forecast_target=node["forecast_target"],
        forecast_horizon=_horizon_from_wire(
            node["forecast_horizon"], "a forecast link forecast_horizon"
        ),
        information=tuple(_anchor_from_wire(anchor) for anchor in information_nodes),
    )


# --------------------------------------------------------------------------- #
# D3 — PathResult                                                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, init=False)
class PathResult:
    """The digest-plus-structural-summary link to one portfolio path.

    Exactly six fields, all derived from the consumed ``LedgerPath``
    and its metric sequence: ``ledger_digest``, ``n_periods``,
    ``period_start``/``period_end`` (the first/last row bounds), and
    ``final_wealth`` from the path itself, plus ``metrics`` — the
    canonically ordered, duplicate-free sequence of
    ``MetricsBlock`` values. The rows never ride on the record; the
    digest is the link. No public constructor accepts a derived
    field, and an empty path or empty metric sequence rejects.
    """

    ledger_digest: str
    n_periods: int
    period_start: datetime
    period_end: datetime
    final_wealth: float
    metrics: tuple[MetricsBlock, ...]

    def __init__(
        self,
        path: LedgerPath,
        metrics: Sequence[MetricsBlock],
        *,
        ledger_digest: str | None = None,
        n_periods: int | None = None,
        period_start: datetime | None = None,
        period_end: datetime | None = None,
        final_wealth: float | None = None,
        **unknown: Any,
    ) -> None:
        """Derive the digest and structural summary from the path."""
        if unknown:
            raise TypeError(
                f"PathResult got unexpected keyword argument(s) "
                f"{sorted(unknown)}: the path result is exactly six fields."
            )
        for field, supplied in (
            ("ledger_digest", ledger_digest),
            ("n_periods", n_periods),
            ("period_start", period_start),
            ("period_end", period_end),
            ("final_wealth", final_wealth),
        ):
            if supplied is not None:
                raise ResultStructureError(
                    f"PathResult.{field} is derived from the consumed "
                    "LedgerPath — never accepted from a caller."
                )
        if not isinstance(path, LedgerPath):
            raise ResultStructureError(
                f"PathResult consumes a LedgerPath; got {type(path).__name__}."
            )
        if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence):
            raise ResultStructureError(
                "PathResult.metrics must be a sequence of MetricsBlock "
                f"values; got {metrics!r}."
            )
        blocks = tuple(metrics)
        if not blocks:
            raise ResultStructureError(
                "PathResult.metrics must carry at least one MetricsBlock: "
                "a path result without its metric sequence is not a result."
            )
        for block in blocks:
            if not isinstance(block, MetricsBlock):
                raise ResultStructureError(
                    "PathResult.metrics carries a non-block element "
                    f"{block!r}: every element is a MetricsBlock."
                )
        rows = path.rows
        if not rows:
            raise ResultStructureError(
                "PathResult consumes a nonempty LedgerPath: a zero-row "
                "path carries no wealth path to record."
            )
        for row in rows:
            _require_aware(row.period_start, "row period_start")
            _require_aware(row.period_end, "row period_end")
        wealth = _require_finite_number(path.final_wealth, "PathResult.final_wealth")
        ordered_blocks = _ordered_and_unique(blocks, _block_wire, "PathResult.metrics")
        object.__setattr__(self, "ledger_digest", _rows_digest(path))
        object.__setattr__(self, "n_periods", path.n_periods)
        object.__setattr__(self, "period_start", rows[0].period_start)
        object.__setattr__(self, "period_end", rows[-1].period_end)
        object.__setattr__(self, "final_wealth", wealth)
        object.__setattr__(self, "metrics", ordered_blocks)


_PATH_KEYS = frozenset(
    {
        "ledger_digest",
        "n_periods",
        "period_start",
        "period_end",
        "final_wealth",
        "metrics",
    }
)


def _path_wire(path_result: PathResult) -> dict[str, Any]:
    """The six-key wire image of one path result."""
    return {
        "ledger_digest": path_result.ledger_digest,
        "n_periods": path_result.n_periods,
        "period_start": path_result.period_start.isoformat(),
        "period_end": path_result.period_end.isoformat(),
        "final_wealth": path_result.final_wealth,
        "metrics": [_block_wire(block) for block in path_result.metrics],
    }


def _path_from_wire(node: Any) -> PathResult:
    """Validate one parsed path result in wire form (never re-derived)."""
    _require_exact_keys(node, _PATH_KEYS, "a path result")
    digest = _require_nonblank_str(node["ledger_digest"], "path result ledger_digest")
    n_periods = node["n_periods"]
    if isinstance(n_periods, bool) or not isinstance(n_periods, int) or n_periods <= 0:
        raise ResultStructureError(
            f"path result n_periods must be a strictly positive int; got {n_periods!r}."
        )
    block_nodes = _require_wire_sequence(node["metrics"], "path metrics")
    if not block_nodes:
        raise ResultStructureError("path result metrics must carry at least one block.")
    blocks = [_block_from_wire(block_node) for block_node in block_nodes]
    _require_wire_order([_block_wire(block) for block in blocks], "path result metrics")
    path_result = PathResult.__new__(PathResult)
    object.__setattr__(path_result, "ledger_digest", digest)
    object.__setattr__(path_result, "n_periods", n_periods)
    object.__setattr__(
        path_result,
        "period_start",
        _parse_instant(node["period_start"], "path period_start"),
    )
    object.__setattr__(
        path_result,
        "period_end",
        _parse_instant(node["period_end"], "path period_end"),
    )
    object.__setattr__(
        path_result,
        "final_wealth",
        _require_finite_number(node["final_wealth"], "path final_wealth"),
    )
    object.__setattr__(path_result, "metrics", tuple(blocks))
    return path_result


# --------------------------------------------------------------------------- #
# D6 — SanctionedDeviation and ConformanceRecord                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SanctionedDeviation:
    """One sanctioned declared/effective difference.

    Exactly five fields, all required and all complete: the differing
    top-level settings ``field`` name, the sanctioning ``authority``,
    a traceable ``reference``, the ``reason``, and the aware
    ``approved_at`` instant. Blank strings reject with
    ``ResultStructureError``; naive approval instants reject with the
    timing module's ``NaiveTimestampError``.
    """

    field: str
    authority: str
    reference: str
    reason: str
    approved_at: datetime

    def __post_init__(self) -> None:
        """Validate the five fields are complete."""
        _require_nonblank_str(self.field, "SanctionedDeviation.field")
        _require_nonblank_str(self.authority, "SanctionedDeviation.authority")
        _require_nonblank_str(self.reference, "SanctionedDeviation.reference")
        _require_nonblank_str(self.reason, "SanctionedDeviation.reason")
        _require_aware(self.approved_at, "SanctionedDeviation.approved_at")


_DEVIATION_KEYS = frozenset(
    {"field", "authority", "reference", "reason", "approved_at"}
)


def _deviation_wire(deviation: SanctionedDeviation) -> dict[str, Any]:
    """The five-key wire image of one sanction."""
    return {
        "field": deviation.field,
        "authority": deviation.authority,
        "reference": deviation.reference,
        "reason": deviation.reason,
        "approved_at": deviation.approved_at.isoformat(),
    }


@dataclass(frozen=True, init=False)
class ConformanceRecord:
    """The computed declared/effective conformance verdict.

    Exactly four fields: ``declared`` and ``effective`` — deep
    immutable snapshots under the recursive settings grammar
    (``None``, ``bool``, ``int``, finite ``float`` with signed zero
    unified, ``str``, tuples, and mappings with non-blank string
    keys; insertion order is never content) — plus the canonically
    ordered ``deviations`` and the computed ``verdict``. The verdict
    is derived at construction and recomputed at parse; no
    constructor accepts it.
    """

    declared: Mapping[str, Any]
    effective: Mapping[str, Any]
    deviations: tuple[SanctionedDeviation, ...]
    verdict: str

    def __init__(
        self,
        *,
        declared: Mapping[str, Any],
        effective: Mapping[str, Any],
        deviations: Sequence[SanctionedDeviation] = (),
        **unknown: Any,
    ) -> None:
        """Snapshot both sides and derive the verdict fail-closed."""
        if unknown:
            raise TypeError(
                f"ConformanceRecord got unexpected keyword argument(s) "
                f"{sorted(unknown)}: the verdict is computed from the "
                "declared/effective pair, never accepted."
            )
        frozen_declared = _freeze_settings(declared, "ConformanceRecord.declared")
        frozen_effective = _freeze_settings(effective, "ConformanceRecord.effective")
        for side, frozen in (
            ("declared", frozen_declared),
            ("effective", frozen_effective),
        ):
            if not isinstance(frozen, Mapping):
                raise ResultStructureError(
                    f"ConformanceRecord.{side} must be a mapping of "
                    f"settings; got {frozen!r}."
                )
        if isinstance(deviations, (str, bytes)) or not isinstance(deviations, Sequence):
            raise ResultStructureError(
                "ConformanceRecord.deviations must be a sequence of "
                f"SanctionedDeviation values; got {deviations!r}."
            )
        sanctioned = tuple(deviations)
        for deviation in sanctioned:
            if not isinstance(deviation, SanctionedDeviation):
                raise ResultStructureError(
                    "ConformanceRecord.deviations carries a non-sanction "
                    f"element {deviation!r}: every element is a "
                    "SanctionedDeviation."
                )
        verdict = _derive_verdict(frozen_declared, frozen_effective, sanctioned)
        ordered = _ordered_and_unique(
            sanctioned, _deviation_wire, "ConformanceRecord.deviations"
        )
        object.__setattr__(self, "declared", frozen_declared)
        object.__setattr__(self, "effective", frozen_effective)
        object.__setattr__(self, "deviations", ordered)
        object.__setattr__(self, "verdict", verdict)


_CONFORMANCE_KEYS = frozenset({"declared", "effective", "deviations", "verdict"})


def _conformance_wire(conformance: ConformanceRecord) -> dict[str, Any]:
    """The four-key wire image of the conformance record."""
    return {
        "declared": _settings_wire(conformance.declared),
        "effective": _settings_wire(conformance.effective),
        "deviations": [
            _deviation_wire(deviation) for deviation in conformance.deviations
        ],
        "verdict": conformance.verdict,
    }


def _settings_from_wire(node: Any, what: str) -> Mapping[str, Any]:
    """Validate one parsed settings tree under the settings grammar."""
    if not isinstance(node, dict):
        raise ResultStructureError(
            f"{what} must be a JSON object of settings; got {type(node).__name__}."
        )
    frozen = _freeze_settings(node, what)
    if not isinstance(frozen, Mapping):
        raise ResultStructureError(f"{what} must be a JSON object of settings.")
    return frozen


def _conformance_from_wire(node: Any) -> ConformanceRecord:
    """Validate one parsed conformance record; recompute the verdict."""
    _require_exact_keys(node, _CONFORMANCE_KEYS, "the conformance record")
    declared = _settings_from_wire(node["declared"], "declared settings")
    effective = _settings_from_wire(node["effective"], "effective settings")
    deviation_nodes = _require_wire_sequence(
        node["deviations"], "conformance deviations"
    )
    deviations = tuple(
        SanctionedDeviation(
            field=deviation["field"],
            authority=deviation["authority"],
            reference=deviation["reference"],
            reason=deviation["reason"],
            approved_at=_parse_instant(
                deviation["approved_at"], "sanction approved_at"
            ),
        )
        for deviation in deviation_nodes
    )
    _require_wire_order(
        [_deviation_wire(deviation) for deviation in deviations],
        "conformance deviations",
    )
    built = ConformanceRecord(
        declared=declared, effective=effective, deviations=deviations
    )
    stored_verdict = node["verdict"]
    if stored_verdict != built.verdict:
        raise ResultStructureError(
            f"the stored verdict {stored_verdict!r} disagrees with the "
            f"verdict recomputed from the stored declared/effective pair "
            f"({built.verdict!r}): the verdict is computed, never trusted."
        )
    return built


# --------------------------------------------------------------------------- #
# D8 — ParameterRelation                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, init=False)
class ParameterRelation:
    """One recorded γ/λ identity-or-difference choice.

    Exactly five fields: ``role_a``/``role_b`` (non-blank role
    names), ``value_a``/``value_b`` (finite floats, signed zero
    unified), and the computed ``relation`` — ``"identical"`` iff
    the two values are equal after the settings grammar's signed-zero
    normalization, else ``"distinct"``. The token is never an input.
    """

    role_a: str
    role_b: str
    value_a: float
    value_b: float
    relation: str

    def __init__(
        self,
        *,
        role_a: str,
        role_b: str,
        value_a: float,
        value_b: float,
        relation: Any = _UNSET,
        **unknown: Any,
    ) -> None:
        """Validate the two roles and values; compute the token."""
        if unknown:
            raise TypeError(
                f"ParameterRelation got unexpected keyword argument(s) "
                f"{sorted(unknown)}: the relation is exactly five fields."
            )
        if relation is not _UNSET:
            raise TypeError(
                "ParameterRelation.relation is computed from value_a and "
                "value_b (identical iff equal after signed-zero "
                "normalization, else distinct) — never an input."
            )
        role_a_value = _require_nonblank_str(role_a, "ParameterRelation.role_a")
        role_b_value = _require_nonblank_str(role_b, "ParameterRelation.role_b")
        value_a_number = _require_finite_number(value_a, "ParameterRelation.value_a")
        value_b_number = _require_finite_number(value_b, "ParameterRelation.value_b")
        token = "identical" if value_a_number == value_b_number else "distinct"
        object.__setattr__(self, "role_a", role_a_value)
        object.__setattr__(self, "role_b", role_b_value)
        object.__setattr__(self, "value_a", value_a_number)
        object.__setattr__(self, "value_b", value_b_number)
        object.__setattr__(self, "relation", token)


_RELATION_KEYS = frozenset({"role_a", "role_b", "value_a", "value_b", "relation"})


def _relation_wire(relation: ParameterRelation) -> dict[str, Any]:
    """The five-key wire image of one parameter relation."""
    return {
        "role_a": relation.role_a,
        "role_b": relation.role_b,
        "value_a": relation.value_a,
        "value_b": relation.value_b,
        "relation": relation.relation,
    }


def _relation_from_wire(node: Any) -> ParameterRelation:
    """Validate one parsed relation; the token must agree with its values."""
    _require_exact_keys(node, _RELATION_KEYS, "a parameter relation")
    built = ParameterRelation(
        role_a=node["role_a"],
        role_b=node["role_b"],
        value_a=node["value_a"],
        value_b=node["value_b"],
    )
    stored_token = node["relation"]
    if stored_token != built.relation:
        raise ResultStructureError(
            f"a parameter relation stores the token {stored_token!r} while "
            f"its own values recompute {built.relation!r}: the identity-"
            "or-difference token is computed, never trusted."
        )
    return built


def _require_relation_coverage(
    paths: Mapping[str, PathResult],
    conformance: ConformanceRecord,
    relations: tuple[ParameterRelation, ...],
) -> None:
    """Enforce the γ/λ co-occurrence rule.

    The decision risk aversion is looked up only as the top-level
    ``effective["risk_aversion"]`` key — never a nested key, never
    from ``declared`` alone. When present, every distinct pairing of
    a ``MetricsBlock.gamma`` with that value requires one covering
    ``ParameterRelation`` (roles ``evaluation_gamma`` and
    ``decision_risk_aversion``, values equal after signed-zero
    normalization); any missing pairing rejects construction.
    """
    effective = conformance.effective
    if "risk_aversion" not in effective:
        return
    risk_aversion = _require_finite_number(
        effective["risk_aversion"], "effective settings risk_aversion"
    )
    gammas = {
        _require_finite_number(block.gamma, "MetricsBlock.gamma")
        for path_result in paths.values()
        for block in path_result.metrics
    }
    covered = {
        (relation.value_a, relation.value_b)
        for relation in relations
        if relation.role_a == _GAMMA_ROLE and relation.role_b == _RISK_AVERSION_ROLE
    }
    missing = sorted(gamma for gamma in gammas if (gamma, risk_aversion) not in covered)
    if missing:
        raise ResultStructureError(
            "the γ/λ co-occurrence rule requires one ParameterRelation "
            f"(roles {_GAMMA_ROLE!r}/{_RISK_AVERSION_ROLE!r}) for every "
            f"distinct (gamma, effective risk_aversion) pairing; missing "
            f"coverage for gamma values {missing} against the effective "
            f"risk_aversion {risk_aversion!r}."
        )


# --------------------------------------------------------------------------- #
# D2 — ResultRecord                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, init=False)
class ResultRecord:
    """One complete result record: the run link, paths, and provenance.

    Exactly nine fields: ``schema_version`` (the ``"1.0.0"`` literal
    — the only version field), ``strategy_id``, ``created_at``,
    ``run_id`` and ``manifest_digest`` (both derived from the
    consumed ``RunManifest`` — the manifest itself is never
    embedded), ``paths`` (named ``PathResult`` values), ``forecasts``
    (a possibly-empty, canonically ordered sequence of
    ``ForecastLink``), ``conformance`` (the computed verdict), and
    ``parameter_relations``. The record is immutable and
    deliberately unhashable; equality is canonical-content
    equality. ``to_json``/``from_json`` are the only serialization
    surface — no persistence, comparison, or archive API exists
    here.
    """

    schema_version: str
    strategy_id: str
    created_at: datetime
    run_id: str
    manifest_digest: str
    paths: Mapping[str, PathResult]
    forecasts: tuple[ForecastLink, ...]
    conformance: ConformanceRecord
    parameter_relations: tuple[ParameterRelation, ...]

    # §5.9: hash is deliberately not provided — the record is
    # unhashable by design and no consumer may rely on hashing it.
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        manifest: RunManifest,
        *,
        strategy_id: str,
        paths: Mapping[str, PathResult],
        forecasts: Sequence[ForecastLink],
        conformance: ConformanceRecord,
        parameter_relations: Sequence[ParameterRelation] = (),
        created_at: datetime,
        schema_version: str | None = None,
        run_id: str | None = None,
        manifest_digest: str | None = None,
        **unknown: Any,
    ) -> None:
        """Consume the manifest and derive the run identity from it."""
        if unknown:
            raise TypeError(
                f"ResultRecord got unexpected keyword argument(s) "
                f"{sorted(unknown)}: the record is exactly nine fields."
            )
        if schema_version is not None and schema_version != _SCHEMA_VERSION:
            raise ResultStructureError(
                f"ResultRecord.schema_version is the {_SCHEMA_VERSION!r} "
                f"literal of this spec; got {schema_version!r}."
            )
        if run_id is not None:
            raise ResultStructureError(
                "ResultRecord.run_id is derived from the consumed "
                "RunManifest — never accepted from a caller."
            )
        if manifest_digest is not None:
            raise ResultStructureError(
                "ResultRecord.manifest_digest is derived from the consumed "
                "RunManifest — never accepted from a caller."
            )
        if not isinstance(manifest, RunManifest):
            raise ResultStructureError(
                f"ResultRecord consumes a RunManifest; got {type(manifest).__name__}."
            )
        strategy = _require_nonblank_str(strategy_id, "ResultRecord.strategy_id")
        moment = _require_aware(created_at, "ResultRecord.created_at")
        if not isinstance(paths, Mapping):
            raise ResultStructureError(
                f"ResultRecord.paths must be a mapping of named PathResult "
                f"values; got {paths!r}."
            )
        if not paths:
            raise ResultStructureError(
                "ResultRecord.paths must name at least one path: a "
                "zero-path record records no result."
            )
        named: dict[str, PathResult] = {}
        for name, path_result in paths.items():
            _require_nonblank_str(name, "a ResultRecord path name")
            if not isinstance(path_result, PathResult):
                raise ResultStructureError(
                    f"the path named {name!r} is not a PathResult: got {path_result!r}."
                )
            named[name] = path_result
        if isinstance(forecasts, (str, bytes)) or not isinstance(forecasts, Sequence):
            raise ResultStructureError(
                "ResultRecord.forecasts must be a sequence of ForecastLink "
                f"values (possibly empty); got {forecasts!r}."
            )
        for link in forecasts:
            if not isinstance(link, ForecastLink):
                raise ResultStructureError(
                    f"ResultRecord.forecasts carries a non-link element "
                    f"{link!r}: every element is a ForecastLink."
                )
        ordered_forecasts = _ordered_and_unique(
            tuple(forecasts), _link_wire, "ResultRecord.forecasts"
        )
        if not isinstance(conformance, ConformanceRecord):
            raise ResultStructureError(
                "ResultRecord.conformance must be a ConformanceRecord; got "
                f"{conformance!r}."
            )
        if isinstance(parameter_relations, (str, bytes)) or not isinstance(
            parameter_relations, Sequence
        ):
            raise ResultStructureError(
                "ResultRecord.parameter_relations must be a sequence of "
                f"ParameterRelation values; got {parameter_relations!r}."
            )
        for relation in parameter_relations:
            if not isinstance(relation, ParameterRelation):
                raise ResultStructureError(
                    "ResultRecord.parameter_relations carries a non-relation "
                    f"element {relation!r}: every element is a "
                    "ParameterRelation."
                )
        ordered_relations = _ordered_and_unique(
            tuple(parameter_relations),
            _relation_wire,
            "ResultRecord.parameter_relations",
        )
        _require_relation_coverage(
            MappingProxyType(named), conformance, ordered_relations
        )
        object.__setattr__(self, "schema_version", _SCHEMA_VERSION)
        object.__setattr__(self, "strategy_id", strategy)
        object.__setattr__(self, "created_at", moment)
        object.__setattr__(self, "run_id", manifest.run_id)
        object.__setattr__(
            self,
            "manifest_digest",
            sha256(manifest.to_json().encode("utf-8")).hexdigest(),
        )
        object.__setattr__(self, "paths", MappingProxyType(named))
        object.__setattr__(self, "forecasts", ordered_forecasts)
        object.__setattr__(self, "conformance", conformance)
        object.__setattr__(self, "parameter_relations", ordered_relations)

    def to_json(self) -> str:
        """Serialize the record to its canonical JSON form.

        Sorted keys at every level, compact separators, ASCII-only,
        and fail-closed on nonfinite float content. The run link is
        the flat ``run_id``/``manifest_digest`` pair — the manifest
        itself is never embedded — and the seed appears only inside
        forecast links.
        """
        return _dumps(
            {
                "schema_version": self.schema_version,
                "strategy_id": self.strategy_id,
                "created_at": self.created_at.isoformat(),
                "run_id": self.run_id,
                "manifest_digest": self.manifest_digest,
                "paths": {
                    name: _path_wire(path_result)
                    for name, path_result in self.paths.items()
                },
                "forecasts": [_link_wire(link) for link in self.forecasts],
                "conformance": _conformance_wire(self.conformance),
                "parameter_relations": [
                    _relation_wire(relation) for relation in self.parameter_relations
                ],
            }
        )

    @classmethod
    def from_json(cls, text: str) -> ResultRecord:
        """Parse canonical JSON back to a record, fail-closed.

        Duplicate JSON object keys reject at every nesting depth;
        nonfinite float literals reject; unknown versions reject;
        naive instants reject with the timing module's error; every
        fixed-shape object carries an exact key set; every sequence
        must arrive in canonical order; the verdict and every
        relation token are recomputed from stored content. The
        linked artifacts (manifest, ledger paths, information sets)
        are not on the wire, so their derived values are validated
        in wire form, never re-derived.
        """
        if not isinstance(text, str):
            raise ResultStructureError(
                f"ResultRecord.from_json consumes JSON text; got {type(text).__name__}."
            )
        try:
            tree = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_constant,
            )
        except json.JSONDecodeError as unparsable:
            raise ResultStructureError(
                f"ResultRecord.from_json requires well-formed JSON text; "
                f"got {unparsable.msg} at offset {unparsable.pos}."
            ) from None
        return _record_from_wire(tree)


_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "strategy_id",
        "created_at",
        "run_id",
        "manifest_digest",
        "paths",
        "forecasts",
        "conformance",
        "parameter_relations",
    }
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """The object-pairs-hook rule: no last-write-wins anywhere."""
    node: dict[str, Any] = {}
    for key, value in pairs:
        if key in node:
            raise ResultStructureError(
                f"the JSON object carries the duplicate key {key!r}: "
                "duplicate keys reject at every nesting depth — no "
                "last-write-wins anywhere."
            )
        node[key] = value
    return node


def _reject_constant(name: str) -> float:
    """Reject ``NaN``/``Infinity`` literals on read."""
    raise ResultStructureError(
        f"the JSON text carries the nonfinite literal {name!r}: every "
        "float on the result surface is finite, fail-closed in both "
        "directions."
    )


def _parse_instant(node: Any, field: str) -> datetime:
    """Parse one ISO-8601 instant from the wire, aware-only."""
    if not isinstance(node, str):
        raise ResultStructureError(f"{field} must be ISO-8601 text; got {node!r}.")
    try:
        moment = datetime.fromisoformat(node)
    except ValueError:
        raise ResultStructureError(
            f"{field} must be ISO-8601 text; got {node!r}."
        ) from None
    # Naive text rejects with the timing module's own error — this
    # module owns no second timezone policy.
    _require_aware(moment, field)
    return moment


def _record_from_wire(tree: Any) -> ResultRecord:
    """Validate the parsed record tree and rebuild the record."""
    _require_exact_keys(tree, _RECORD_KEYS, "the result record")
    version = tree["schema_version"]
    if version != _SCHEMA_VERSION:
        raise ResultStructureError(
            f"the record carries the unknown schema_version {version!r}: "
            f"this spec reads exactly {_SCHEMA_VERSION!r}, with no silent "
            "forward-compatibility."
        )
    strategy = _require_nonblank_str(tree["strategy_id"], "record strategy_id")
    created = _parse_instant(tree["created_at"], "record created_at")
    run_id = _require_nonblank_str(tree["run_id"], "record run_id")
    manifest_digest = _require_nonblank_str(
        tree["manifest_digest"], "record manifest_digest"
    )
    paths_node = tree["paths"]
    if not isinstance(paths_node, dict):
        raise ResultStructureError(
            f"the record paths must be a JSON object; got {paths_node!r}."
        )
    if not paths_node:
        raise ResultStructureError(
            "the record names no paths: a zero-path record records no result."
        )
    named = {
        _require_nonblank_str(name, "a record path name"): _path_from_wire(path_node)
        for name, path_node in paths_node.items()
    }
    forecast_nodes = _require_wire_sequence(tree["forecasts"], "record forecasts")
    links = [_link_from_wire(link_node) for link_node in forecast_nodes]
    _require_wire_order([_link_wire(link) for link in links], "record forecasts")
    conformance = _conformance_from_wire(tree["conformance"])
    relation_nodes = _require_wire_sequence(
        tree["parameter_relations"], "record parameter_relations"
    )
    relations = [_relation_from_wire(relation_node) for relation_node in relation_nodes]
    _require_wire_order(
        [_relation_wire(relation) for relation in relations],
        "record parameter_relations",
    )
    frozen_paths = MappingProxyType(named)
    _require_relation_coverage(frozen_paths, conformance, tuple(relations))
    record = ResultRecord.__new__(ResultRecord)
    object.__setattr__(record, "schema_version", _SCHEMA_VERSION)
    object.__setattr__(record, "strategy_id", strategy)
    object.__setattr__(record, "created_at", created)
    object.__setattr__(record, "run_id", run_id)
    object.__setattr__(record, "manifest_digest", manifest_digest)
    object.__setattr__(record, "paths", frozen_paths)
    object.__setattr__(record, "forecasts", tuple(links))
    object.__setattr__(record, "conformance", conformance)
    object.__setattr__(record, "parameter_relations", tuple(relations))
    return record
