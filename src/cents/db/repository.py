"""Repository layer for CRUD operations with shared helpers."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Sequence, Type

logger = logging.getLogger(__name__)

from cents.models import (
    Alert,
    AlertType,
    Backtest,
    BacktestSignal,
    Event,
    EventPolarity,
    EventTagStatus,
    Delisting,
    Evidence,
    EvidenceType,
    Experiment,
    FactoryRun,
    ShadowOpen,
    LLMUsage,
    ThesisDimension,
    Universe,
    UniverseSource,
    Outcome,
    Position,
    PositionSide,
    PositionStatus,
    Thesis,
    ThesisAccuracy,
    ThesisCohort,
    ThesisStatus,
    Valuation,
    TimeHorizon,
    ThesisOutcome,
    WatchlistItem,
)
from cents.db.schema import get_connection


def _identity(value: Any) -> Any:
    return value


def _isoformat(value: date | datetime | None) -> str | None:
    return value.isoformat() if value else None


@dataclass
class ModelField:
    """Mapping metadata for a model attribute and database column."""

    attr: str
    column: str | None = None
    serialize: Callable[[Any], Any] = _identity
    deserialize: Callable[[Any], Any] = _identity
    update: bool = True

    def __post_init__(self) -> None:
        if self.column is None:
            self.column = self.attr


@dataclass
class ModelMeta:
    """Model/table metadata used by BaseRepository helpers."""

    table: str
    model: Type[Any]
    fields: list[ModelField]
    default_order: str | None = None


class BaseRepository:
    """Common helpers for repository classes."""

    def __init__(self, conn: sqlite3.Connection | None = None):
        self.conn = conn or get_connection()

    @staticmethod
    def dumps_json(value: Any) -> str:
        return json.dumps(value)

    @staticmethod
    def loads_json(raw: str | None, default: Any, context: str) -> Any:
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse %s JSON", context)
            return default

    def _row_to_model(self, meta: ModelMeta, row: sqlite3.Row):
        kwargs = {field.attr: field.deserialize(row[field.column]) for field in meta.fields}
        return meta.model(**kwargs)

    def _insert(self, meta: ModelMeta, model: Any, *, replace: bool = False) -> Any:
        columns = ", ".join(field.column for field in meta.fields)
        placeholders = ", ".join(["?"] * len(meta.fields))
        values = [field.serialize(getattr(model, field.attr)) for field in meta.fields]
        verb = "INSERT OR REPLACE" if replace else "INSERT"
        self.conn.execute(
            f"{verb} INTO {meta.table} ({columns}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()
        return model

    def _update(self, meta: ModelMeta, model: Any, key_attr: str = "id") -> Any:
        updatable = [field for field in meta.fields if field.update and field.attr != key_attr]
        assignments = ", ".join(f"{field.column} = ?" for field in updatable)
        values = [field.serialize(getattr(model, field.attr)) for field in updatable]
        values.append(getattr(model, key_attr))
        self.conn.execute(
            f"UPDATE {meta.table} SET {assignments} WHERE {key_attr} = ?",
            values,
        )
        self.conn.commit()
        return model

    def _get_by_id(self, meta: ModelMeta, item_id: str, key_column: str = "id"):
        row = self.conn.execute(
            f"SELECT * FROM {meta.table} WHERE {key_column} = ?",
            (item_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_model(meta, row)

    def _list(
        self,
        meta: ModelMeta,
        *,
        where: str | None = None,
        params: Sequence[Any] = (),
        order_by: str | None = None,
        limit: int | None = None,
    ) -> list[Any]:
        query = f"SELECT * FROM {meta.table}"
        if where:
            query += f" WHERE {where}"
        order = order_by or meta.default_order
        if order:
            query += f" ORDER BY {order}"
        if limit is not None:
            query += " LIMIT ?"
            params = [*params, limit]
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_model(meta, row) for row in rows]

    def _delete(self, meta: ModelMeta, where: str, params: Sequence[Any]) -> int:
        cursor = self.conn.execute(
            f"DELETE FROM {meta.table} WHERE {where}",
            params,
        )
        self.conn.commit()
        return cursor.rowcount


class ThesisRepository(BaseRepository):
    """CRUD operations for theses."""

    _META = ModelMeta(
        table="theses",
        model=Thesis,
        fields=[
            ModelField("id"),
            ModelField("title"),
            ModelField("hypothesis"),
            ModelField("status", serialize=lambda v: v.value, deserialize=ThesisStatus),
            ModelField("conviction"),
            ModelField("tags", serialize=BaseRepository.dumps_json, deserialize=lambda raw: BaseRepository.loads_json(raw, [], "tags")),
            ModelField("symbol"),
            ModelField("business_quality"),
            ModelField("valuation", serialize=lambda v: v.value if v else None, deserialize=lambda raw: Valuation(raw) if raw else None),
            ModelField("moat"),
            ModelField("time_horizon", serialize=lambda v: v.value if v else None, deserialize=lambda raw: TimeHorizon(raw) if raw else None),
            ModelField("horizon_end", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw) if raw else None),
            ModelField("key_risks", serialize=BaseRepository.dumps_json, deserialize=lambda raw: BaseRepository.loads_json(raw, [], "key_risks")),
            ModelField("target_price"),
            ModelField("stop_price"),
            ModelField("outcome", serialize=lambda v: v.value if v else None, deserialize=lambda raw: ThesisOutcome(raw) if raw else None),
            ModelField("closed_at", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw) if raw else None),
            ModelField("cohort", serialize=lambda v: v.value, deserialize=lambda raw: ThesisCohort(raw) if raw else ThesisCohort.DIRECTIONAL),
            ModelField("hedge_symbol", serialize=lambda v: v.upper() if v else None),
            ModelField("paired_thesis_id"),
            ModelField("premise_tags", serialize=BaseRepository.dumps_json, deserialize=lambda raw: BaseRepository.loads_json(raw, [], "premise_tags")),
            ModelField("premise_direction", serialize=BaseRepository.dumps_json, deserialize=lambda raw: BaseRepository.loads_json(raw, {}, "premise_direction")),
            ModelField("regime_snapshot", serialize=BaseRepository.dumps_json, deserialize=lambda raw: BaseRepository.loads_json(raw, {}, "regime_snapshot")),
            ModelField("discovery_source"),
            ModelField("calibrated_p_correct"),
            ModelField("calibration_fit_at"),
            ModelField("orchestrator_label"),
            ModelField("experiment_id"),
            ModelField("created_at", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw), update=False),
            ModelField("updated_at", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw)),
        ],
        default_order="updated_at DESC",
    )

    def create(self, thesis: Thesis) -> Thesis:
        """Insert a new thesis."""
        return self._insert(self._META, thesis)

    def get(self, thesis_id: str) -> Thesis | None:
        """Get thesis by ID."""
        return self._get_by_id(self._META, thesis_id)

    def list(self, status: ThesisStatus | None = None) -> list[Thesis]:
        """List theses, optionally filtered by status."""
        if status:
            return self._list(self._META, where="status = ?", params=(status.value,))
        return self._list(self._META)

    def update(self, thesis: Thesis) -> Thesis:
        """Update an existing thesis."""
        thesis.updated_at = datetime.now()
        return self._update(self._META, thesis)

    def delete(self, thesis_id: str) -> bool:
        """Delete a thesis by ID."""
        return self._delete(self._META, "id = ?", (thesis_id,)) > 0


class PositionRepository(BaseRepository):
    """CRUD operations for positions."""

    _META = ModelMeta(
        table="positions",
        model=Position,
        fields=[
            ModelField("id"),
            ModelField("thesis_id"),
            ModelField("symbol"),
            ModelField("side", serialize=lambda v: v.value, deserialize=PositionSide),
            ModelField("entry_price"),
            ModelField("entry_date", serialize=_isoformat, deserialize=lambda raw: date.fromisoformat(raw)),
            ModelField("size"),
            ModelField("status", serialize=lambda v: v.value, deserialize=PositionStatus),
            ModelField("exit_price"),
            ModelField("exit_date", serialize=_isoformat, deserialize=lambda raw: date.fromisoformat(raw) if raw else None),
            ModelField("paper", serialize=lambda v: 1 if v else 0, deserialize=lambda raw: bool(raw)),
            ModelField("notes"),
            ModelField("created_at", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw), update=False),
            # Cost-aware accounting (v0.10) — see cents/models/position.py.
            ModelField("costs_applied_usd", serialize=lambda v: v or 0.0, deserialize=lambda raw: float(raw) if raw is not None else 0.0),
            ModelField("realized_exit_price"),
            ModelField("sizing_method"),
            ModelField("borrow_rate_pa_pct"),
        ],
        default_order="created_at DESC",
    )

    def create(self, position: Position) -> Position:
        """Insert a new position."""
        return self._insert(self._META, position)

    def get(self, position_id: str) -> Position | None:
        """Get position by ID."""
        return self._get_by_id(self._META, position_id)

    def list(self, status: PositionStatus | None = None) -> list[Position]:
        """List positions, optionally filtered by status."""
        if status:
            return self._list(self._META, where="status = ?", params=(status.value,))
        return self._list(self._META)

    def update(self, position: Position) -> Position:
        """Update an existing position."""
        return self._update(self._META, position)

    def delete(self, position_id: str) -> bool:
        """Delete a position by ID."""
        return self._delete(self._META, "id = ?", (position_id,)) > 0


class EvidenceRepository(BaseRepository):
    """CRUD operations for evidence."""

    _META = ModelMeta(
        table="evidence",
        model=Evidence,
        fields=[
            ModelField("id"),
            ModelField("thesis_id"),
            ModelField("symbol", serialize=lambda v: v.upper() if v else None),
            ModelField("agent"),
            ModelField("type", serialize=lambda v: v.value, deserialize=EvidenceType),
            ModelField("content"),
            ModelField("source"),
            ModelField("confidence"),
            ModelField("dimension", serialize=lambda v: v.value if v else None, deserialize=lambda raw: ThesisDimension(raw) if raw else None),
            ModelField("metadata", serialize=BaseRepository.dumps_json, deserialize=lambda raw: BaseRepository.loads_json(raw, {}, "metadata")),
            ModelField("timestamp", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw)),
        ],
        default_order="timestamp DESC",
    )

    # Provenance columns are stored alongside Evidence rows but live on the
    # ``provenance`` dict on the model. Handled via override hooks below rather
    # than as ModelFields so the generic CRUD helpers stay simple.
    _PROVENANCE_COLUMNS = (
        "llm_call_id",
        "model_snapshot",
        "prompt_sha256",
        "input_sha256",
        "output_sha256",
    )

    def create(self, evidence: Evidence, dedupe: bool = False) -> Evidence | None:
        """Insert new evidence, optionally deduping.

        Provenance columns are persisted alongside the row when
        ``evidence.provenance`` is populated.
        """
        if dedupe and self.exists_similar(evidence):
            return None
        self._insert_with_provenance(evidence)
        return evidence

    def _insert_with_provenance(self, evidence: Evidence) -> None:
        meta = self._META
        base_columns = [field.column for field in meta.fields]
        base_values = [field.serialize(getattr(evidence, field.attr)) for field in meta.fields]
        prov = evidence.provenance if isinstance(evidence.provenance, dict) else {}
        prov_values = [prov.get(col) for col in self._PROVENANCE_COLUMNS]

        all_columns = base_columns + list(self._PROVENANCE_COLUMNS)
        all_values = base_values + prov_values
        placeholders = ", ".join(["?"] * len(all_columns))
        column_list = ", ".join(all_columns)
        self.conn.execute(
            f"INSERT INTO {meta.table} ({column_list}) VALUES ({placeholders})",
            all_values,
        )
        self.conn.commit()

    def _row_to_model(self, meta, row):  # type: ignore[override]
        evidence = super()._row_to_model(meta, row)
        if meta is self._META:
            prov: dict[str, str] = {}
            try:
                row_keys = set(row.keys())
            except Exception:
                row_keys = set()
            for col in self._PROVENANCE_COLUMNS:
                if col in row_keys and row[col] is not None:
                    prov[col] = row[col]
            evidence.provenance = prov or None
        return evidence

    def get(self, evidence_id: str) -> Evidence | None:
        """Get evidence by ID."""
        return self._get_by_id(self._META, evidence_id)

    def exists_similar(self, evidence: Evidence) -> bool:
        """Check if similar evidence already exists.

        Matches on (thesis_id OR symbol) + agent + content.
        """
        if evidence.thesis_id:
            cursor = self.conn.execute(
                "SELECT 1 FROM evidence WHERE thesis_id = ? AND agent = ? AND content = ? LIMIT 1",
                (evidence.thesis_id, evidence.agent, evidence.content),
            )
        elif evidence.symbol:
            cursor = self.conn.execute(
                "SELECT 1 FROM evidence WHERE symbol = ? AND agent = ? AND content = ? LIMIT 1",
                (evidence.symbol.upper(), evidence.agent, evidence.content),
            )
        else:
            return False
        return cursor.fetchone() is not None

    def list_for_thesis(self, thesis_id: str) -> list[Evidence]:
        """List all evidence for a thesis."""
        return self._list(self._META, where="thesis_id = ?", params=(thesis_id,))

    def list_for_symbol(self, symbol: str) -> list[Evidence]:
        """List all evidence for a symbol (including orphan evidence)."""
        return self._list(self._META, where="symbol = ?", params=(symbol.upper(),))

    def list_orphans(self, symbol: str | None = None) -> list[Evidence]:
        """List evidence without a thesis, optionally filtered by symbol."""
        if symbol:
            return self._list(
                self._META,
                where="thesis_id IS NULL AND symbol = ?",
                params=(symbol.upper(),),
            )
        return self._list(self._META, where="thesis_id IS NULL")

    def link_to_thesis(self, evidence_id: str, thesis_id: str) -> bool:
        """Link orphan evidence to a thesis."""
        cursor = self.conn.execute(
            "UPDATE evidence SET thesis_id = ? WHERE id = ?",
            (thesis_id, evidence_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def link_symbol_to_thesis(self, symbol: str, thesis_id: str) -> int:
        """Link all orphan evidence for a symbol to a thesis. Returns count updated."""
        cursor = self.conn.execute(
            "UPDATE evidence SET thesis_id = ? WHERE thesis_id IS NULL AND symbol = ?",
            (thesis_id, symbol.upper()),
        )
        self.conn.commit()
        return cursor.rowcount

    def delete(self, evidence_id: str) -> bool:
        """Delete evidence by ID."""
        return self._delete(self._META, "id = ?", (evidence_id,)) > 0

    def delete_for_thesis(self, thesis_id: str) -> int:
        """Delete all evidence for a thesis. Returns count deleted."""
        return self._delete(self._META, "thesis_id = ?", (thesis_id,))

    def prune_for_closed_theses(self, retention_days: int = 30) -> int:
        """Delete evidence for theses closed more than retention_days ago."""
        cursor = self.conn.execute(
            """
            DELETE FROM evidence
            WHERE thesis_id IN (
                SELECT id FROM theses
                WHERE status = 'closed'
                AND closed_at IS NOT NULL
                AND date(closed_at) < date('now', ?)
            )
            """,
            (f"-{retention_days} days",),
        )
        self.conn.commit()
        return cursor.rowcount


class OutcomeRepository(BaseRepository):
    """CRUD operations for outcomes."""

    _META = ModelMeta(
        table="outcomes",
        model=Outcome,
        fields=[
            ModelField("id"),
            ModelField("position_id"),
            ModelField("pnl"),
            ModelField("pnl_pct"),
            ModelField("thesis_accuracy", serialize=lambda v: v.value, deserialize=ThesisAccuracy),
            ModelField("agent_performance", serialize=BaseRepository.dumps_json, deserialize=lambda raw: BaseRepository.loads_json(raw, {}, "agent_performance")),
            ModelField("retrospective"),
            ModelField("recorded_at", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw)),
        ],
        default_order="recorded_at DESC",
    )

    def create(self, outcome: Outcome) -> Outcome:
        """Insert a new outcome."""
        return self._insert(self._META, outcome)

    def get_for_position(self, position_id: str) -> Outcome | None:
        """Get outcome for a position."""
        return self._get_by_id(self._META, position_id, key_column="position_id")

    def list(self) -> list[Outcome]:
        """List all outcomes."""
        return self._list(self._META)

    def delete(self, outcome_id: str) -> bool:
        """Delete an outcome by ID."""
        return self._delete(self._META, "id = ?", (outcome_id,)) > 0


class WatchlistRepository(BaseRepository):
    """CRUD operations for watchlist."""

    _META = ModelMeta(
        table="watchlist",
        model=WatchlistItem,
        fields=[
            ModelField("id"),
            ModelField("symbol", serialize=lambda v: v.upper()),
            ModelField("notes"),
            ModelField("thesis_id"),
            ModelField("threshold"),
            ModelField("alert_destination"),
            ModelField("last_scanned", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw) if raw else None),
            ModelField("created_at", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw), update=False),
        ],
        default_order="created_at DESC",
    )

    def add(self, item: WatchlistItem) -> WatchlistItem:
        """Add a symbol to watchlist."""
        return self._insert(self._META, item, replace=True)

    def remove(self, symbol: str) -> bool:
        """Remove a symbol from watchlist."""
        return self._delete(self._META, "symbol = ?", (symbol.upper(),)) > 0

    def get(self, symbol: str) -> WatchlistItem | None:
        """Get watchlist item by symbol."""
        return self._get_by_id(self._META, symbol.upper(), key_column="symbol")

    def list(self) -> list[WatchlistItem]:
        """List all watchlist items."""
        return self._list(self._META)

    def update_scanned(self, symbol: str) -> None:
        """Update last_scanned timestamp."""
        self.conn.execute(
            "UPDATE watchlist SET last_scanned = ? WHERE symbol = ?",
            (datetime.now().isoformat(), symbol.upper()),
        )
        self.conn.commit()


class AlertRepository(BaseRepository):
    """CRUD operations for alerts."""

    _META = ModelMeta(
        table="alerts",
        model=Alert,
        fields=[
            ModelField("id"),
            ModelField("symbol"),
            ModelField("alert_type", serialize=lambda v: v.value, deserialize=AlertType),
            ModelField("message"),
            ModelField("data", serialize=BaseRepository.dumps_json, deserialize=lambda raw: BaseRepository.loads_json(raw, {}, "data")),
            ModelField("read", serialize=lambda v: 1 if v else 0, deserialize=lambda raw: bool(raw)),
            ModelField("created_at", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw)),
        ],
        default_order="created_at DESC",
    )

    def create(self, alert: Alert) -> Alert:
        """Create a new alert."""
        return self._insert(self._META, alert)

    def list_unread(self) -> list[Alert]:
        """List unread alerts."""
        return self._list(self._META, where="read = 0")

    def list_all(self, limit: int = 50) -> list[Alert]:
        """List all alerts."""
        return self._list(self._META, limit=limit)

    def find_invalidation_for(
        self, thesis_id: str, since: datetime | None = None
    ) -> Alert | None:
        """Return the most-recent PREMISE_INVALIDATION alert for this thesis.

        SQL-side filter on alert_type + json_extract(data, '$.thesis_id') + since
        so the factory close-phase can't get a silent false negative once total
        alert volume exceeds whatever limit a Python-side scan was using.
        """
        params: list[Any] = [AlertType.PREMISE_INVALIDATION.value, thesis_id]
        where = "alert_type = ? AND json_extract(data, '$.thesis_id') = ?"
        if since is not None:
            where += " AND created_at >= ?"
            params.append(_isoformat(since))
        rows = self._list(self._META, where=where, params=tuple(params), limit=1)
        return rows[0] if rows else None

    def mark_read(self, alert_id: str) -> bool:
        """Mark an alert as read."""
        cursor = self.conn.execute(
            "UPDATE alerts SET read = 1 WHERE id = ?",
            (alert_id,),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def mark_all_read(self) -> int:
        """Mark all alerts as read."""
        cursor = self.conn.execute("UPDATE alerts SET read = 1 WHERE read = 0")
        self.conn.commit()
        return cursor.rowcount

    def delete(self, alert_id: str) -> bool:
        """Delete an alert by ID."""
        return self._delete(self._META, "id = ?", (alert_id,)) > 0

    def delete_read(self) -> int:
        """Delete all read alerts. Returns count deleted."""
        return self._delete(self._META, "read = 1", ())


class BacktestRepository(BaseRepository):
    """CRUD operations for backtests and signals."""

    _META = ModelMeta(
        table="backtests",
        model=Backtest,
        fields=[
            ModelField("id"),
            ModelField("symbol", serialize=lambda v: v.upper()),
            ModelField("start_date", serialize=_isoformat, deserialize=lambda raw: date.fromisoformat(raw)),
            ModelField("end_date", serialize=_isoformat, deserialize=lambda raw: date.fromisoformat(raw)),
            ModelField("created_at", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw)),
        ],
        default_order="created_at DESC",
    )

    _SIGNAL_META = ModelMeta(
        table="backtest_signals",
        model=BacktestSignal,
        fields=[
            ModelField("id"),
            ModelField("backtest_id"),
            ModelField("date", serialize=_isoformat, deserialize=lambda raw: date.fromisoformat(raw)),
            ModelField("agent_name"),
            ModelField("conviction_delta"),
            ModelField("dimension_scores", serialize=BaseRepository.dumps_json, deserialize=lambda raw: BaseRepository.loads_json(raw, {}, "dimension_scores")),
            ModelField("forward_returns", serialize=BaseRepository.dumps_json, deserialize=lambda raw: BaseRepository.loads_json(raw, {}, "forward_returns")),
        ],
        default_order="date ASC",
    )

    def create(self, backtest: Backtest) -> Backtest:
        """Create a new backtest."""
        return self._insert(self._META, backtest)

    def get(self, backtest_id: str) -> Backtest | None:
        """Get backtest by ID."""
        return self._get_by_id(self._META, backtest_id)

    def get_signals(self, backtest_id: str) -> list[BacktestSignal]:
        """Get all signals for a backtest."""
        return self._list(
            self._SIGNAL_META,
            where="backtest_id = ?",
            params=(backtest_id,),
            order_by="date ASC",
        )

    def get_signal_history(
        self,
        symbol: str,
        agent_name: str,
        limit: int = 50,
        horizon: str = "20d",
    ) -> list[BacktestSignal]:
        """Get recent signals for a symbol/agent with forward returns."""
        rows = self.conn.execute(
            """
            SELECT bs.* FROM backtest_signals bs
            JOIN backtests b ON bs.backtest_id = b.id
            WHERE b.symbol = ? AND bs.agent_name = ?
            ORDER BY bs.date DESC
            LIMIT ?
            """,
            (symbol.upper(), agent_name, limit * 2),
        ).fetchall()

        signals: list[BacktestSignal] = []
        for row in rows:
            signal = self._row_to_model(self._SIGNAL_META, row)
            if horizon in signal.forward_returns:
                signals.append(signal)
                if len(signals) >= limit:
                    break

        return signals

    def list(self, symbol: str | None = None) -> list[Backtest]:
        """List backtests, optionally filtered by symbol."""
        if symbol:
            return self._list(self._META, where="symbol = ?", params=(symbol.upper(),))
        return self._list(self._META)

    def delete(self, backtest_id: str) -> bool:
        """Delete a backtest by ID (cascades to signals)."""
        return self._delete(self._META, "id = ?", (backtest_id,)) > 0

    def add_signal(self, signal: BacktestSignal) -> BacktestSignal:
        """Add a signal to a backtest."""
        return self._insert(self._SIGNAL_META, signal)


class EventRepository(BaseRepository):
    """CRUD operations for events."""

    _META = ModelMeta(
        table="events",
        model=Event,
        fields=[
            ModelField("id"),
            ModelField("source"),
            ModelField("source_id"),
            ModelField("event_type"),
            ModelField("title"),
            ModelField("summary"),
            ModelField("url"),
            ModelField("occurred_at", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw)),
            ModelField("affected_symbols", serialize=BaseRepository.dumps_json, deserialize=lambda raw: BaseRepository.loads_json(raw, [], "affected_symbols")),
            ModelField("affected_sectors", serialize=BaseRepository.dumps_json, deserialize=lambda raw: BaseRepository.loads_json(raw, [], "affected_sectors")),
            ModelField("tags", serialize=BaseRepository.dumps_json, deserialize=lambda raw: BaseRepository.loads_json(raw, [], "tags")),
            ModelField("polarity", serialize=lambda v: v.value, deserialize=EventPolarity),
            ModelField("confidence"),
            ModelField("raw_text"),
            ModelField("metadata", serialize=BaseRepository.dumps_json, deserialize=lambda raw: BaseRepository.loads_json(raw, {}, "metadata")),
            ModelField(
                "tag_status",
                serialize=lambda v: v.value,
                deserialize=lambda raw: EventTagStatus(raw) if raw else EventTagStatus.TAGGER_SKIPPED,
            ),
            ModelField("ingested_at", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw)),
        ],
        default_order="occurred_at DESC",
    )

    def create(self, event: Event) -> Event | None:
        """Insert a new event. Returns None if a duplicate (same source + source_id) exists."""
        if self.exists(event.source, event.source_id):
            return None
        return self._insert(self._META, event)

    def exists(self, source: str, source_id: str) -> bool:
        """Check if an event from `source` with `source_id` is already stored."""
        cursor = self.conn.execute(
            "SELECT 1 FROM events WHERE source = ? AND source_id = ? LIMIT 1",
            (source, source_id),
        )
        return cursor.fetchone() is not None

    def get(self, event_id: str) -> Event | None:
        """Get event by ID."""
        return self._get_by_id(self._META, event_id)

    def list_recent(
        self,
        since: datetime | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """List events occurring since a given time, optionally filtered by tag.

        Tag filter matches if ANY of the requested tags appears in the event's tags.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if since is not None:
            clauses.append("occurred_at >= ?")
            params.append(since.isoformat())
        where = " AND ".join(clauses) if clauses else None
        events = self._list(self._META, where=where, params=params, limit=limit)
        if not tags:
            return events
        tag_set = set(tags)
        return [e for e in events if tag_set & set(e.tags)]

    def latest_occurred_at(self, source: str) -> datetime | None:
        """Most recent occurred_at for a given source — used to bound incremental pulls."""
        row = self.conn.execute(
            "SELECT MAX(occurred_at) AS ts FROM events WHERE source = ?",
            (source,),
        ).fetchone()
        if row is None or row["ts"] is None:
            return None
        return datetime.fromisoformat(row["ts"])

    def delete(self, event_id: str) -> bool:
        """Delete an event by ID."""
        return self._delete(self._META, "id = ?", (event_id,)) > 0


class LLMUsageRepository(BaseRepository):
    """CRUD operations for LLM usage records."""

    _META = ModelMeta(
        table="llm_usage",
        model=LLMUsage,
        fields=[
            ModelField("id"),
            ModelField("model"),
            ModelField("agent"),
            ModelField("operation"),
            ModelField("input_tokens"),
            ModelField("output_tokens"),
            ModelField("cache_read_input_tokens"),
            ModelField("cache_creation_input_tokens"),
            ModelField("context"),
            ModelField("called_at", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw)),
        ],
        default_order="called_at DESC",
    )

    def create(self, usage: LLMUsage) -> LLMUsage:
        """Insert a new usage record."""
        return self._insert(self._META, usage)

    def get(self, usage_id: str) -> LLMUsage | None:
        """Get a usage record by ID."""
        return self._get_by_id(self._META, usage_id)

    def list_recent(
        self,
        since: datetime | None = None,
        limit: int | None = 100,
    ) -> list[LLMUsage]:
        """List usage records since a given time, newest first.

        Pass ``limit=None`` to disable the limit (used by per-thesis attribution
        in ``factory analyze`` which needs to walk the full window).
        """
        if since is not None:
            return self._list(
                self._META, where="called_at >= ?", params=(since.isoformat(),), limit=limit
            )
        return self._list(self._META, limit=limit)

    def aggregate(
        self,
        dimension: str,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Aggregate usage by a dimension. Returns list of dicts ordered by total calls DESC.

        `dimension` must be one of: "agent", "model", "operation", "day".
        """
        if dimension not in ("agent", "model", "operation", "day"):
            raise ValueError(f"Unsupported dimension: {dimension}")

        # SQLite stores called_at as an ISO8601 string; substr(1,10) gives YYYY-MM-DD.
        group_expr = "substr(called_at, 1, 10)" if dimension == "day" else dimension

        clauses: list[str] = []
        params: list[Any] = []
        if since is not None:
            clauses.append("called_at >= ?")
            params.append(since.isoformat())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

        query = (
            f"SELECT {group_expr} AS bucket, "
            "model AS model, "
            "COUNT(*) AS calls, "
            "SUM(input_tokens) AS input_tokens, "
            "SUM(output_tokens) AS output_tokens, "
            "SUM(cache_read_input_tokens) AS cache_read, "
            "SUM(cache_creation_input_tokens) AS cache_write "
            f"FROM llm_usage{where} "
            f"GROUP BY {group_expr}, model "
            "ORDER BY calls DESC"
        )
        rows = self.conn.execute(query, params).fetchall()
        return [
            {
                "bucket": row["bucket"],
                "model": row["model"],
                "calls": int(row["calls"] or 0),
                "input_tokens": int(row["input_tokens"] or 0),
                "output_tokens": int(row["output_tokens"] or 0),
                "cache_read": int(row["cache_read"] or 0),
                "cache_write": int(row["cache_write"] or 0),
            }
            for row in rows
        ]


class UniverseRepository(BaseRepository):
    """CRUD operations for universes."""

    _META = ModelMeta(
        table="universes",
        model=Universe,
        fields=[
            ModelField("name"),
            ModelField("description"),
            ModelField("source", serialize=lambda v: v.value, deserialize=UniverseSource),
            ModelField("source_config", serialize=BaseRepository.dumps_json, deserialize=lambda raw: BaseRepository.loads_json(raw, {}, "source_config")),
            ModelField("symbols", serialize=BaseRepository.dumps_json, deserialize=lambda raw: BaseRepository.loads_json(raw, [], "symbols")),
            ModelField("is_default", serialize=lambda v: 1 if v else 0, deserialize=lambda raw: bool(raw)),
            ModelField("id"),
            ModelField("created_at", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw), update=False),
            ModelField("updated_at", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw)),
        ],
        default_order="name ASC",
    )

    def create(self, universe: Universe) -> Universe:
        return self._insert(self._META, universe)

    def get(self, name: str) -> Universe | None:
        return self._get_by_id(self._META, name.strip().lower(), key_column="name")

    def list(self) -> list[Universe]:
        return self._list(self._META)

    def update(self, universe: Universe) -> Universe:
        universe.updated_at = datetime.now()
        return self._update(self._META, universe, key_attr="name")

    def delete(self, name: str) -> bool:
        return self._delete(self._META, "name = ?", (name.strip().lower(),)) > 0

    def get_default(self) -> Universe | None:
        rows = self._list(self._META, where="is_default = 1", limit=1)
        return rows[0] if rows else None

    def set_default(self, name: str) -> Universe | None:
        target = self.get(name)
        if target is None:
            return None
        self.conn.execute("UPDATE universes SET is_default = 0")
        self.conn.execute(
            "UPDATE universes SET is_default = 1, updated_at = ? WHERE name = ?",
            (datetime.now().isoformat(), target.name),
        )
        self.conn.commit()
        return self.get(target.name)


class FactoryRunRepository(BaseRepository):
    """CRUD operations for factory run logs."""

    _META = ModelMeta(
        table="factory_runs",
        model=FactoryRun,
        fields=[
            ModelField("id"),
            ModelField("universe_name"),
            ModelField("started_at", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw)),
            ModelField("completed_at", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw) if raw else None),
            ModelField("theses_opened"),
            ModelField("theses_closed"),
            ModelField("positions_opened"),
            ModelField("positions_closed"),
            ModelField("preemptions"),
            ModelField("events_refreshed"),
            ModelField("llm_input_tokens"),
            ModelField("llm_output_tokens"),
            ModelField("llm_cost_usd"),
            ModelField("dry_run", serialize=lambda v: 1 if v else 0, deserialize=lambda raw: bool(raw)),
            ModelField("summary_json", serialize=BaseRepository.dumps_json, deserialize=lambda raw: BaseRepository.loads_json(raw, {}, "summary_json")),
            ModelField("error"),
        ],
        default_order="started_at DESC",
    )

    def create(self, run: FactoryRun) -> FactoryRun:
        return self._insert(self._META, run)

    def update(self, run: FactoryRun) -> FactoryRun:
        return self._update(self._META, run)

    def get(self, run_id: str) -> FactoryRun | None:
        return self._get_by_id(self._META, run_id)

    def list(self, limit: int | None = None) -> list[FactoryRun]:
        return self._list(self._META, limit=limit)

    def latest(self) -> FactoryRun | None:
        rows = self._list(self._META, limit=1)
        return rows[0] if rows else None

    def delete(self, run_id: str) -> bool:
        return self._delete(self._META, "id = ?", (run_id,)) > 0


class ExperimentRepository(BaseRepository):
    """CRUD operations for registered experiments (cents-hvz)."""

    _META = ModelMeta(
        table="experiments",
        model=Experiment,
        fields=[
            ModelField("id"),
            ModelField("name"),
            ModelField("hypothesis"),
            ModelField("primary_metric"),
            ModelField("minimum_n_per_arm"),
            ModelField("stopping_rule"),
            ModelField("minimum_calendar_days"),
            ModelField("frozen_config_sha"),
            ModelField("frozen_config_json"),
            ModelField("started_at", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw)),
            ModelField("finalized_at", serialize=_isoformat, deserialize=lambda raw: datetime.fromisoformat(raw) if raw else None),
            ModelField("verdict_json"),
            ModelField("status"),
        ],
        default_order="started_at DESC",
    )

    def create(self, exp: Experiment) -> Experiment:
        return self._insert(self._META, exp)

    def update(self, exp: Experiment) -> Experiment:
        return self._update(self._META, exp)

    def get(self, exp_id: str) -> Experiment | None:
        return self._get_by_id(self._META, exp_id)

    def get_by_name(self, name: str) -> Experiment | None:
        rows = self._list(self._META, where="name = ?", params=(name,), limit=1)
        return rows[0] if rows else None

    def list(self, status: str | None = None) -> list[Experiment]:
        if status:
            return self._list(self._META, where="status = ?", params=(status,))
        return self._list(self._META)

    def list_active(self) -> list[Experiment]:
        return self.list(status="active")

    def delete(self, exp_id: str) -> bool:
        return self._delete(self._META, "id = ?", (exp_id,)) > 0


class DelistingsRepository(BaseRepository):
    """CRUD operations for tracked delistings (cents-5fh).

    Used by the universe resolver to reconstruct point-in-time membership of
    screener-sourced universes — symbols delisted between the as-of date and
    today still need to appear in the resolved member list, otherwise every
    backtest is biased toward survivors.
    """

    _META = ModelMeta(
        table="delistings",
        model=Delisting,
        fields=[
            ModelField("symbol", serialize=lambda v: v.upper()),
            ModelField(
                "delisted_on",
                serialize=_isoformat,
                deserialize=lambda raw: date.fromisoformat(raw),
            ),
            ModelField("last_close"),
            ModelField("source"),
            ModelField(
                "ingested_at",
                serialize=_isoformat,
                deserialize=lambda raw: datetime.fromisoformat(raw),
            ),
        ],
        default_order="delisted_on DESC",
    )

    def upsert(self, delisting: Delisting) -> Delisting:
        """Insert or replace a delisting record by symbol."""
        return self._insert(self._META, delisting, replace=True)

    def get(self, symbol: str) -> Delisting | None:
        return self._get_by_id(self._META, symbol.strip().upper(), key_column="symbol")

    def list_since(self, since: date) -> list[Delisting]:
        return self._list(
            self._META,
            where="delisted_on >= ?",
            params=(since.isoformat(),),
        )

    def list_all(self) -> list[Delisting]:
        return self._list(self._META)

    def delete(self, symbol: str) -> bool:
        return self._delete(
            self._META, "symbol = ?", (symbol.strip().upper(),),
        ) > 0
class ShadowOpenRepository(BaseRepository):
    """CRUD operations for shadow-opens (rejected factory candidates)."""

    _META = ModelMeta(
        table="shadow_opens",
        model=ShadowOpen,
        fields=[
            ModelField("id"),
            ModelField("run_id"),
            ModelField("symbol", serialize=lambda v: v.upper() if v else None),
            ModelField("would_be_entry_price"),
            ModelField("conviction_delta"),
            ModelField("primary_side"),
            ModelField(
                "premise_tags",
                serialize=BaseRepository.dumps_json,
                deserialize=lambda raw: BaseRepository.loads_json(raw, [], "premise_tags"),
            ),
            ModelField(
                "premise_direction",
                serialize=BaseRepository.dumps_json,
                deserialize=lambda raw: BaseRepository.loads_json(raw, {}, "premise_direction"),
            ),
            ModelField(
                "regime_snapshot",
                serialize=BaseRepository.dumps_json,
                deserialize=lambda raw: BaseRepository.loads_json(raw, {}, "regime_snapshot"),
            ),
            ModelField("reason"),
            ModelField("orchestrator_label"),
            ModelField("experiment_id"),
            ModelField("discovery_source"),
            ModelField("horizon_days"),
            ModelField("forward_return_30d"),
            ModelField("forward_return_60d"),
            ModelField(
                "backfilled_at",
                serialize=_isoformat,
                deserialize=lambda raw: datetime.fromisoformat(raw) if raw else None,
            ),
            ModelField(
                "created_at",
                serialize=_isoformat,
                deserialize=lambda raw: datetime.fromisoformat(raw),
                update=False,
            ),
        ],
        default_order="created_at DESC",
    )

    def create(self, shadow: ShadowOpen) -> ShadowOpen:
        """Insert a new shadow-open row."""
        return self._insert(self._META, shadow)

    def get(self, shadow_id: str) -> ShadowOpen | None:
        """Get a shadow-open row by ID."""
        return self._get_by_id(self._META, shadow_id)

    def update(self, shadow: ShadowOpen) -> ShadowOpen:
        """Update an existing shadow-open row (e.g. after backfill)."""
        return self._update(self._META, shadow)

    def list(
        self,
        *,
        reason: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> "list[ShadowOpen]":
        """List shadow-opens, optionally filtered by reason or run_id."""
        clauses: "list[str]" = []
        params: "list[Any]" = []
        if reason is not None:
            clauses.append("reason = ?")
            params.append(reason)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        where = " AND ".join(clauses) if clauses else None
        return self._list(self._META, where=where, params=params, limit=limit)

    def list_pending_backfill(
        self,
        *,
        since: datetime | None = None,
        horizon_days: int = 30,
    ) -> "list[ShadowOpen]":
        """List shadow-opens older than `horizon_days` whose forward return is unset.

        Args:
            since: Only include rows created at-or-after this datetime (optional bound).
            horizon_days: Which horizon column to check (30 or 60). Default 30.
        """
        column = "forward_return_60d" if horizon_days >= 60 else "forward_return_30d"
        clauses: "list[str]" = [f"{column} IS NULL"]
        params: "list[Any]" = []
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since.isoformat())
        where = " AND ".join(clauses)
        return self._list(self._META, where=where, params=params)

    def delete(self, shadow_id: str) -> bool:
        """Delete a shadow-open row by ID."""
        return self._delete(self._META, "id = ?", (shadow_id,)) > 0
