import json
import sqlite3
import time
from typing import Dict, List, Optional, Any


class TelemetryCollector:
    """Collects per-turn decision data from games into SQLite.

    Designed for batch simulation. Buffers rows in memory and flushes
    periodically to minimize I/O overhead during fast simulations.
    """

    FLUSH_INTERVAL = 100

    def __init__(self, db_path: str = "telemetry.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._turn_buffer: List[tuple] = []
        self._game_buffer: List[tuple] = []
        self._ticket_buffer: List[tuple] = []
        self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                version TEXT,
                player_types TEXT,
                started_at REAL,
                ended_at REAL,
                winner_id INTEGER,
                final_scores TEXT,
                completed_tickets TEXT,
                total_tickets TEXT,
                claimed_routes TEXT,
                longest_path_lengths TEXT
            );

            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                turn_number INTEGER,
                player_id INTEGER,
                player_type TEXT,
                action INTEGER,
                move_valid INTEGER,
                hand_json TEXT,
                trains_remaining INTEGER,
                score INTEGER,
                tickets_held INTEGER,
                tickets_completed INTEGER,
                detail_json TEXT,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            );

            CREATE TABLE IF NOT EXISTS ticket_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                turn_number INTEGER,
                player_id INTEGER,
                tickets_offered TEXT,
                tickets_kept TEXT,
                tickets_discarded TEXT,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            );

            CREATE INDEX IF NOT EXISTS idx_turns_game ON turns(game_id);
            CREATE INDEX IF NOT EXISTS idx_turns_player ON turns(player_id);
            CREATE INDEX IF NOT EXISTS idx_turns_action ON turns(action);
        """)
        self._conn.commit()

    def start_game(self, game_id: str, version: str, player_types: List[str]) -> str:
        self._game_buffer.append((
            game_id, version, json.dumps(player_types), time.time(),
            None, None, None, None, None, None, None,
        ))
        return game_id

    def record_turn(
        self,
        game_id: str,
        turn_number: int,
        player_id: int,
        player_type: str,
        action: int,
        state_snapshot: Dict[str, Any],
        decision_detail: Dict[str, Any],
        move_valid: bool,
    ):
        self._turn_buffer.append((
            game_id,
            turn_number,
            player_id,
            player_type,
            action,
            1 if move_valid else 0,
            json.dumps(state_snapshot.get('hand', {})),
            state_snapshot.get('trains_remaining', 0),
            state_snapshot.get('score', 0),
            state_snapshot.get('tickets_held', 0),
            state_snapshot.get('tickets_completed', 0),
            json.dumps(decision_detail),
        ))

        if len(self._turn_buffer) >= self.FLUSH_INTERVAL:
            self.flush()

    def record_ticket_decision(
        self,
        game_id: str,
        turn_number: int,
        player_id: int,
        tickets_offered: List[Dict],
        tickets_kept: List[int],
        tickets_discarded: List[int],
    ):
        self._ticket_buffer.append((
            game_id, turn_number, player_id,
            json.dumps(tickets_offered),
            json.dumps(tickets_kept),
            json.dumps(tickets_discarded),
        ))

    def record_game_result(self, game_id: str, stats: Dict):
        scores = stats.get('score', [])
        winner_id = scores.index(max(scores)) if scores else None

        self._conn.execute("""
            UPDATE games SET
                ended_at = ?,
                winner_id = ?,
                final_scores = ?,
                completed_tickets = ?,
                total_tickets = ?,
                claimed_routes = ?,
                longest_path_lengths = ?
            WHERE game_id = ?
        """, (
            time.time(),
            winner_id,
            json.dumps(scores),
            json.dumps(stats.get('completed_tickets', [])),
            json.dumps(stats.get('total_tickets', [])),
            json.dumps(stats.get('claimed_routes', [])),
            json.dumps(stats.get('longest_path_length', [])),
            game_id,
        ))
        self.flush()

    def flush(self):
        if self._game_buffer:
            self._conn.executemany(
                "INSERT OR IGNORE INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                self._game_buffer,
            )
            self._game_buffer.clear()

        if self._turn_buffer:
            self._conn.executemany(
                "INSERT INTO turns (game_id, turn_number, player_id, player_type, "
                "action, move_valid, hand_json, trains_remaining, score, "
                "tickets_held, tickets_completed, detail_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                self._turn_buffer,
            )
            self._turn_buffer.clear()

        if self._ticket_buffer:
            self._conn.executemany(
                "INSERT INTO ticket_decisions (game_id, turn_number, player_id, "
                "tickets_offered, tickets_kept, tickets_discarded) "
                "VALUES (?,?,?,?,?,?)",
                self._ticket_buffer,
            )
            self._ticket_buffer.clear()

        self._conn.commit()

    def close(self):
        self.flush()
        if self._conn:
            self._conn.close()
            self._conn = None
