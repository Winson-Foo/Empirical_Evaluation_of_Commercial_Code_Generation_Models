import os
import time
from pathlib import Path
from typing import Optional
from sqlite3 import Cursor, connect
from nboost import defaults

CREATE_TABLE = '''
    CREATE TABLE IF NOT EXISTS searches (
        time REAL,
        topk INTEGER,
        choices INTEGER,
        qa_time REAL,
        model_mrr REAL,
        server_mrr REAL,
        rerank_time REAL,
        response_time REAL
    );
'''

INSERT_ROW = '''
    INSERT INTO searches (
        time,
        topk,
        choices,
        qa_time,
        model_mrr,
        server_mrr,
        rerank_time,
        response_time
    )
    VALUES(?,?,?,?,?,?,?,?);
'''

SELECT_STATS = '''
    SELECT
        AVG(topk) AS avg_topk,
        AVG(choices) AS avg_num_choices,
        AVG(rerank_time) AS avg_rerank_time,
        AVG(response_time) AS avg_response_time,
        AVG(model_mrr) AS avg_model_mrr,
        AVG(server_mrr) AS avg_server_mrr
    FROM searches
'''

class Database:
    def __init__(self, db_file: Path = defaults.db_file, **kwargs) -> None:
        os.makedirs(db_file.parent, exist_ok=True)
        self.db_file = db_file

    def new_row(self) -> 'DatabaseRow':
        return DatabaseRow()

    def insert(self, row: 'DatabaseRow') -> None:
        with connect(self.db_file) as conn:
            conn.execute(CREATE_TABLE)
            conn.execute(INSERT_ROW, (
                time.time(),
                row.topk,
                row.choices,
                row.qa_time,
                row.model_mrr,
                row.server_mrr,
                row.rerank_time,
                row.response_time
            ))

    def get_stats(self) -> dict[str, float]:
        with connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(SELECT_STATS)
            stats = cursor.fetchone()
            columns = [column[0] for column in cursor.description]
            return dict(zip(columns, stats))


class DatabaseRow:
    def __init__(self) -> None:
        self.topk: Optional[int] = None
        self.choices: Optional[int] = None
        self.qa_time: Optional[float] = None
        self.model_mrr: Optional[float] = None
        self.server_mrr: Optional[float] = None
        self.rerank_time: Optional[float] = None
        self.response_time: Optional[float] = None