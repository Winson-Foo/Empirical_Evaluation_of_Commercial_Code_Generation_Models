import os
import time
from typing import Optional
from sqlite3 import Cursor
import sqlite3
from nboost import defaults


DB_FILE = defaults.db_file
TABLE_NAME = 'searches'
COLUMN_NAMES = [
    'time', 'topk', 'choices', 'qa_time',
    'model_mrr', 'server_mrr', 'rerank_time', 'response_time'
]


class SearchStatsDatabase:
    def __init__(self, db_file: Optional[str] = DB_FILE):
        os.makedirs(db_file.parent, exist_ok=True)
        self.db_file = db_file

    def _execute_query(self, query: str, params: Optional[tuple] = None):
        with sqlite3.connect(str(self.db_file), isolation_level=None) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)

    def insert(self, topk: int, choices: int, qa_time: float,
               model_mrr: float, server_mrr: float,
               rerank_time: float, response_time: float):
        self._execute_query(f'''
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                time REAL, topk INTEGER, choices INTEGER, qa_time REAL,
                model_mrr REAL, server_mrr REAL, rerank_time REAL, response_time REAL
            );
        ''')
        self._execute_query(f'''
            INSERT INTO {TABLE_NAME}({", ".join(COLUMN_NAMES[1:])})
            VALUES(?,?,?,?,?,?,?);
        ''', (
            time.time(), topk, choices, qa_time,
            model_mrr, server_mrr, rerank_time, response_time
        ))

    def get_stats(self) -> dict:
        with sqlite3.connect(str(self.db_file), isolation_level=None) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f'''
                    SELECT AVG(topk), AVG(choices), AVG(rerank_time),
                           AVG(response_time), AVG(model_mrr), AVG(server_mrr)
                    FROM {TABLE_NAME}
                ''')
                stats = cursor.fetchone()
                return dict(zip(COLUMN_NAMES[1:], stats))


    class SearchStats:
        def __init__(self):
            self.topk = None
            self.choices = None
            self.qa_time = None
            self.model_mrr = None
            self.server_mrr = None
            self.rerank_time = None
            self.response_time = None