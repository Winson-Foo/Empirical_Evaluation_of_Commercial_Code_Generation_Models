import os
import time
from typing import Optional
from sqlite3 import Cursor, connect


class Database:
    """
    A class for interacting with a SQLite database.
    """
    CREATE_SEARCHES_TABLE_SQL = '''
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

    INSERT_SEARCH_SQL = '''
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
        VALUES(?, ?, ?, ?, ?, ?, ?, ?);
    '''

    GET_STATS_SQL = '''
        SELECT
            AVG(topk) AS avg_topk,
            AVG(choices) AS avg_num_choices,
            AVG(rerank_time) AS avg_rerank_time,
            AVG(response_time) AS avg_response_time,
            AVG(model_mrr) AS avg_model_mrr,
            AVG(server_mrr) AS avg_server_mrr
        FROM searches;
    '''

    def __init__(self, db_file: str = 'data/searches.db') -> None:
        os.makedirs(os.path.dirname(db_file), exist_ok=True)
        self.db_file = db_file

    def new_search(self) -> 'Search':
        """
        Creates and returns a new search with default values.
        """
        return Search()

    def _get_cursor(self) -> Cursor:
        """
        Returns a cursor to interact with the database.
        """
        conn = connect(self.db_file, isolation_level=None)
        return conn.cursor()

    def insert(self, search: 'Search') -> None:
        """
        Inserts a search into the database.
        """
        with self._get_cursor() as cursor:
            cursor.execute(self.CREATE_SEARCHES_TABLE_SQL)
            cursor.execute(self.INSERT_SEARCH_SQL, search.as_tuple() + (time.time(),))

    def get_stats(self) -> dict:
        """
        Returns the average statistics of all searches in the database.
        """
        with self._get_cursor() as cursor:
            stats = cursor.execute(self.GET_STATS_SQL).fetchone()
            columns = [column[0] for column in cursor.description]
            return dict(zip(columns, stats))


class Search:
    """
    A class representing a search and its associated data.
    """
    def __init__(self) -> None:
        self.topk: Optional[int] = None
        self.choices: Optional[int] = None
        self.qa_time: Optional[float] = None
        self.model_mrr: Optional[float] = None
        self.server_mrr: Optional[float] = None
        self.rerank_time: Optional[float] = None
        self.response_time: Optional[float] = None

    def as_tuple(self) -> tuple:
        """
        Returns the search data as a tuple.
        """
        return (
            self.topk,
            self.choices,
            self.qa_time,
            self.model_mrr,
            self.server_mrr,
            self.rerank_time,
            self.response_time,
        )