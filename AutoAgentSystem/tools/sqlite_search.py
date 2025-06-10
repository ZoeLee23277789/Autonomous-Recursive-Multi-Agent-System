# tools/sqlite_search.py

import sqlite3
import json
from kani import ai_function
from ._base import ToolBase

class SQLiteSearch(ToolBase):
    def __init__(self, db_path, **kwargs):
        super().__init__(**kwargs)
        self.db_path = db_path
        self.conn = None

    async def setup(self):
        self.conn = sqlite3.connect(self.db_path)

    async def cleanup(self):
        if self.conn:
            self.conn.close()

    @ai_function(desc="Search Feverous Wiki database by page_id and element_id and return the text content.")
    async def search_feverous(self, page_id: str, element_id: str) -> str:
        # （你原本的 code）
        query = "SELECT data FROM wiki WHERE id = ?"
        cursor = self.conn.cursor()
        cursor.execute(query, (page_id,))
        row = cursor.fetchone()
        if not row:
            return "NOT FOUND (page_id not found)"
        data = json.loads(row[0])

        if not element_id.startswith(page_id):
            return "Invalid element_id for page_id"
        pure_element_id = element_id[len(page_id) + 1:]
        value = data.get(pure_element_id)
        if value:
            return value if isinstance(value, str) else value.get("text", str(value))
        else:
            return "NOT FOUND (element_id not found)"

    @ai_function(desc="Search Feverous Wiki database by keyword in text.")
    async def search_by_text(self, keyword: str) -> str:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, data FROM wiki")
        results = []
        for row in cursor.fetchall():
            page_id = row[0]
            data = json.loads(row[1])
            for key, value in data.items():
                if isinstance(value, str) and keyword.lower() in value.lower():
                    results.append(f"{page_id} - {key}: {value}")
                    if len(results) >= 5:
                        break
            if len(results) >= 5:
                break
        if not results:
            return "NOT FOUND"
        return "\n".join(results)
