# tools/sqlite_search_testable.py
import aiosqlite

class SQLiteSearchTestable:
    def __init__(self, db_path="C:/Users/USER/Downloads/Test_Agent/Test_5/Dataset/feverous_wikiv1.db"):
        self.db_path = db_path
        self.conn = None

    async def setup(self):
        self.conn = await aiosqlite.connect(self.db_path)

    async def cleanup(self):
        if self.conn:
            await self.conn.close()

    async def search_feverous(self, page_id: str, element_id: str) -> str:
        query = """
        SELECT value FROM data
        WHERE page_id = ? AND element_id = ?
        """
        async with self.conn.execute(query, (page_id, element_id)) as cursor:
            row = await cursor.fetchone()
            if row:
                return row[0]
            else:
                return "NOT FOUND"
