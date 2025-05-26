# tools/semantic.py
import httpx
from kani import ai_function
from tools import ToolBase

class SemanticScholarSearch(ToolBase):
    @ai_function()
    async def search_semantic(self, query: str):
        """Query Semantic Scholar for relevant papers."""
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=3&fields=title,abstract,url,citationCount"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            data = resp.json()
        
        return "\n\n".join(
            f"📄 {paper['title']}\n🔗 {paper['url']}\n📚 Cited: {paper['citationCount']}\n📝 {paper['abstract']}"
            for paper in data.get("data", [])
        )
