# tools/wikipedia.py
from langchain_community.utilities import WikipediaAPIWrapper
from kani import ai_function
from tools import ToolBase

class WikipediaSearch(ToolBase):
    def __init__(self, app, kani):
        super().__init__(app, kani)
        self.wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1000)

    @ai_function()
    async def search_wikipedia(self, query: str) -> str:
        """Search Wikipedia for the given query and return a relevant excerpt."""
        try:
            return self.wrapper.run(query)
        except Exception as e:
            return f"Error fetching Wikipedia content: {e}"
