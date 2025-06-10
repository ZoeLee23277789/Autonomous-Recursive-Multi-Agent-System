import json
from pathlib import Path
from kani import ai_function
from tools import ToolBase

class WikipediaSearch(ToolBase):
    def __init__(self, app, kani, wiki_dir, prebuilt_index=None):
        self.app = app
        self.kani = kani
        self.wiki_dir = wiki_dir
        self.page_index = prebuilt_index or {}
        self.index_built = bool(prebuilt_index)

    async def setup(self):
        if not self.index_built:
            print("🚀 [WikipediaSearch] Building Index...")
            await self.build_index()
            self.index_built = True

    async def build_index(self):
        for filename in Path(self.wiki_dir).glob("*.jsonl"):
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    page_id = data['id']
                    self.page_index[page_id] = str(filename)
        print(f"✅ [WikipediaSearch] Index built: {len(self.page_index)} pages.")

    @ai_function(desc="Search and return a specific sentence from a given Wikipedia page.")
    async def search_sentence(self, page_id: str, sentence_id: int) -> str:
        """
        Given a Wikipedia page ID and sentence ID, retrieve and return the sentence content.
        """
        if page_id not in self.page_index:
            return f"Page {page_id} not found."
    
        file_path = self.page_index[page_id]
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data['id'] == page_id:
                    try:
                        return data['text'][sentence_id]
                    except IndexError:
                        return f"Sentence ID {sentence_id} not found in {page_id}."
        return f"Page {page_id} not found in {file_path}."

