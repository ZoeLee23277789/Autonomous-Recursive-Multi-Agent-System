# tools/pubmed.py
from Bio import Entrez
from kani import ai_function
from tools import ToolBase

class PubMedSearch(ToolBase):
    @ai_function()
    async def search_pubmed(self, query: str):
        """Search PubMed for recent biomedical papers and return titles + summaries."""
        Entrez.email = "zoe23277789@gmail.com"  # NCBI 要求加上 email
        handle = Entrez.esearch(db="pubmed", term=query, retmax=3)
        record = Entrez.read(handle)
        ids = record["IdList"]

        summaries = []
        for id in ids:
            fetch = Entrez.efetch(db="pubmed", id=id, rettype="abstract", retmode="text")
            summaries.append(fetch.read())
        return "\n\n".join(summaries)
