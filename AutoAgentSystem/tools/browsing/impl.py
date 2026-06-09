import os
import aiofiles
import asyncio
import contextlib
import logging
import tempfile
import arxiv
from typing import Optional, TYPE_CHECKING

from duckduckgo_search import DDGS
from runtime import ChatMessage, ChatRole, ai_function
from runtime import BaseEngine

try:
    import httpx
    import pymupdf
    import pymupdf4llm
    from playwright.async_api import (
        BrowserContext,
        TimeoutError as PlaywrightTimeoutError,
        async_playwright,
        Error as PlaywrightError,
    )
except ImportError as e:
    # 不中止程式，只是標記某些功能不能用
    print("⚠️ 缺少瀏覽工具所需依賴，Browsing 工具將無法使用。")
    httpx = None
    pymupdf = None
    pymupdf4llm = None
    BrowserContext = None
    PlaywrightTimeoutError = Exception
    async_playwright = None
    PlaywrightError = Exception


from tools import ToolBase
from .webutils import web_markdownify, web_summarize

if TYPE_CHECKING:
    from playwright.async_api import Page

log = logging.getLogger(__name__)


class Browsing(ToolBase):
    """
    A tool that provides tools to search Google and visit webpages.

    Renders webpages in Markdown and has basic support for reading PDFs.
    """

    # app-global browser instance
    playwright = None
    browser = None
    browser_context = None

    def __init__(
        self,
        *args,
        long_engine: BaseEngine = None,
        max_webpage_len: int = None,
        page_concurrency_sem: asyncio.Semaphore | None = None,
        **kwargs,
    ):
        """
        :param long_engine: If a webpage is longer than *max_webpage_len*, send it to this engine to summarize it. If
            not supplied, uses the agent's engine.
        :param max_webpage_len: The maximum length of a webpage to send to the agent at once
            (default max context len / 3).
        :param page_concurrency_sem: A semaphore that this tool will acquire when opening a browser page.
        """
        super().__init__(*args, **kwargs)
        self.http = httpx.AsyncClient(follow_redirects=True)
        self.page: Optional["Page"] = None
        self.long_engine = long_engine
        self.page_concurrency_sem = page_concurrency_sem

        # the max number of tokens before asking for a summary - default 1/3rd ctx len
        if max_webpage_len is None:
            max_webpage_len = self.agent.engine.max_context_size // 3
        self.max_webpage_len = max_webpage_len

        # content handlers
        self.content_handlers = {
            "application/pdf": self.pdf_content,
            "application/json": self.json_content,
            "text/": self.html_content,
        }

    # === resources + app lifecycle ===
    # noinspection PyMethodMayBeStatic
    async def get_browser(self, **kwargs) -> BrowserContext:
        """Get the current active browser context, or launch it on the first call."""
        if Browsing.playwright is None:
            Browsing.playwright = await async_playwright().start()
        if Browsing.browser is None:
            Browsing.browser = await Browsing.playwright.chromium.launch(**kwargs)
        if Browsing.browser_context is None:
            Browsing.browser_context = await Browsing.browser.new_context()
        return Browsing.browser_context

    async def get_page(self, create=True) -> Optional["Page"]:
        """Get the current page.

        Returns None if the browser is not on a page unless `create` is True, in which case it creates a new page.
        """
        if self.page is None and create:
            context = await self.get_browser()
            if self.page_concurrency_sem:
                await self.page_concurrency_sem.acquire()
            self.page = await context.new_page()
        return self.page

    async def cleanup(self):
        await super().cleanup()
        if self.page is not None:
            await self.page.close()
            if self.page_concurrency_sem:
                self.page_concurrency_sem.release()
            self.page = None

    async def close(self):
        await super().close()
        try:
            if (browser := Browsing.browser) is not None:
                Browsing.browser = None
                await browser.close()
            if (pw := Browsing.playwright) is not None:
                Browsing.playwright = None
                await pw.stop()
        except PlaywrightError:
            # sometimes playwright doesn't like closing in parallel
            pass

    # ==== functions ====
    @ai_function()
    async def search(self, query: str):
        """Search for a query on DuckDuckGo using DDGS."""
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=3)
                return "\n\n".join(
                    f"📌 {r['title']}\n🔗 {r['href']}\n📝 {r['body']}"
                    for r in results if "title" in r and "href" in r and "body" in r
                )
        except Exception as e:
            return f"❌ 搜尋失敗：{e}"

    # @ai_function()
    # async def visit_page(self, href: str):
    #     """Visit a web page and view its contents."""
    #     # first, let's do a HEAD request and get the content-type so we know how to actually process the info
    #     resp = await self.http.head(href)
    #     content_type = resp.headers.get("Content-Type", "").lower()

    #     # then delegate to the content type handler
    #     handler = next((f for t, f in self.content_handlers.items() if content_type.startswith(t)), None)
    #     if handler is None:
    #         log.warning(f"Could not find handler for content type: {content_type}")
    #         handler = self.html_content

    #     return await handler(href)
    @ai_function()
    async def visit_page(self, href: str):
        """Visit a web page and view its contents."""
        try:
            # 用 GET 確保 Cloudflare 等不阻擋 HEAD 請求
            resp = await self.http.get(href, timeout=10.0)
            content_type = resp.headers.get("Content-Type", "").lower()
        except Exception as e:
            log.exception(f"Error visiting {href}")
            return f"❌ Failed to visit page: {e}"
    
        # 根據 content-type 選擇解析方式
        handler = next((f for t, f in self.content_handlers.items() if content_type.startswith(t)), None)
        if handler is None:
            log.warning(f"No handler found for {content_type}, using HTML parser")
            handler = self.html_content
    
        try:
            return await handler(href)
        except Exception as e:
            log.exception(f"Handler failed for {href}")
            return f"⚠️ Could not extract content from {href}: {e}"
    


    async def pdf_content(self, href: str) -> str:
        """Handler for application/pdf content types."""
        # 建立臨時檔案
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)  # 關閉檔案描述器，避免 Windows 鎖住
    
        try:
            # 下載 PDF 到臨時檔案
            async with self.http.stream("GET", href) as response:
                async with aiofiles.open(path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        await f.write(chunk)
    
            # 解析 PDF
            doc = pymupdf.open(path)
            content = pymupdf4llm.to_markdown(doc)
    
            # 摘要處理（必要時）
            return await self.maybe_summarize(content)
    
        finally:
            # 清理臨時檔案
            os.remove(path)

    async def json_content(self, href: str) -> str:
        """Handler for application/json content types."""
        resp = await self.http.get(href)
        resp.raise_for_status()
        await resp.aread()
        return resp.text

    # async def html_content(self, href: str) -> str:
    #     """Default handler for all other content types."""
    #     page = await self.get_page()
    #     await page.goto(href)
    #     with contextlib.suppress(PlaywrightTimeoutError):
    #         await page.wait_for_load_state("networkidle", timeout=10_000)
    #     # header
    #     title = await page.title()
    #     header = f"{title}\n{'=' * len(title)}\n{page.url}\n\n"

    #     content_html = await page.content()
    #     content = web_markdownify(content_html)
    #     # summarization
    #     content = await self.maybe_summarize(content)
    #     # result
    #     result = header + content
    #     return result

    async def html_content(self, href: str) -> str:
        """Default handler for HTML content."""
        try:
            page = await self.get_page()
            await page.goto(href, wait_until="domcontentloaded", timeout=10000)
        except Exception as e:
            log.exception(f"Playwright.goto failed for {href}")
            return f"❌ Failed to load the page via Playwright: {e}"
    
        try:
            with contextlib.suppress(PlaywrightTimeoutError):
                await page.wait_for_load_state("networkidle", timeout=10000)
    
            title = await page.title()
            content_html = await page.content()
            content = web_markdownify(content_html)
            summarized = await self.maybe_summarize(content)
    
            return f"{title}\n{'=' * len(title)}\n{page.url}\n\n{summarized}"
    
        except Exception as e:
            log.exception(f"Error extracting HTML from {href}")
            return f"⚠️ Failed to parse page content: {e}"


    # ==== helpers ====
    async def maybe_summarize(self, content, max_len=None):
        max_len = max_len or self.max_webpage_len
        if self.agent.message_token_len(ChatMessage.function("visit_page", content)) > max_len:
            msg_ctx = "\n\n".join(
                m.text for m in self.agent.chat_history if m.role != ChatRole.FUNCTION and m.text is not None
            )
            content = await web_summarize(
                content,
                parent=self.agent,
                long_engine=self.long_engine or self.agent.engine,
                task=(
                    "Keep the current context in mind:\n"
                    f"<context>\n{msg_ctx}\n</context>\n\n"
                    "Keeping the context and task in mind, please summarize the main content above."
                ),
            )
        return content
        
class ArxivSearch(ToolBase):
    @ai_function()
    async def search_arxiv(self, query: str):
        """搜尋 Arxiv 並摘要前幾篇相關論文的摘要與標題"""
        results = arxiv.Search(query=query, max_results=3).results()
        output = []
        for result in results:
            output.append(
                f"📄 **{result.title}**\n{result.summary.strip()}\n🔗 {result.entry_id}"
            )
        return "\n\n".join(output)
