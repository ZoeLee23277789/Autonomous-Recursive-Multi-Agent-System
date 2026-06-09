# import json
# import asyncio
# import re
# import os
# import events
# import pandas as pd
# from tqdm import tqdm
# from runtime import ChatRole
# from dotenv import load_dotenv

# load_dotenv()
# import sys
# sys.path.append(".")  # 確保當前路徑可以匯入

# # 匯入系統與 Wiki 工具
# from app import AutoAgentSystem
# from tools.wiki_search import WikipediaSearch

# # --- FEVER 資料 ---
# with open(r"C:\Users\USER\Downloads\Test_Agent\Test_5\Dataset\FEVER\shared_task_dev.jsonl", "r", encoding="utf-8") as f:
#     samples = [json.loads(line) for line in f]

# # --- 改良版 Prompt ---
# def prepare_prompt(claim):
#     prompt = (
#         "你是一個 Chief Autonomous Agent，負責判斷以下的事實陳述是否正確：\n\n"
#         f"【Claim】\n{claim}\n\n"
#         "你的工作步驟如下：\n"
#         "1. 仔細閱讀 Claim，拆分出主要的實體（entities）和關鍵概念（concepts）。\n"
#         "2. 指派子助理，並**只允許使用 WikipediaSearch 工具**搜尋證據。\n"
#         "3. ⚠️ **禁止使用未授權的工具或外部資源，例如 Google Search、Browsing。**\n"
#         "4. 每個子助理必須回報：\n"
#         "   - 是否找到證據？（找到或未找到）\n"
#         "   - 找到的證據內容。\n"
#         "   - 證據是支持還是反駁？（SUPPORTS / REFUTES）\n"
#         "5. 如果沒有找到任何證據，子助理必須回報：NOT ENOUGH INFO。\n"
#         "6. ⚠️ **嚴格要求：子助理回報時不可猜測、不可創造內容。**\n\n"
#         "你的最終任務是，整合所有子助理的回報，根據以下規則做出最終判斷：\n"
#         "- 只要有子助理回報 REFUTES，最終結果是 REFUTES。\n"
#         "- 如果沒有 REFUTES，但有 SUPPORTS，最終結果是 SUPPORTS。\n"
#         "- 如果所有子助理都是 NOT ENOUGH INFO，最終結果是 NOT ENOUGH INFO。\n\n"
#         "⚠️ 嚴格要求：只能根據子助理的證據回報做判斷，不可推測。\n\n"
#         "請你最後直接輸出一個詞（只允許輸出 SUPPORTS / REFUTES / NOT ENOUGH INFO）。\n"
#         "❌ 不要加句子，不要加解釋，不要加任何其他內容，只能輸出單一詞彙。"
#     )
#     return prompt



# # --- LLM 回覆標準化 ---
# def normalize_label(text):
#     text = text.lower()
#     if "support" in text:
#         return "SUPPORTS"
#     elif "refute" in text:
#         return "REFUTES"
#     elif "not enough info" in text or "not enough information" in text:
#         return "NOT ENOUGH INFO"
#     else:
#         return "UNKNOWN"

# # --- LLM 單次測試 ---
# async def chat_once(agent_system, user_input: str) -> str:
#     await agent_system.ensure_init()
#     response_text = ""
#     async for stream_manager in agent_system.root_agent.full_round_stream(user_input):
#         message = await stream_manager.message()
#         if message.role == ChatRole.ASSISTANT:
#             response_text += message.content or ""
#     return response_text.strip()

# # --- 主程式 ---
# async def main():
#     # ✅ 預先初始化 WikipediaSearch，避免重複建立
#     wiki_search_tool = WikipediaSearch(
#         app=None,
#         agent=None,
#         wiki_dir=r"C:\Users\USER\Downloads\Test_Agent\Test_5\Dataset\FEVER\wiki-pages"
#     )
#     print("🚀 開始建置 Wikipedia Index...")   # 🔥 新增：開始訊息
#     await wiki_search_tool.build_index()       # ✅ 只建一次
#     print("✅ Wikipedia Index 構建完成！")    # 🔥 新增：完成訊息

#     app = AutoAgentSystem(
#         tool_configs={
#             WikipediaSearch: {
#                 "always_include": True,
#                 "kwargs": {
#                     "wiki_dir": wiki_search_tool.wiki_dir,
#                     "prebuilt_index": wiki_search_tool.page_index  # ✅ 共用 index
#                 }
#             }
#         },
#         root_has_tools=True,
#         max_delegation_depth=3,  # ✅ 最多遞迴 3 層
#     )

#     # ✅ 加 event logger
#     async def event_logger(event):
#         if isinstance(event, events.AgentDelegated):
#             print(f"\n🤖 子 Agent 建立：{event.child_id}")
#             print(f"📄 任務指派內容：{event.instructions}")
#         if isinstance(event, events.AgentMessage):
#             if event.msg.role == ChatRole.ASSISTANT and event.msg.tool_calls:
#                 print(f"🛠️ 子 Agent 使用工具：{event.msg.tool_calls}")

#     app.add_listener(event_logger)

#     print("\n✅ AutoAgentSystem 啟動！開始 FEVER 自動遞迴批次測試...\n")

#     num_samples = 5
#     results = []
#     total_correct = 0

#     for i, sample in enumerate(tqdm(samples[:num_samples])):
#         claim = sample["claim"]
#         ground_truth = sample["label"]

#         prompt = prepare_prompt(claim)

#         try:
#             prediction = await chat_once(app, prompt)
#         except Exception as e:
#             print(f"❌ 錯誤: {e}")
#             prediction = ""

#         prediction_label = normalize_label(prediction)

#         print(f"\n▶️ 測試第 {i+1} 題")
#         print(f"Claim: {claim}")
#         print(f"📝 回答: {prediction_label}")
#         print(f"✅ 正確答案: {ground_truth}")

#         correct = int(prediction_label == ground_truth)
#         total_correct += correct

#         results.append({
#             "claim": claim,
#             "prediction": prediction_label,
#             "ground_truth": ground_truth,
#             "correct": correct
#         })

#         app.visualizer.render(f"agent_tree_{i+1}", view=True)

#     accuracy = total_correct / num_samples

#     print("\n🎯 全部測試完成")
#     print(f"📊 Accuracy 平均: {accuracy:.4f}")

#     with open("fever_recursive_agent_test_results.json", "w", encoding="utf-8") as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)

#     df = pd.DataFrame(results)
#     df.to_csv("fever_recursive_agent_test_results.csv", index=False, encoding="utf-8-sig")

# if __name__ == "__main__":
#     asyncio.run(main())


# import json
# import asyncio
# import re
# import os
# import events
# import pandas as pd
# from tqdm import tqdm
# from runtime import ChatRole
# from dotenv import load_dotenv
# from pathlib import Path

# load_dotenv()
# import sys
# sys.path.append(".")  # 確保當前路徑可以匯入

# # 匯入系統與 Wiki 工具
# from app import AutoAgentSystem
# from tools.wiki_search import WikipediaSearch

# # =========================================================
# # ✅ 你的資料路徑（已改成你提供的新位置）
# # =========================================================
# BASE = Path(r"C:\Users\USER\Downloads\Autonomous Recursive Multi-Agent System\AutoAgentSystem")
# DEV_PATH = BASE / "shared_task_dev.jsonl"
# WIKI_DIR = BASE / "wiki-pages"

# # --- 路徑檢查（避免跑一半才爆）
# if not DEV_PATH.exists():
#     raise FileNotFoundError(f"找不到 FEVER dev 檔案：{DEV_PATH}")

# if not WIKI_DIR.exists() or not WIKI_DIR.is_dir():
#     raise FileNotFoundError(f"找不到 wiki-pages 資料夾：{WIKI_DIR}")

# wiki_files = list(WIKI_DIR.glob("wiki-*.jsonl"))
# if len(wiki_files) == 0:
#     raise FileNotFoundError(f"wiki-pages 裡沒有找到 wiki-*.jsonl 檔案，請確認你有解壓到：{WIKI_DIR}")

# # --- FEVER 資料 ---
# with open(DEV_PATH, "r", encoding="utf-8") as f:
#     samples = [json.loads(line) for line in f]

# # =========================================================
# # ✅ 改良版 Prompt（更容易觸發遞迴）
# # =========================================================
# def prepare_prompt(claim: str) -> str:
#     prompt = (
#         "你是一個 Chief Autonomous Agent，負責判斷以下的事實陳述是否正確：\n\n"
#         f"【Claim】\n{claim}\n\n"
#         "你的工作步驟如下：\n"
#         "1. 仔細閱讀 Claim，拆分出主要的實體（entities）和關鍵概念（concepts）。\n"
#         "2. 你必須至少建立 3 個子助理（3 個子 Agent），每個子助理負責查不同實體或關鍵詞。\n"
#         "   若子助理回報資訊不足，你必須再建立『第二層』子助理去補查（最多遞迴 2 層）。\n"
#         "3. 指派子助理時，**只允許使用 WikipediaSearch 工具**搜尋證據。\n"
#         "4. ⚠️ **禁止使用未授權的工具或外部資源，例如 Google Search、Browsing。**\n"
#         "5. 每個子助理必須回報：\n"
#         "   - 是否找到證據？（找到或未找到）\n"
#         "   - 找到的證據內容（引用原句）。\n"
#         "   - 證據是支持還是反駁？（SUPPORTS / REFUTES）\n"
#         "6. 如果沒有找到任何證據，子助理必須回報：NOT ENOUGH INFO。\n"
#         "7. ⚠️ **嚴格要求：子助理回報時不可猜測、不可創造內容。**\n\n"
#         "你的最終任務是，整合所有子助理的回報，根據以下規則做出最終判斷：\n"
#         "- 只要有子助理回報 REFUTES，最終結果是 REFUTES。\n"
#         "- 如果沒有 REFUTES，但有 SUPPORTS，最終結果是 SUPPORTS。\n"
#         "- 如果所有子助理都是 NOT ENOUGH INFO，最終結果是 NOT ENOUGH INFO。\n\n"
#         "⚠️ 嚴格要求：只能根據子助理的證據回報做判斷，不可推測。\n\n"
#         "請你最後直接輸出一個詞（只允許輸出 SUPPORTS / REFUTES / NOT ENOUGH INFO）。\n"
#         "❌ 不要加句子，不要加解釋，不要加任何其他內容，只能輸出單一詞彙。"
#     )
#     return prompt


# # --- LLM 回覆標準化 ---
# def normalize_label(text: str) -> str:
#     text = (text or "").lower()
#     if "support" in text:
#         return "SUPPORTS"
#     elif "refute" in text:
#         return "REFUTES"
#     elif "not enough info" in text or "not enough information" in text:
#         return "NOT ENOUGH INFO"
#     else:
#         return "UNKNOWN"


# # --- LLM 單次測試 ---
# async def chat_once(agent_system, user_input: str) -> str:
#     await agent_system.ensure_init()
#     response_text = ""
#     async for stream_manager in agent_system.root_agent.full_round_stream(user_input):
#         message = await stream_manager.message()
#         if message.role == ChatRole.ASSISTANT:
#             response_text += message.content or ""
#     return response_text.strip()


# # =========================================================
# # ✅ 遞迴觀察版 Logger（避免 search_sentence 洗版）
# # =========================================================
# SEARCH_SENTENCE_PRINT_LIMIT = 8
# _search_sentence_count = 0

# async def recursive_debug_logger(event):
#     global _search_sentence_count

#     # 1) 遞迴：建立子 agent
#     if isinstance(event, events.AgentDelegated):
#         print("\n" + "=" * 90)
#         print(f"🤖 [DELEGATE] parent={getattr(event, 'parent_id', '?')} -> child={event.child_id}")
#         print(f"👤 who={getattr(event, 'who', '?')}")
#         print("📌 instructions:")
#         print(event.instructions)
#         print("=" * 90 + "\n")
#         return

#     # 2) agent 訊息：assistant 回覆 + tool calls
#     if isinstance(event, events.AgentMessage):
#         msg = event.msg
#         role = msg.role
#         content = (msg.content or "").strip()

#         if role == ChatRole.ASSISTANT and content:
#             print(f"\n🗣️ [ASSISTANT] {content[:700]}")
#             if len(content) > 700:
#                 print("...(truncated)")

#         if msg.tool_calls:
#             for tc in msg.tool_calls:
#                 s = str(tc)
#                 # 限制 search_sentence 洗版
#                 if "search_sentence" in s:
#                     _search_sentence_count += 1
#                     if _search_sentence_count <= SEARCH_SENTENCE_PRINT_LIMIT:
#                         print(f"\n🛠️ [TOOL CALL search_sentence #{_search_sentence_count}] {s}")
#                     elif _search_sentence_count == SEARCH_SENTENCE_PRINT_LIMIT + 1:
#                         print("\n🛠️ search_sentence 太多，後面省略...(你已經知道它在掃句子)")
#                 else:
#                     print(f"\n🛠️ [TOOL CALL] {s}")
#         return

#     # 3) 如果 events 裡有 ToolResult 類，就印工具回傳（有些版本沒有）
#     ToolResultCls = getattr(events, "ToolResult", None)
#     if ToolResultCls is not None and isinstance(event, ToolResultCls):
#         result = getattr(event, "result", None)
#         print("\n📦 [TOOL RESULT]")
#         print(str(result)[:1200])
#         if result and len(str(result)) > 1200:
#             print("...(truncated)")


# # --- 主程式 ---
# async def main():
#     # ✅ 預先初始化 WikipediaSearch，避免重複建立
#     wiki_search_tool = WikipediaSearch(
#         app=None,
#         agent=None,
#         wiki_dir=str(WIKI_DIR)
#     )

#     print("🚀 開始建置 Wikipedia Index...")
#     await wiki_search_tool.build_index()  # ✅ 只建一次
#     print("✅ Wikipedia Index 構建完成！")

#     app = AutoAgentSystem(
#         tool_configs={
#             WikipediaSearch: {
#                 "always_include": True,
#                 "kwargs": {
#                     "wiki_dir": wiki_search_tool.wiki_dir,
#                     "prebuilt_index": wiki_search_tool.page_index  # ✅ 共用 index
#                 }
#             }
#         },
#         root_has_tools=True,
#         max_delegation_depth=5,  # ✅ 提高深度：更容易看到遞迴
#     )

#     # ✅ 加入遞迴觀察 logger
#     app.add_listener(recursive_debug_logger)

#     print("\n✅ AutoAgentSystem 啟動！開始『遞迴行為觀察』測試...\n")

#     # ✅ 只跑 1 題：乾淨觀察遞迴
#     num_samples = 1
#     results = []

#     for i, sample in enumerate(tqdm(samples[:num_samples])):
#         claim = sample["claim"]
#         ground_truth = sample.get("label", "UNKNOWN")

#         prompt = prepare_prompt(claim)

#         try:
#             prediction = await chat_once(app, prompt)
#         except Exception as e:
#             print(f"❌ 錯誤: {e}")
#             prediction = ""

#         prediction_label = normalize_label(prediction)

#         print("\n" + "#" * 90)
#         print(f"▶️ 測試第 {i+1} 題（遞迴觀察）")
#         print(f"Claim: {claim}")
#         print(f"📝 Model Output Raw: {prediction}")
#         print(f"📝 Prediction Label: {prediction_label}")
#         print(f"✅ Ground Truth: {ground_truth}")
#         print("#" * 90 + "\n")

#         results.append({
#             "claim": claim,
#             "prediction_raw": prediction,
#             "prediction": prediction_label,
#             "ground_truth": ground_truth,
#         })

#         # ✅ 存 tree 圖（不要一直跳視窗）
#         app.visualizer.render("agent_tree_debug", view=False)
#         print("✅ agent tree saved: agent_tree_debug.png")

#     with open("fever_recursive_agent_debug.json", "w", encoding="utf-8") as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)

#     print("\n🎯 遞迴觀察測試完成！")
#     print("📄 輸出：fever_recursive_agent_debug.json / agent_tree_debug.png")


# if __name__ == "__main__":
#     asyncio.run(main())

import json
import asyncio
import re
import os
import time
from collections import defaultdict
import events
import pandas as pd
from tqdm import tqdm
from runtime import ChatRole
from dotenv import load_dotenv
from pathlib import Path

# ✅ 需要 Pillow 來把文字畫到圖片下方
from PIL import Image, ImageDraw, ImageFont

load_dotenv()
import sys
sys.path.append(".")  # 確保當前路徑可以匯入

# 匯入系統與 Wiki 工具
from app import AutoAgentSystem
from tools.wiki_search import WikipediaSearch


# =========================================================
# ✅ 你的資料路徑（已改成你提供的新位置）
# =========================================================
BASE = Path(r"C:\Users\USER\Downloads\Autonomous Recursive Multi-Agent System\AutoAgentSystem")
DEV_PATH = BASE / "shared_task_dev.jsonl"
WIKI_DIR = BASE / "wiki-pages"

# --- 路徑檢查（避免跑一半才爆）
if not DEV_PATH.exists():
    raise FileNotFoundError(f"找不到 FEVER dev 檔案：{DEV_PATH}")

if not WIKI_DIR.exists() or not WIKI_DIR.is_dir():
    raise FileNotFoundError(f"找不到 wiki-pages 資料夾：{WIKI_DIR}")

wiki_files = list(WIKI_DIR.glob("wiki-*.jsonl"))
if len(wiki_files) == 0:
    raise FileNotFoundError(f"wiki-pages 裡沒有找到 wiki-*.jsonl 檔案，請確認你有解壓到：{WIKI_DIR}")

# --- FEVER 資料 ---
with open(DEV_PATH, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]


# =========================================================
# ✅ Prompt（更容易觸發遞迴）
# =========================================================
def prepare_prompt(claim: str) -> str:
    prompt = (
        "你是一個 Chief Autonomous Agent，負責判斷以下的事實陳述是否正確：\n\n"
        f"【Claim】\n{claim}\n\n"
        "你的工作步驟如下：\n"
        "1. 拆分 Claim 的主要 entities / concepts。\n"
        "2. 你必須至少建立 3 個子助理（3 個子 Agent），每個子助理負責查不同實體或關鍵詞。\n"
        "   若子助理回報資訊不足，你必須再建立『第二層』子助理去補查（最多遞迴 2 層）。\n"
        "3. 指派子助理時，**只允許使用 WikipediaSearch 工具**。\n"
        "4. 禁止使用 Google Search / Browsing 等外部資源。\n"
        "5. 子助理回報需包含：證據原句 + SUPPORTS/REFUTES 或 NOT ENOUGH INFO。\n\n"
        "最終請只輸出一個詞：SUPPORTS / REFUTES / NOT ENOUGH INFO。\n"
        "❌ 不要輸出任何解釋或多餘文字。"
    )
    return prompt


def normalize_label(text: str) -> str:
    text = (text or "").lower()
    if "support" in text:
        return "SUPPORTS"
    elif "refute" in text:
        return "REFUTES"
    elif "not enough info" in text or "not enough information" in text:
        return "NOT ENOUGH INFO"
    else:
        return "UNKNOWN"


async def chat_once(agent_system, user_input: str) -> str:
    await agent_system.ensure_init()
    response_text = ""
    async for stream_manager in agent_system.root_agent.full_round_stream(user_input):
        message = await stream_manager.message()
        if message.role == ChatRole.ASSISTANT:
            response_text += message.content or ""
    return response_text.strip()


# =========================================================
# ✅ Agent 協作表收集器
# =========================================================
agent_rows = defaultdict(lambda: {
    "agent_id": "",           # UUID
    "display_name": "",       # alpha/beta/gamma/...
    "parent_id": "",
    "instructions": "",
    "tool_calls": 0,
    "tools_used": set(),
    "search_queries": [],
    "visited_urls": [],
    "assistant_messages": [],
    "start_ts": None,
    "end_ts": None,
})

def _ensure_agent(agent_uuid: str):
    row = agent_rows[agent_uuid]
    if not row["agent_id"]:
        row["agent_id"] = agent_uuid
    if row["start_ts"] is None:
        row["start_ts"] = time.time()
    return row

def _finalize_agent(agent_uuid: str):
    row = agent_rows.get(agent_uuid)
    if row and row["end_ts"] is None:
        row["end_ts"] = time.time()


# =========================================================
# ✅ UUID -> 顯示名稱（alpha/beta/...）映射
#    這個就是修「圖是 alpha 但文字不是」的關鍵
# =========================================================
agent_name_map = {}  # {uuid: "alpha"}
parent_child_order = defaultdict(int)

FALLBACK_NAMES = [
    "root",
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"
]

def get_or_assign_display_name(agent_uuid: str, parent_uuid: str = "") -> str:
    if not agent_uuid:
        return "UNKNOWN"
    if agent_uuid in agent_name_map:
        return agent_name_map[agent_uuid]

    # 根節點保底叫 root（如果你的系統本來就是 root）
    if not parent_uuid:
        agent_name_map[agent_uuid] = "root"
        return "root"

    parent_child_order[parent_uuid] += 1
    idx = parent_child_order[parent_uuid]  # 1,2,3...
    name = FALLBACK_NAMES[idx] if idx < len(FALLBACK_NAMES) else f"agent_{idx}"
    agent_name_map[agent_uuid] = name
    return name

def disp(uuid_: str) -> str:
    return agent_name_map.get(uuid_, uuid_)


# =========================================================
# ✅ 嘗試從 event 抓出 agent 的 UUID + name（不同版本 events 結構不一樣）
# =========================================================
def _extract_agent_uuid_and_name(event):
    """
    回傳 (agent_uuid, agent_display_name)
    """
    agent = getattr(event, "agent", None) or getattr(event, "sender", None) or getattr(event, "source", None)

    agent_uuid = None
    agent_name = None

    if agent is not None:
        agent_uuid = getattr(agent, "id", None) or getattr(agent, "agent_id", None)
        agent_name = getattr(agent, "name", None) or getattr(agent, "display_name", None)

    # fallback：某些版本直接掛在 event 上
    if agent_uuid is None:
        agent_uuid = getattr(event, "agent_id", None) or getattr(event, "id", None)

    return agent_uuid, agent_name


# =========================================================
# ✅ Logger（避免 search_sentence 洗版 + 收集協作表）
# =========================================================
SEARCH_SENTENCE_PRINT_LIMIT = 8
_search_sentence_count = 0

async def recursive_debug_logger(event):
    global _search_sentence_count

    # 1) delegate 事件：最穩定可以拿到 parent_id / child_id / instructions
    if isinstance(event, events.AgentDelegated):
        parent = getattr(event, "parent_id", "") or ""
        child = getattr(event, "child_id", None) or getattr(event, "child", None) or getattr(event, "id", None)
        # 你的版本是 event.child_id
        child = getattr(event, "child_id", child)

        instr = getattr(event, "instructions", "") or ""

        # ✅ 確保 name map 有值（如果 event 有帶 child_name 就吃，沒有就照順序分 alpha/beta）
        child_name = getattr(event, "child_name", None) or getattr(event, "name", None)
        if child_name:
            agent_name_map[child] = child_name
        else:
            get_or_assign_display_name(child, parent_uuid=parent)

        # 紀錄 row
        r = _ensure_agent(child)
        r["parent_id"] = parent
        r["instructions"] = instr
        r["display_name"] = disp(child)

        print("\n" + "=" * 90)
        print(f"🤖 [DELEGATE] parent={parent} ({disp(parent)}) -> child={child} ({disp(child)})")
        print("📌 instructions:")
        print(instr)
        print("=" * 90 + "\n")
        return

    # 2) message 事件：抓 UUID + name，才能把 assistant/tool_calls 記到正確 agent
    if isinstance(event, events.AgentMessage):
        msg = event.msg
        role = msg.role
        content = (msg.content or "").strip()

        agent_uuid, agent_display = _extract_agent_uuid_and_name(event)
        agent_uuid = agent_uuid or "UNKNOWN_AGENT"

        # 若 events 有提供顯示名，就存起來；不然用既有/fallback
        if agent_display:
            agent_name_map[agent_uuid] = agent_display
        else:
            # 如果之前 delegate 有分配過就用；沒有就先保底（不亂分，避免跟 tree 不一致）
            if agent_uuid not in agent_name_map:
                agent_name_map[agent_uuid] = agent_uuid  # 先不猜

        # assistant message
        if role == ChatRole.ASSISTANT and content:
            r = _ensure_agent(agent_uuid)
            r["display_name"] = disp(agent_uuid)
            r["assistant_messages"].append(content)
            _finalize_agent(agent_uuid)

            print(f"\n🗣️ [{disp(agent_uuid)}] {content[:700]}")
            if len(content) > 700:
                print("...(truncated)")

        # tool calls
        if msg.tool_calls:
            r = _ensure_agent(agent_uuid)
            r["display_name"] = disp(agent_uuid)

            for tc in msg.tool_calls:
                s = str(tc)
                r["tool_calls"] += 1

                tool_name = None
                for name in ["delegate", "search", "visit_page", "search_sentence", "wait"]:
                    if f"name='{name}'" in s or f'name="{name}"' in s:
                        tool_name = name
                        break
                if tool_name is None:
                    tool_name = "tool_call"

                r["tools_used"].add(tool_name)

                if tool_name == "search":
                    m = re.search(r'"query"\s*:\s*"([^"]+)"', s)
                    if m:
                        r["search_queries"].append(m.group(1))
                elif tool_name == "visit_page":
                    m = re.search(r'"href"\s*:\s*"([^"]+)"', s)
                    if m:
                        r["visited_urls"].append(m.group(1))

                if "search_sentence" in s:
                    _search_sentence_count += 1
                    if _search_sentence_count <= SEARCH_SENTENCE_PRINT_LIMIT:
                        print(f"\n🛠️ [TOOL CALL search_sentence #{_search_sentence_count}] {s}")
                    elif _search_sentence_count == SEARCH_SENTENCE_PRINT_LIMIT + 1:
                        print("\n🛠️ search_sentence 太多，後面省略...(你已經知道它在掃句子)")
                else:
                    print(f"\n🛠️ [TOOL CALL] {s}")
        return


# =========================================================
# ✅ 協作表輸出（加入 display_name，且讓表更容易讀）
# =========================================================
def export_collaboration_table():
    table_rows = []
    for aid, r in agent_rows.items():
        tools_used = sorted(list(r["tools_used"])) if r["tools_used"] else []
        duration = None
        if r["start_ts"] and r["end_ts"]:
            duration = r["end_ts"] - r["start_ts"]

        last_msg = r["assistant_messages"][-1] if r["assistant_messages"] else ""
        last_msg = (last_msg[:300] + "...") if len(last_msg) > 300 else last_msg

        table_rows.append({
            "display_name": r.get("display_name") or disp(aid),
            "agent_id": r["agent_id"] or aid,
            "parent_display": disp(r["parent_id"]) if r["parent_id"] else "",
            "parent_id": r["parent_id"],
            "instructions": (r["instructions"][:220] + "...") if len(r["instructions"]) > 220 else r["instructions"],
            "tool_calls": r["tool_calls"],
            "tools_used": ", ".join(tools_used),
            "search_queries_count": len(r["search_queries"]),
            "visited_urls_count": len(r["visited_urls"]),
            "search_queries_sample": " | ".join(r["search_queries"][:3]),
            "visited_urls_sample": " | ".join(r["visited_urls"][:2]),
            "last_assistant_msg": last_msg,
            "duration_sec": round(duration, 2) if duration is not None else None,
        })

    df_agents = pd.DataFrame(table_rows)
    if not df_agents.empty:
        df_agents = df_agents.sort_values(by=["parent_id", "display_name"], ascending=[True, True])

    print("\n" + "=" * 90)
    print("🤝 Agent 協作表（Collaboration Table）")
    if df_agents.empty:
        print("(協作表為空：可能是 event 裡抓不到 agent，但不影響 delegate/工具log)")
    else:
        print(df_agents.to_string(index=False))
    print("=" * 90 + "\n")

    df_agents.to_csv("agent_collaboration_table.csv", index=False, encoding="utf-8-sig")
    with open("agent_collaboration_table.json", "w", encoding="utf-8") as f:
        json.dump(table_rows, f, ensure_ascii=False, indent=2)

    print("✅ 已輸出：agent_collaboration_table.csv / agent_collaboration_table.json")
    return df_agents


# =========================================================
# ✅ 在 tree 圖下面加上「每個 agent 的工作 + 合作方式」
#    重點：用 display_name（alpha/beta/...）顯示，並附上 UUID
# =========================================================
def build_agent_work_and_collab_text(max_agents: int = 30) -> str:
    # parent -> children
    children_map = defaultdict(list)
    for aid, r in agent_rows.items():
        parent = r.get("parent_id") or ""
        if parent and parent != "?":
            children_map[parent].append(aid)

    # 只顯示 tree 可能有的：有 parent 或有 instructions 或有 tool_calls 的
    def is_visible(aid):
        r = agent_rows[aid]
        return bool(r.get("parent_id")) or bool(r.get("instructions")) or (r.get("tool_calls", 0) > 0)

    agent_ids = [aid for aid in agent_rows.keys() if is_visible(aid)]
    agent_ids = sorted(agent_ids, key=lambda x: (0 if agent_rows[x].get("instructions") else 1, disp(x)))

    lines = []
    lines.append("Agent Collaboration Notes (Key Points)")
    lines.append("=" * 80)

    count = 0
    for aid in agent_ids:
        if count >= max_agents:
            lines.append(f"...(還有更多 agent，省略 {len(agent_ids) - max_agents} 個)")
            break

        r = agent_rows[aid]
        agent_uuid = r.get("agent_id") or aid
        name = r.get("display_name") or disp(agent_uuid)

        parent = r.get("parent_id") or ""
        instr = (r.get("instructions") or "").strip()
        tools = sorted(list(r.get("tools_used") or []))
        q_sample = (r.get("search_queries") or [])[:3]
        u_sample = (r.get("visited_urls") or [])[:2]
        last_msg = (r.get("assistant_messages") or [""])[-1].strip()
        if len(last_msg) > 220:
            last_msg = last_msg[:220] + "..."

        siblings = []
        if parent and parent in children_map:
            siblings = [x for x in children_map[parent] if x != aid]
        children = children_map.get(aid, [])

        lines.append("")
        lines.append(f"[Agent] {name}  ({agent_uuid})")

        if parent and parent != "?":
            lines.append(f"  - Collaborates with Parent: {disp(parent)} ({parent})")

        if siblings:
            sibs = [f"{disp(x)}" for x in siblings[:6]]
            lines.append(f"  - Collaborates with Siblings: {', '.join(sibs)}{'...' if len(siblings)>6 else ''}")

        if children:
            kids = [f"{disp(x)}" for x in children[:6]]
            lines.append(f"  - Delegates to Children: {', '.join(kids)}{'...' if len(children)>6 else ''}")

        if instr:
            lines.append(f"  - Task: {instr}")

        if tools:
            lines.append(f"  - Tools Used: {', '.join(tools)}")

        if q_sample:
            lines.append(f"  - Search Queries: {' | '.join(q_sample)}")

        if u_sample:
            lines.append(f"  - Visited Pages: {' | '.join(u_sample)}")

        if last_msg:
            lines.append(f"  - Last Report: {last_msg}")

        count += 1

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def annotate_tree_image(tree_png_path: str, out_png_path: str):
    img = Image.open(tree_png_path).convert("RGB")
    text = build_agent_work_and_collab_text()

    # 字型：Windows 優先用 Arial，找不到就用 default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    dummy = Image.new("RGB", (img.width, 10), "white")
    draw_dummy = ImageDraw.Draw(dummy)

    lines = text.splitlines()
    line_height = draw_dummy.textbbox((0, 0), "Ag", font=font)[3] + 6
    text_height = line_height * (len(lines) + 2)

    padding = 20
    new_img = Image.new("RGB", (img.width, img.height + text_height + padding), "white")
    new_img.paste(img, (0, 0))

    draw = ImageDraw.Draw(new_img)
    y = img.height + 10
    x = 20
    for line in lines:
        draw.text((x, y), line, fill="black", font=font)
        y += line_height

    new_img.save(out_png_path)
    print(f"✅ annotated tree saved: {out_png_path}")


# =========================================================
# ✅ 主程式
# =========================================================
async def main():
    wiki_search_tool = WikipediaSearch(
        app=None,
        agent=None,
        wiki_dir=str(WIKI_DIR)
    )

    print("🚀 開始建置 Wikipedia Index...")
    await wiki_search_tool.build_index()
    print("✅ Wikipedia Index 構建完成！")

    app = AutoAgentSystem(
        tool_configs={
            WikipediaSearch: {
                "always_include": True,
                "kwargs": {
                    "wiki_dir": wiki_search_tool.wiki_dir,
                    "prebuilt_index": wiki_search_tool.page_index
                }
            }
        },
        root_has_tools=True,
        max_delegation_depth=5,
    )

    app.add_listener(recursive_debug_logger)

    print("\n✅ AutoAgentSystem 啟動！開始『遞迴行為觀察』測試...\n")

    num_samples = 1
    results = []

    for i, sample in enumerate(tqdm(samples[:num_samples])):
        claim = sample["claim"]
        ground_truth = sample.get("label", "UNKNOWN")

        prompt = prepare_prompt(claim)

        try:
            prediction = await chat_once(app, prompt)
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            prediction = ""

        prediction_label = normalize_label(prediction)

        print("\n" + "#" * 90)
        print(f"▶️ 測試第 {i+1} 題（遞迴觀察）")
        print(f"Claim: {claim}")
        print(f"📝 Model Output Raw: {prediction}")
        print(f"📝 Prediction Label: {prediction_label}")
        print(f"✅ Ground Truth: {ground_truth}")
        print("#" * 90 + "\n")

        results.append({
            "claim": claim,
            "prediction_raw": prediction,
            "prediction": prediction_label,
            "ground_truth": ground_truth,
        })

        # ✅ 存 tree 圖（不要一直跳視窗）
        app.visualizer.render("agent_tree_debug", view=False)
        print("✅ agent tree saved: agent_tree_debug.png")

    with open("fever_recursive_agent_debug.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ✅ 輸出 agent 協作表
    export_collaboration_table()

    # ✅ 把 agent 工作/協作資訊寫到 tree 圖下面
    tree_png = "agent_tree_debug.png"
    annotated_png = "agent_tree_debug_annotated.png"
    if os.path.exists(tree_png):
        annotate_tree_image(tree_png, annotated_png)
    else:
        print(f"⚠️ 找不到 {tree_png}，所以無法產生 annotated 圖")

    print("\n🎯 遞迴觀察測試完成！")
    print("📄 輸出：fever_recursive_agent_debug.json / agent_tree_debug.png / agent_tree_debug_annotated.png")
    print("📄 協作表：agent_collaboration_table.csv / agent_collaboration_table.json")


if __name__ == "__main__":
    asyncio.run(main())
