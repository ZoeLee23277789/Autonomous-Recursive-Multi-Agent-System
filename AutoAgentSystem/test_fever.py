import json
import asyncio
import re
import os
import events
import pandas as pd
from tqdm import tqdm
from kani import ChatRole
from dotenv import load_dotenv

load_dotenv()
import sys
sys.path.append(".")  # 確保當前路徑可以匯入

# 匯入系統與 Wiki 工具
from app import AutoAgentSystem
from tools.wiki_search import WikipediaSearch

# --- FEVER 資料 ---
with open(r"C:\Users\USER\Downloads\Test_Agent\Test_5\Dataset\FEVER\shared_task_dev.jsonl", "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]

# --- 改良版 Prompt ---
def prepare_prompt(claim):
    prompt = (
        "你是一個 Chief Autonomous Agent，負責判斷以下的事實陳述是否正確：\n\n"
        f"【Claim】\n{claim}\n\n"
        "你的工作步驟如下：\n"
        "1. 仔細閱讀 Claim，拆分出主要的實體（entities）和關鍵概念（concepts）。\n"
        "2. 指派子助理，並**只允許使用 WikipediaSearch 工具**搜尋證據。\n"
        "3. ⚠️ **禁止使用未授權的工具或外部資源，例如 Google Search、Browsing。**\n"
        "4. 每個子助理必須回報：\n"
        "   - 是否找到證據？（找到或未找到）\n"
        "   - 找到的證據內容。\n"
        "   - 證據是支持還是反駁？（SUPPORTS / REFUTES）\n"
        "5. 如果沒有找到任何證據，子助理必須回報：NOT ENOUGH INFO。\n"
        "6. ⚠️ **嚴格要求：子助理回報時不可猜測、不可創造內容。**\n\n"
        "你的最終任務是，整合所有子助理的回報，根據以下規則做出最終判斷：\n"
        "- 只要有子助理回報 REFUTES，最終結果是 REFUTES。\n"
        "- 如果沒有 REFUTES，但有 SUPPORTS，最終結果是 SUPPORTS。\n"
        "- 如果所有子助理都是 NOT ENOUGH INFO，最終結果是 NOT ENOUGH INFO。\n\n"
        "⚠️ 嚴格要求：只能根據子助理的證據回報做判斷，不可推測。\n\n"
        "請你最後直接輸出一個詞（只允許輸出 SUPPORTS / REFUTES / NOT ENOUGH INFO）。\n"
        "❌ 不要加句子，不要加解釋，不要加任何其他內容，只能輸出單一詞彙。"
    )
    return prompt



# --- LLM 回覆標準化 ---
def normalize_label(text):
    text = text.lower()
    if "support" in text:
        return "SUPPORTS"
    elif "refute" in text:
        return "REFUTES"
    elif "not enough info" in text or "not enough information" in text:
        return "NOT ENOUGH INFO"
    else:
        return "UNKNOWN"

# --- LLM 單次測試 ---
async def chat_once(agent_system, user_input: str) -> str:
    await agent_system.ensure_init()
    response_text = ""
    async for stream_manager in agent_system.root_kani.full_round_stream(user_input):
        message = await stream_manager.message()
        if message.role == ChatRole.ASSISTANT:
            response_text += message.content or ""
    return response_text.strip()

# --- 主程式 ---
async def main():
    # ✅ 預先初始化 WikipediaSearch，避免重複建立
    wiki_search_tool = WikipediaSearch(
        app=None,
        kani=None,
        wiki_dir=r"C:\Users\USER\Downloads\Test_Agent\Test_5\Dataset\FEVER\wiki-pages"
    )
    print("🚀 開始建置 Wikipedia Index...")   # 🔥 新增：開始訊息
    await wiki_search_tool.build_index()       # ✅ 只建一次
    print("✅ Wikipedia Index 構建完成！")    # 🔥 新增：完成訊息

    app = AutoAgentSystem(
        tool_configs={
            WikipediaSearch: {
                "always_include": True,
                "kwargs": {
                    "wiki_dir": wiki_search_tool.wiki_dir,
                    "prebuilt_index": wiki_search_tool.page_index  # ✅ 共用 index
                }
            }
        },
        root_has_tools=True,
        max_delegation_depth=3,  # ✅ 最多遞迴 3 層
    )

    # ✅ 加 event logger
    async def event_logger(event):
        if isinstance(event, events.KaniDelegated):
            print(f"\n🤖 子 Agent 建立：{event.child_id}")
            print(f"📄 任務指派內容：{event.instructions}")
        if isinstance(event, events.KaniMessage):
            if event.msg.role == ChatRole.ASSISTANT and event.msg.tool_calls:
                print(f"🛠️ 子 Agent 使用工具：{event.msg.tool_calls}")

    app.add_listener(event_logger)

    print("\n✅ AutoAgentSystem 啟動！開始 FEVER 自動遞迴批次測試...\n")

    num_samples = 5
    results = []
    total_correct = 0

    for i, sample in enumerate(tqdm(samples[:num_samples])):
        claim = sample["claim"]
        ground_truth = sample["label"]

        prompt = prepare_prompt(claim)

        try:
            prediction = await chat_once(app, prompt)
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            prediction = ""

        prediction_label = normalize_label(prediction)

        print(f"\n▶️ 測試第 {i+1} 題")
        print(f"Claim: {claim}")
        print(f"📝 回答: {prediction_label}")
        print(f"✅ 正確答案: {ground_truth}")

        correct = int(prediction_label == ground_truth)
        total_correct += correct

        results.append({
            "claim": claim,
            "prediction": prediction_label,
            "ground_truth": ground_truth,
            "correct": correct
        })

        app.visualizer.render(f"agent_tree_{i+1}", view=True)

    accuracy = total_correct / num_samples

    print("\n🎯 全部測試完成")
    print(f"📊 Accuracy 平均: {accuracy:.4f}")

    with open("fever_recursive_agent_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(results)
    df.to_csv("fever_recursive_agent_test_results.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    asyncio.run(main())
