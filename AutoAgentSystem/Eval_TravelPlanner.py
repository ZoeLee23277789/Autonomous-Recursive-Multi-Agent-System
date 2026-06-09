import sys
import asyncio
import time
import json
import os
from dotenv import load_dotenv
from datasets import load_dataset
from runtime import ChatRole

load_dotenv()

# 你的系統
from app import AutoAgentSystem
from tools.browsing.impl import Browsing
from tools.pubmed import PubMedSearch
from tools.semantic import SemanticScholarSearch

async def chat_once(agent_system, user_input: str) -> str:
    await agent_system.ensure_init()

    response_text = ""
    async for stream_manager in agent_system.root_agent.full_round_stream(user_input):
        message = await stream_manager.message()
        if message.role == ChatRole.ASSISTANT:
            response_text += message.content

    return response_text.strip()

async def main():
    app = AutoAgentSystem(
        tool_configs={
            Browsing: {"always_include": True},
            PubMedSearch: {"always_include": True},
            SemanticScholarSearch: {"always_include": True},
        },
        root_has_tools=True
    )

    print("\n✅ AutoAgentSystem 啟動！開始批次測試 TravelPlanner 問題...\n")

    # ⚡ 正確讀 TravelPlanner Dataset
    dataset = load_dataset('osunlp/TravelPlanner', 'validation')['validation']

    num_questions = 10  # 你要測幾題
    results = []

    for i, sample in enumerate(dataset.select(range(num_questions))):
        query = sample['query']  # TravelPlanner 問題是 query
    
        print(f"\n▶️ 測試第 {i+1} 題: {query}")
    
        start_time = time.time()
    
        try:
            response = await chat_once(app, query)
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            response = ""
    
        end_time = time.time()
        elapsed_time = end_time - start_time
    
        print(f"📝 回答: {response}")
    
        results.append({
            "query": query,
            "response": response,
            "time": elapsed_time
        })

    # 存成 json
    with open("travelplanner_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n🎯 全部測試完成，結果已保存到 travelplanner_results.json")

if __name__ == "__main__":
    asyncio.run(main())
