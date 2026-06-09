import json
import asyncio
import re
from runtime import ChatRole
from dotenv import load_dotenv
import os

load_dotenv()
import sys

sys.path.append(".")

# 匯入系統與工具
from app import AutoAgentSystem
from tools.browsing.impl import Browsing
from tools.pubmed import PubMedSearch
from tools.semantic import SemanticScholarSearch

# 讀 JSON
with open("unfiltered-web-dev.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

samples = dataset["Data"]

def prepare_prompt(sample):
    question = sample["Question"]
    search_results = sample["SearchResults"]

    search_context = ""
    for idx, result in enumerate(search_results):
        desc = result.get("Description", "") or ""
        search_context += f"搜尋結果 {idx+1}: {desc}\n\n"

    prompt = (
        "你是一個問題解答協調者。請仔細閱讀問題和搜尋結果，規劃合理的子任務，指派子助理協助解答。\n\n"
        "⚠️ 若子任務複雜，必須進一步拆解並委派給更多子助理協助完成。\n"
        "⚠️ 特別注意：最終回答必須簡潔，只回答正確答案本身，不要加任何解釋或額外內容。\n"
        "⚠️ 如果有本名與藝名，請選擇大眾熟知的藝名。\n"
        "⚠️ 如果問題需要推理，例如從生日計算星座，請先找出生日期再推理對應的星座。\n\n"
        f"# 問題\n{question}\n\n"
        f"# 搜尋結果\n{search_context}\n"
        "請開始你的規劃與解答："
    )
    return prompt

async def chat_once(agent_system, user_input: str) -> str:
    await agent_system.ensure_init()

    response_text = ""
    async for stream_manager in agent_system.root_agent.full_round_stream(user_input):
        message = await stream_manager.message()
        if message.role == ChatRole.ASSISTANT:
            response_text += message.content or ""

    return response_text.strip()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text

def exact_match(prediction, ground_truth_list):
    pred_norm = normalize_text(prediction)
    return any(pred_norm == normalize_text(ans) for ans in ground_truth_list)

def f1(prediction, ground_truth_list):
    pred_tokens = normalize_text(prediction).split()
    best_f1 = 0.0
    for ans in ground_truth_list:
        gt_tokens = normalize_text(ans).split()
        common = set(pred_tokens) & set(gt_tokens)
        if len(common) == 0:
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        f1_score = 2 * (precision * recall) / (precision + recall)
        best_f1 = max(best_f1, f1_score)
    return best_f1

async def main():
    app = AutoAgentSystem(
        tool_configs={
            Browsing: {"always_include": True},
            PubMedSearch: {"always_include": True},
            SemanticScholarSearch: {"always_include": True},
        },
        root_has_tools=True
    )

    print("\n✅ AutoAgentSystem 啟動！開始批次測試...\n")

    num_questions = 2
    results = []
    total_em, total_f1 = 0, 0

    for i, sample in enumerate(samples[:num_questions]):
        prompt = prepare_prompt(sample)
        answer_data = sample["Answer"]

        # ⬇️ 這裡組合 Value 和 Aliases
        answer_list = [answer_data["Value"]] + answer_data.get("Aliases", [])

        print(f"\n▶️ 測試第 {i+1} 題: {sample['Question']}")

        try:
            prediction = await chat_once(app, prompt)
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            prediction = ""

        print(f"📝 回答: {prediction}")
        print(f"✅ 正確答案: {answer_list}")

        em = exact_match(prediction, answer_list)
        f1_score = f1(prediction, answer_list)

        total_em += em
        total_f1 += f1_score

        results.append({
            "question": sample["Question"],
            "prediction": prediction,
            "ground_truth": answer_list,
            "EM": em,
            "F1": f1_score
        })
        app.visualizer.render(f"agent_tree_{i+1}", view=False) 
    avg_em = total_em / num_questions
    avg_f1 = total_f1 / num_questions

    print("\n🎯 全部測試完成")
    print(f"📊 EM (Exact Match) 平均: {avg_em:.4f}")
    print(f"📊 F1 Score 平均: {avg_f1:.4f}")

    with open("recursive_agent_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
