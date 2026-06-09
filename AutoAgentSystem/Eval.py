import sys
import asyncio
import time
import json
import re
from runtime import ChatRole  # 要有！

# 你的系統
from app import AutoAgentSystem
from tools.browsing.impl import Browsing
from tools.pubmed import PubMedSearch
from tools.semantic import SemanticScholarSearch

# 讀本地 HotpotQA 資料集
with open("hotpot_dev_distractor_v1.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 🛠️ 正確建 Prompt：把 context + 問題合成一個 User Prompt
def build_prompt(sample):
    context_text = ""
    for idx, (title, sentences) in enumerate(sample['context']):
        paragraph = f"段落 {idx+1}: 【{title}】\n" + "\n".join(sentences)
        context_text += paragraph + "\n\n"

    prompt = (
        "請根據以下背景資料回答問題，並且**僅依據資料內容推理，不要依賴外部知識**。\n\n"
        "# 背景資料\n"
        f"{context_text}\n"
        "# 問題\n"
        f"{sample['question']}\n"
        "請只回覆答案，並以最簡潔的名詞或短語形式，無需完整句子或多餘解釋。"
    )
    return prompt

# 🧠 後處理：抽取第一句短回答
def extract_first_short_answer(text):
    # 取第一個句號/句點/問號/驚嘆號前的部分
    sentence_end = re.search(r"[。.!?]", text)
    if sentence_end:
        return text[:sentence_end.start()].strip()
    else:
        return text.strip()

# ✅ 單次提問
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

    print("\n✅ AutoAgentSystem 啟動！開始批次測試 HotpotQA 問題...\n")

    num_questions = 10  # 你要測幾題
    results = []

    for i, sample in enumerate(dataset[:num_questions]):
        question = sample['question']
        ground_truth = sample['answer']

        prompt = build_prompt(sample)

        print(f"\n▶️ 測試第 {i+1} 題: {question}")

        start_time = time.time()

        try:
            raw_response = await chat_once(app, prompt)
            response = extract_first_short_answer(raw_response)  # ⬅️ 用短句處理
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            response = ""

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"📝 回答: {response}")
        print(f"✅ 正確答案: {ground_truth}")
        print(f"⏱️ 用時: {elapsed_time:.2f} 秒")

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "response": response,
            "time": elapsed_time
        })

    with open("hotpotqa_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n🎯 全部測試完成，結果已保存到 hotpotqa_test_results.json")

    evaluate(results)

# 🧠 小工具：算 Exact Match 和 F1 Score
def normalize_text(text):
    import re
    import string
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = " ".join(text.split())
    return text

def exact_match(prediction, ground_truth):
    return normalize_text(prediction) == normalize_text(ground_truth)

def f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if len(common) == 0:
        return 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * (precision * recall) / (precision + recall)

def evaluate(results):
    em_total = 0
    f1_total = 0
    n = len(results)

    for r in results:
        pred = r["response"]
        gt = r["ground_truth"]
        if pred:
            em_total += exact_match(pred, gt)
            f1_total += f1(pred, gt)

    print("\n📊 測試總結：")
    print(f"EM (Exact Match) 平均: {em_total / n:.4f}")
    print(f"F1 Score 平均: {f1_total / n:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
