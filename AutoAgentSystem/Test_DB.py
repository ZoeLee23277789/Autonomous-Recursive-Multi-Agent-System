import asyncio
import json
import nltk
from app import AutoAgentSystem
from tools.sqlite_search import SQLiteSearch

nltk.download('punkt')

def split_claim_into_phrases(claim, max_phrases=5):
    from nltk.tokenize import sent_tokenize
    phrases = sent_tokenize(claim)
    return phrases[:max_phrases]

async def evaluate_dataset(dataset_path, num_examples=2):
    app = AutoAgentSystem(
        tool_configs={
            SQLiteSearch: {
                "always_include": True,
                "kwargs": {
                    "db_path": "C:/Users/USER/Downloads/Test_Agent/Test_5/Dataset/feverous_wikiv1.db"
                }
            }
        },
        root_has_tools=True
    )
    kani = await app.ensure_init()

    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = lines[:num_examples]

    results = []

    for line in lines:
        sample = json.loads(line)
        claim = sample["claim"]
        label = sample["label"]

        claim_phrases = split_claim_into_phrases(claim, max_phrases=5)

        print(f"\n🧩 Evaluating Claim ID {sample['id']}")
        print(f"Claim: {claim}")
        print(f"Ground Truth Label: {label}")

        instructions = []
        for phrase in claim_phrases:
            instructions.append(f"請使用 search_by_text 工具，搜尋與 '{phrase}' 相關的資料，並總結找到的證據。")

        # 只會派 5 個以內 sub-agent
        for instr in instructions:
            await kani.delegator.delegate(instr)

        # 等所有 sub agent 回來
        await kani.delegator.wait(until="all")

        # 打印所有 sub agent 任務情況
        print("\n=== Sub-Agent 任務列表 ===")
        for task in app.global_task_log:
            print(f"Agent {task['agent']} 查詢: {task['task']} → 狀態: {task['status']}")

        print("=== 子 Agent 回報完成 ===\n")

    await app.close()

if __name__ == "__main__":
    dataset_path = r"C:/Users/USER/Downloads/Test_Agent/Test_5/Dataset/feverous_dev_challenges.jsonl"
    asyncio.run(evaluate_dataset(dataset_path, num_examples=2))