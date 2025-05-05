# commander_agent.py
import openai
from dotenv import load_dotenv
import os
from expert_factory import ExpertFactory
from communication import Communicator
from memory import Memory

class Commander:
    def __init__(self):
        load_dotenv()  # ⭐ 讀取 .env
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key)  # ⭐ 使用讀到的 API KEY
        self.memory = Memory()
        self.communicator = Communicator(self.memory)

    def receive_goal(self, goal):
        print(f"\n[Commander] 收到任務：{goal}")
        self.analyze_and_create_experts(goal)

    def analyze_and_create_experts(self, goal):
        print("[Commander] 使用 LLM 推理需要哪些專家...\n")
        expert_roles = self.llm_determine_experts(goal)
        experts = [ExpertFactory.create_dynamic_expert(role, self.communicator) for role in expert_roles]
        self.communicator.assign_tasks(experts)
        self.communicator.coordinate()
        self.final_report()

    def llm_determine_experts(self, goal):
        prompt = f"""
你是一位任務規劃師。根據下列任務，列出需要的專家角色，每個角色一句話描述其負責的工作：
任務：「{goal}」
請以以下格式回答：
角色名稱：職責描述
角色名稱：職責描述
..."""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是專業的AI任務規劃助手。"},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content

        expert_list = []
        for line in content.strip().split("\n"):
            if ":" in line:
                role, _ = line.split(":", 1)
                expert_list.append(role.strip())

        return expert_list

    def final_report(self):
        print("\n[Commander] 任務完成，以下是總結：")
        print(self.memory.collect_all_notes())
