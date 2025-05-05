### ===== communication.py =====
class Communicator:
    def __init__(self, memory):
        self.memory = memory
        self.messages = []

    def assign_tasks(self, experts):
        self.experts = experts

    def coordinate(self):
        for expert in self.experts:
            try:
                expert.act(self)
            except Exception as e:
                print(f"[警告] {expert.role} 遇到問題: {str(e)}")
                if self.should_ask_user(e):
                    answer = input("是否需要協助？(y/n)：")
                    if answer.lower() == 'y':
                        suggestion = input("請提供你的建議：")
                        self.memory.record_note("User Intervention", suggestion)
                    else:
                        self.memory.record_note("Commander Decision", "自動跳過問題繼續執行")
                else:
                    self.memory.record_note("Commander Decision", "自動處理異常")

    def submit_work(self, expert_role, result):
        self.memory.record_note(expert_role, result)

    def add_message(self, sender, content):
        self.messages.append({"sender": sender, "content": content})

    def get_recent_messages(self, exclude=None):
        return [msg for msg in self.messages if msg['sender'] != exclude]

    def should_ask_user(self, error):
        return True  # 可根據錯誤內容判斷，這裡簡化為一律詢問

