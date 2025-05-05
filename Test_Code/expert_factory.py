### ===== expert_factory.py =====
class ExpertFactory:
    @staticmethod
    def create_dynamic_expert(role, communicator):
        return DynamicExpert(role, communicator)

class DynamicExpert:
    def __init__(self, role, communicator):
        self.role = role
        self.communicator = communicator

    def act(self, context):
        self_output = f"[{self.role}] 初步完成了子任務。"
        context.add_message(self.role, self_output)

        others_output = context.get_recent_messages(exclude=self.role)
        if others_output:
            for msg in others_output:
                reply = f"[{self.role}] 根據 {msg['sender']} 的資訊補充了內容。"
                context.add_message(self.role, reply)

        self.communicator.submit_work(self.role, f"{self.role} 討論後的最終結果")
