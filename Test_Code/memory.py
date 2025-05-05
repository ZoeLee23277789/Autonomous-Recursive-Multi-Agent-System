### ===== memory.py =====
class Memory:
    def __init__(self):
        self.notes = []

    def record_note(self, role, content):
        self.notes.append(f"{role}完成：{content}")

    def collect_all_notes(self):
        return "\n".join(self.notes)
