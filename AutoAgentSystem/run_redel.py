# run_redel.py
import sys
import asyncio

# 加入路徑，讓 Python 能找到當前資料夾
sys.path.append(".")

# 匯入系統與工具
from app import AutoAgentSystem
from tools.browsing.impl import Browsing  # ✅ 加入這行，導入你的工具

async def main():
    app = AutoAgentSystem(
        tool_configs={
            Browsing: {
                "always_include": True  # ✅ 所有 agent 都可以用此工具
            }
        },
        root_has_tools=True  # ✅ 讓 root agent 也可以用工具
    )

    print("\n✅ AutoAgentSystem 啟動！直接輸入你的總任務，Ctrl+C / exit 可退出。\n")
    try:
        await app.chat_in_terminal()
    except KeyboardInterrupt:
        print("\n👋 使用者中斷。再見！")

if __name__ == "__main__":
    asyncio.run(main())
