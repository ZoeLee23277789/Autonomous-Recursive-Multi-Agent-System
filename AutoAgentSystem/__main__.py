import sys
import asyncio

# 讓 Python 能找到當前目錄（Test_5），以便正確 import app.py 中的系統類別
sys.path.append(".")

from app import AutoAgentSystem
from eventlogger import otel_status_summary

async def main():
    app = AutoAgentSystem()
    print("\n✅ AutoAgentSystem 啟動！直接輸入你的總任務，Ctrl+C / exit 可退出。\n")
    print(f"📈 OpenTelemetry: {otel_status_summary()}\n")
    try:
        await app.chat_in_terminal()
    except KeyboardInterrupt:
        print("\n👋 使用者中斷。再見！")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 使用者中斷。再見！")
