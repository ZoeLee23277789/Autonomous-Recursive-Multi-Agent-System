# Autonomous Recursive Multi-Agent System

## 🧠 專案簡介
這是一個能夠「自主決策 + 自動生成協作團隊 + 必要時請示人類」的 AI Agent 系統。

透過這個系統，你只需要給出一個最終任務，主 Agent 就會自動：

- 🔍 推理出完成這個任務所需要的各種子角色（專家 Agent）
- ⚙️ 動態生成這些專家角色
- 🤝 安排他們互相溝通與協作
- 🧩 如果有問題，主 Agent 會自己判斷是否要請示人類
- ✅ 最終主動將完整解決方案回報給你

---

## 🔧 安裝說明

```bash
# 建議使用虛擬環境
conda create -n multiagent_env python=3.10
conda activate multiagent_env

# 安裝依賴
pip install -r requirements.txt
```

✅ 請在根目錄建立 `.env` 檔案，內容如下：

```
OPENAI_API_KEY=你的OpenAI API Key
```

---

## 🚀 使用方式

執行主程式：

```bash
python main.py
```

輸入任務需求，例如：
```
幫我寫一篇報導，關於LLM
```

程式會：
1. 呼叫 LLM 推理出你需要的專家角色
2. 自動建立這些專家並執行子任務
3. 專家會互相討論並完成任務
4. 如果遇到無法決策的情況，Commander 會詢問你（Human-in-the-loop）
5. 回傳最終報告

---

## 📂 專案結構

```
├── main.py                   # 程式進入點
├── commander_agent.py        # 主AI Agent，負責指揮任務與召喚專家
├── expert_factory.py         # 專家產生器，根據角色名稱建立對應的專家 Agent
├── communication.py          # 控制 Agent 間溝通、請示人類等互動流程
├── memory.py                 # 任務過程中所有筆記與成果彙整
├── requirements.txt          # 所需套件列表
├── .env                      # 儲存 OpenAI API KEY
```

---

## ✅ 本系統支援功能

| 功能 | 說明 |
|------|------|
| 🧠 任務分析 | 主 Agent 可自動分析任務並判斷所需專家 |
| ⚙️ 動態生成 Agent | 根據 LLM 推理，隨時建立新角色 |
| 🤝 Agent 協作 | 專家間可互相參考彼此資訊並整合成果 |
| 🧍 人類決策介入 | 僅在遇到需要你意見時，系統會主動詢問你 |
| 📋 結果回報 | 所有討論與最終成果由主 Agent 整理並呈現 |

---

## 📌 適合應用情境

- 自動化報告撰寫
- 求職資料整理
- 市場分析與對比
- 多步驟任務規劃
- AI 系統模擬與協作測試

---

## 🧾 License
MIT License