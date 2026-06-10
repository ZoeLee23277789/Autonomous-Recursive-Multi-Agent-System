
# Autonomous Recursive Multi-Agent System

## Project Overview

This project implements an **Autonomous Recursive Multi-Agent System** capable of:

- autonomous decision making  
- dynamically generating specialized agents  
- coordinating collaboration among agents  
- requesting human input only when necessary  

Given a single high-level task, the **main agent automatically**:

-  analyzes the task and determines what types of expert agents are required  
-  dynamically creates those specialized agents  
-  coordinates communication and collaboration between agents  
-  decides when human input is necessary (Human-in-the-loop)  
-  aggregates all results and returns a final solution  

The system supports **recursive task delegation**, meaning that sub-agents can further create their own sub-agents when necessary.

---

## Installation

It is recommended to use a virtual environment.

```bash
conda create -n multiagent_env python=3.10
conda activate multiagent_env

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key
```

---

## Running the System

Run the main interactive agent system:

```bash
python AutoAgentSystem\__main__.py
```

Example prompt:

```
Write a report about large language models.
```

The system will automatically:

1. Use an LLM to infer required expert roles
2. Dynamically generate those expert agents
3. Assign subtasks to each agent
4. Allow agents to collaborate and share information
5. Request human input if necessary
6. Produce a final integrated result

---

## OpenTelemetry Tracing

If you installed with `requirements.txt`, the observability dependencies are already included. Otherwise install them with:

```bash
pip install -e ".[observability]"
```

Start Grafana and Tempo locally:

```powershell
docker compose -f docker-compose.observability.yml up -d
```

Then enable the built-in OTLP exporter before running the system:

```powershell
$env:AUTO_AGENT_OTEL_ENABLED="true"
$env:OTEL_SERVICE_NAME="auto-agent-system"
$env:OTEL_EXPORTER_OTLP_ENDPOINT="localhost:4317"
$env:OTEL_EXPORTER_OTLP_INSECURE="true"
python AutoAgentSystem\__main__.py
```

Each user request becomes one trace tree. Agent spans are linked by delegation parent/child relationships, and token usage is recorded as `llm.usage.prompt_tokens`, `llm.usage.completion_tokens`, and `llm.usage.total_tokens`.

Open Grafana at http://localhost:3000, choose **Explore**, select the **Tempo** datasource, and search for traces from the `auto-agent-system` service. Each child agent appears as its own span; span duration shows latency, and token usage appears in the span attributes.

---

## Project Structure

```
├── AutoAgentSystem/
│   ├── __main__.py           # Terminal entry point
│   ├── app.py                # Session orchestration, logging, visualization
│   ├── agents.py             # Root/delegate agent prompts and creation
│   ├── runtime.py            # Chat runtime and tool-call loop
│   ├── eventlogger.py        # JSONL logging and OpenTelemetry tracing
│   ├── task_planner.py       # LLM topology advisor
│   ├── delegation/           # Delegate/wait tools
│   └── tools/browsing/       # Optional web browsing tools
├── observability/            # Grafana/Tempo provisioning
├── docker-compose.observability.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## System Features

| Feature                   | Description                                                   |
| ------------------------- | ------------------------------------------------------------- |
| 🧠 Task Analysis          | The main agent analyzes tasks and determines required experts |
| ⚙️ Dynamic Agent Creation | Expert agents are generated dynamically using LLM reasoning   |
| 🤝 Agent Collaboration    | Agents communicate and integrate their results                |
| 🧍 Human-in-the-loop      | Human input is requested only when necessary                  |
| 📋 Result Aggregation     | The main agent summarizes all discussions into a final output |

---

## Potential Applications

* automated report generation
* research assistance
* job information aggregation
* market analysis
* multi-step task planning
* evaluation of collaborative AI systems
