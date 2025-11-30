# NAT Agent Creation System: Complete Project Overview

**Last Updated**: December 2024
**Status**: Phase B - RL Training Ready
**Repository**: https://github.com/athreesh/AgentHands

---

## Executive Summary

This project trains **AgentHands** (a coding agent) to become an expert at creating complete NAT (NeMo Agent Toolkit) agents through Reinforcement Learning.

**Key Innovation**: We use RL to train AgentHands to excel at the complete workflow—tool generation, integration, and validation—while keeping Gemini for cheap, effective planning.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    USER REQUEST                                 │
│  "I want a financial research agent..."                        │
└───────────────────────┬────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────────────┐
│              GEMINI 2.5 PRO PLANNER                             │
│              (Not RL trained - stays cheap)                     │
│                                                                 │
│  Analyzes user intent and creates plan:                        │
│    • Scaffold selection (react_agent, tool_calling_agent)      │
│    • MCP servers from smithery.ai                              │
│    • Custom tools specifications                               │
│    • Test cases for validation                                 │
│                                                                 │
│  Output: Complete AgentPlan                                    │
└───────────────────────┬────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────────────┐
│              AGENTHANDS AGENT ⭐                                │
│              (RL Trained with SkyRL)                            │
│                                                                 │
│  Base LLM: Qwen2.5-Coder-32B-Instruct                         │
│  Agent Type: CodeActAgent (multi-turn coding)                  │
│  Training: GRPO with SkyRL framework                           │
│                                                                 │
│  Creates complete NAT agents:                                   │
│    1. Generate all NAT tools (following patterns exactly)      │
│    2. Write comprehensive test suites                          │
│    3. Create YAML configuration files                          │
│    4. Set up MCP server integration scripts                    │
│    5. Validate end-to-end workflows                            │
│                                                                 │
│  Improves through RL training! 🚀                              │
└───────────────────────┬────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────────────┐
│              SKYRL TRAINING LOOP                                │
│                                                                 │
│  For each training episode:                                     │
│    1. Sample NAT agent task from dataset                       │
│    2. AgentHands attempts to create complete agent            │
│    3. Validate:                                                 │
│       - Tools created and NAT compliant?                       │
│       - YAML config valid?                                     │
│       - MCP setup correct?                                     │
│       - End-to-end tests pass?                                 │
│    4. Calculate reward (0-2.0 scale)                           │
│    5. Update AgentHands policy via GRPO                        │
│                                                                 │
│  Result: AgentHands becomes NAT specialist                     │
└───────────────────────┬────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────────────┐
│              COMPLETE NAT AGENT                                 │
│                                                                 │
│  Deliverables:                                                  │
│    ✅ All NAT tools (nat_tools/*.py)                           │
│    ✅ All tests (tests/test_*.py)                              │
│    ✅ Agent config (agent_config.yml)                          │
│    ✅ MCP setup (setup_mcp.sh)                                 │
│    ✅ End-to-end validation passed                             │
│                                                                 │
│  Ready to deploy to end users! 🎉                             │
└────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Gemini Planner (Phase A - Complete)

**File**: `gemini_planner.py`

**Purpose**: Creates comprehensive plans for NAT agents

**Input**: User request (natural language)

**Output**: `AgentPlan` containing:
- Scaffold type with reasoning
- MCP servers to use
- Custom tools to generate
- Test cases for validation
- YAML config template

**Status**: ✅ Complete, production-ready

**Cost**: ~$0.01-0.05 per plan (very cheap)

### 2. MCP Registry (Phase A - Complete)

**File**: `mcp_registry.py`

**Purpose**: Catalog of available MCP servers from smithery.ai

**Categories**:
- Financial: Yahoo Finance, Financial Modeling Prep
- Research: Exa Search, Linkup
- Data: Google Sheets, Airtable
- Code: GitHub, GitLab

**Status**: ✅ Complete

### 3. AgentHands Agent (Phase B - RL Training)

**What it is**: A coding agent (fork of OpenHands) powered by an LLM

**Base Model**: Qwen2.5-Coder-32B-Instruct

**What it does**:
- Multi-turn coding (create files, run commands, debug)
- Generates NAT-compliant tools
- Creates complete agent deliverables
- Validates everything works

**Training**: RL with SkyRL to specialize in NAT creation

**Status**: 🔄 Ready for RL training

### 4. SkyRL Environment (Phase B - New)

**File**: `skyrl_integration/nat_agent_env.py`

**Purpose**: Defines the RL environment for training AgentHands

**Episode Structure**:
- **State**: Agent spec from Gemini (scaffold, tools, tests)
- **Actions**: AgentHands multi-turn coding
- **Reward**: Success at creating complete working agent
- **Done**: When agent signals completion or max turns reached

**Reward Function**:
```
reward = 0.30 * tool_generation_score +
         0.30 * integration_score +
         0.40 * workflow_score +
         bonuses - penalties

Bonuses:
  +1.0 for complete working agent
  +0.3 for efficiency (fewer turns)

Penalties:
  -0.5 for missing tools or config
```

**Status**: 🔄 Implementation in progress

---

## Training Workflow

### Dataset Preparation

**Script**: `skyrl_integration/prepare_dataset.py`

**Process**:
1. Collect user requests (examples + synthetic)
2. Use Gemini to create agent plans for each
3. Extract specifications for RL training
4. Save as parquet files (train/val split)

**Output**:
- `train.parquet` (900 agent specs)
- `val.parquet` (100 agent specs)

### RL Training

**Framework**: SkyRL (Berkeley Sky Computing Lab)

**Algorithm**: GRPO (Group Relative Policy Optimization)

**Training Configuration**:
```bash
Model: Qwen2.5-Coder-32B-Instruct
Epochs: 100
Batch size: 128
Learning rate: 5e-7
GPUs: 8x H100
Training time: ~1-2 days
```

**Training Loop**:
```
For each epoch:
  For each batch:
    1. Sample 128 agent specs
    2. AgentHands creates agents (in parallel)
    3. Validate each agent
    4. Calculate rewards
    5. Update policy via GRPO

  Every 10 epochs:
    - Save checkpoint
    - Evaluate on validation set
    - Log metrics to WandB
```

### Expected Improvements

| Metric | Pre-Training | After 100 Epochs |
|--------|-------------|------------------|
| Success Rate | ~45% | ~85% |
| Tool Generation | 60% | 95% |
| Integration | 40% | 90% |
| Workflow Validation | 30% | 80% |
| Avg Turns | 25 | 18 |

---

## NAT Tool Pattern

AgentHands learns to generate tools following this exact pattern:

```python
from nat.data_models.function import FunctionBaseConfig
from nat.cli.register_workflow import register_function
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from pydantic import Field

class ToolNameConfig(FunctionBaseConfig, name="tool_name"):
    """Configuration for the tool"""
    description: str = Field(
        default="Tool description",
        description="What this tool does"
    )

@register_function(config_type=ToolNameConfig)
async def tool_name(config: ToolNameConfig, builder: Builder):
    """Tool implementation"""

    async def _arun(param: InputType) -> OutputType:
        """
        Async run method - the actual tool logic

        Args:
            param: Input parameter with type hints

        Returns:
            Output with type hints
        """
        # Implementation here
        result = perform_task(param)
        return result

    # Yield FunctionInfo (required by NAT)
    yield FunctionInfo.from_fn(_arun, description=config.description)
```

**Key Requirements**:
1. Config class inheriting from `FunctionBaseConfig`
2. `@register_function` decorator
3. Async `_arun` method
4. `yield FunctionInfo.from_fn()`
5. Type hints on all parameters
6. Comprehensive docstrings
7. Error handling

---

## Complete Agent Deliverables

After training, AgentHands creates complete agents with:

### 1. NAT Tools (`nat_tools/`)

```
nat_tools/
├── tool_one.py          # NAT-compliant tool
├── tool_two.py          # Following exact pattern
└── tool_three.py        # With type hints & docs
```

### 2. Tests (`tests/`)

```
tests/
├── test_tool_one.py     # 5+ test cases
├── test_tool_two.py     # Normal + edge cases
└── test_tool_three.py   # Error scenarios
```

### 3. Agent Configuration (`agent_config.yml`)

```yaml
llm:
  model_name: gpt-4-turbo

agent:
  type: react_agent

  tools:
    # MCP servers
    - id: yahoo_finance
      type: server_tool
      server_url: "${oc.env:YAHOO_FINANCE_URL}"

    # Custom tools
    - id: tool_one
      type: tool_one
    - id: tool_two
      type: tool_two
```

### 4. MCP Setup Script (`setup_mcp.sh`)

```bash
#!/bin/bash
# Install MCP servers from smithery.ai
npx @smithery/cli install @owner/yahoo-finance --client claude
npx @smithery/cli install exa --client claude
echo "✅ MCP servers installed"
```

### 5. Validation Results

```
✅ All tools created (3/3)
✅ All tests pass (15/15)
✅ YAML config valid
✅ MCP setup complete
✅ End-to-end tests pass (4/4)
🎉 Agent ready for deployment!
```

---

## File Structure

```
openhands/integrations/nat_poc/
├── README.md                      # Quick start guide
├── RL_TRAINING.md                 # Complete training guide
├── PROJECT_OVERVIEW.md            # This file
│
├── gemini_planner.py              # Gemini 2.5 Pro planner
├── mcp_registry.py                # MCP server catalog
├── agenthands_executor.py         # AgentHands executor
├── real_openhands_executor.py     # Real OpenHands executor
│
├── skyrl_integration/             # RL training components
│   ├── nat_agent_env.py          # SkyRL environment
│   ├── prepare_dataset.py         # Dataset generation
│   ├── evaluate_model.py          # Model evaluation
│   └── deploy_trained.py          # Deployment utils
│
├── poc_simple_scenario.py         # Demo: Simple
├── poc_financial_research.py      # Demo: Complex
├── run_end_to_end.py              # Demo: E2E simulation
└── run_real_e2e.py                # Demo: Real execution
```

---

## Development Phases

### ✅ Phase A: Foundation (Complete)
- Gemini planner working
- MCP registry populated
- NAT system prompt created
- End-to-end POC successful

### 🔄 Phase B: RL Training (Current)
- SkyRL environment implemented
- Dataset preparation working
- Training configuration ready
- Ready to train

### 🔜 Phase C: Production (Next)
- Trained model deployed
- API service for agent creation
- User feedback collection
- Continuous improvement

### 🔜 Phase D: Scale (Future)
- Multi-domain specialists
- Human-in-the-loop training
- Agent marketplace
- Automated deployment

---

## Success Metrics

### Training Metrics
- **Mean Reward**: Tracks overall quality
- **Success Rate**: % of complete working agents
- **Tool Quality**: NAT pattern compliance
- **Integration Success**: YAML + MCP correctness
- **Workflow Pass Rate**: End-to-end test success

### Production Metrics
- **User Satisfaction**: Feedback ratings
- **Agent Deployment Rate**: % of agents deployed
- **First-Time Success**: Agents that work without iteration
- **Time to Deploy**: Minutes from request to working agent

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Planning | Gemini 2.5 Pro | Create agent specifications |
| Coding Agent | AgentHands (OpenHands fork) | Generate code & validate |
| Base LLM | Qwen2.5-Coder-32B | Power the agent |
| RL Framework | SkyRL | Train the agent |
| RL Algorithm | GRPO | Policy optimization |
| Training Backend | FSDP2 | Distributed training |
| Inference | vLLM | Fast generation |
| Orchestration | Ray | Distributed execution |
| Logging | WandB | Experiment tracking |
| Target Framework | NAT | Agent deployment |

---

## Key Insights

### Why This Approach Works

1. **Division of Labor**
   - Gemini: Planning (what it's good at, cheap)
   - AgentHands: Implementation (what we train, specialized)

2. **RL Advantage**
   - Learns from successes and failures
   - Discovers patterns humans might miss
   - Improves over time with more data

3. **NAT Specialization**
   - Focused on one thing (NAT agents)
   - Clear success criteria (tests pass)
   - Immediate validation (does it work?)

4. **Practical Value**
   - Delivers complete, working agents
   - Ready to deploy (not just code snippets)
   - Real end-user value

### Challenges Overcome

1. **Tool Generation Quality**
   - Solution: RL reward for NAT pattern compliance

2. **Integration Complexity**
   - Solution: Include YAML + MCP in reward function

3. **Validation Reliability**
   - Solution: Automated test suites + end-to-end checks

4. **Training Efficiency**
   - Solution: SkyRL's optimized distributed training

---

## Getting Started

### Quick Test (No Training)
```bash
cd /path/to/AgentHands
python -m openhands.integrations.nat_poc.poc_simple_scenario
```

### Full RL Training
```bash
# 1. Prepare dataset
python -m openhands.integrations.nat_poc.skyrl_integration.prepare_dataset

# 2. Train with SkyRL
cd /path/to/SkyRL/skyrl-train
bash examples/nat_agent_creation/train_agenthands.sh

# 3. Evaluate
python -m openhands.integrations.nat_poc.skyrl_integration.evaluate_model

# 4. Deploy
python -m openhands.integrations.nat_poc.skyrl_integration.deploy_trained
```

---

## Resources

- **AgentHands Repo**: https://github.com/athreesh/AgentHands
- **SkyRL Docs**: https://skyrl.readthedocs.io
- **NAT Toolkit**: https://github.com/NVIDIA/NeMo-Agent-Toolkit
- **Smithery (MCP)**: https://smithery.ai
- **Questions**: Open issue in AgentHands repo

---

**Status**: Ready for RL training! 🚀

**Last Updated**: December 2024
