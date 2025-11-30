# Software Architecture & Requirements Document
## RL Training for AgentHands NAT Agent Creation

**Document Version**: 1.0
**Date**: December 2024
**Status**: Design Phase
**Estimated Timeline**: 6-8 weeks
**Team Size**: 2-3 engineers

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Hardware Requirements](#hardware-requirements)
4. [Model Selection & Compatibility](#model-selection--compatibility)
5. [Baseline Evaluation](#baseline-evaluation)
6. [Evaluation Harness](#evaluation-harness)
7. [Training Infrastructure](#training-infrastructure)
8. [Implementation Phases](#implementation-phases)
9. [Testing & Validation](#testing--validation)
10. [Deployment Architecture](#deployment-architecture)
11. [Risk Mitigation](#risk-mitigation)
12. [Success Metrics](#success-metrics)

---

## Executive Summary

### Objective
Train AgentHands (a coding agent) using Reinforcement Learning to become an expert at creating complete NAT (NeMo Agent Toolkit) agents from natural language descriptions.

### Key Innovation
- **Gemini 2.5 Pro**: Handles planning (cheap, stays unchanged)
- **AgentHands (RL-trained)**: Creates complete NAT agents (tools + integration + validation)
- **SkyRL Framework**: Provides scalable RL training infrastructure

### Expected Outcomes
- **Baseline** (Vanilla Qwen2.5-Coder-32B): ~45% success rate
- **After RL Training**: ~85% success rate
- **Training Time**: 1-2 days on 8xH100 for 100 epochs
- **Cost**: ~$2,000-3,000 for full training run

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING INFRASTRUCTURE                       │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │   Ray Cluster  │  │  vLLM Engines  │  │  FSDP2 Trainer │   │
│  │   (8 nodes)    │  │  (8 instances) │  │  (8 GPUs)      │   │
│  └────────────────┘  └────────────────┘  └────────────────┘   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              SkyRL Orchestrator                         │    │
│  │  • Episode generation                                   │    │
│  │  • Reward calculation                                   │    │
│  │  • Policy updates                                       │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING LOOP (per epoch)                    │
│                                                                  │
│  For each batch (128 agent specs):                              │
│    1. Sample specs from dataset                                 │
│    2. Generate prompts for AgentHands                          │
│    3. AgentHands creates agents (parallel)                     │
│    4. Validate each agent                                       │
│    5. Calculate rewards                                         │
│    6. Update policy via GRPO                                    │
│                                                                  │
│  Every 10 epochs:                                               │
│    - Save checkpoint                                            │
│    - Run validation set                                         │
│    - Log to WandB                                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    EPISODE STRUCTURE                             │
│                                                                  │
│  Episode = Create one complete NAT agent                        │
│                                                                  │
│  Input:                                                          │
│    • User request                                               │
│    • Gemini plan (scaffold, MCP, tools)                        │
│                                                                  │
│  AgentHands Actions (multi-turn):                              │
│    • Create tool files                                          │
│    • Write tests                                                │
│    • Create YAML config                                         │
│    • Set up MCP integration                                     │
│    • Run validation                                             │
│                                                                  │
│  Output:                                                         │
│    • Complete NAT agent                                         │
│    • Validation results                                         │
│    • Reward signal                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Component Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                        COMPONENTS                               │
│                                                                 │
│  Dataset Generation                                             │
│  ├── Gemini Planner API                                        │
│  ├── User Request Library                                      │
│  └── Parquet Writer                                            │
│                                                                 │
│  SkyRL Environment                                              │
│  ├── NATAgentCreationEnv                                       │
│  ├── Prompt Generator                                          │
│  ├── Validator                                                 │
│  └── Reward Calculator                                         │
│                                                                 │
│  Training Infrastructure                                        │
│  ├── Ray Cluster Manager                                       │
│  ├── vLLM Inference Engines                                    │
│  ├── FSDP2 Trainer                                             │
│  └── Checkpoint Manager                                        │
│                                                                 │
│  Evaluation                                                     │
│  ├── Baseline Evaluator                                        │
│  ├── Validation Set Runner                                     │
│  └── Metrics Collector                                         │
│                                                                 │
│  Deployment                                                     │
│  ├── Model Loader                                              │
│  ├── Inference API                                             │
│  └── Production Monitor                                        │
└────────────────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

### Training Infrastructure

#### Minimum Configuration (Small-scale testing)
```yaml
purpose: Testing and development
nodes: 1
gpus_per_node: 8
gpu_type: A100 80GB or H100
cpu_cores: 64+
ram: 512GB+
disk: 2TB NVMe SSD
network: 100Gbps InfiniBand
estimated_cost: $50-100/hour (cloud)
```

#### Recommended Configuration (Full training)
```yaml
purpose: Production RL training
nodes: 2
gpus_per_node: 8
gpu_type: H100 80GB
cpu_cores: 128+ per node
ram: 1TB+ per node
disk: 4TB NVMe SSD per node
network: 200Gbps InfiniBand
estimated_cost: $200-300/hour (cloud)
training_time: 24-48 hours for 100 epochs
total_cost: $4,800-14,400 per full run
```

#### Hardware Justification

**Why 8xH100?**
- **Model Size**: Qwen2.5-Coder-32B requires ~64GB VRAM
- **FSDP2 Sharding**: Splits model across 8 GPUs (8-10GB per GPU)
- **Batch Size**: Need large batches (128) for GRPO stability
- **vLLM Inference**: Parallel generation of 4 samples per prompt
- **Training Speed**: 8xH100 = ~1-2 days vs 8xA100 = ~3-4 days

**Storage Requirements**:
- Model checkpoints: ~70GB each × 10 checkpoints = 700GB
- Dataset: ~5GB (1000 examples)
- Logs: ~50GB for full run
- Working space: ~500GB
- **Total**: ~2TB minimum

**Network Requirements**:
- Model parameter syncing between GPUs
- Checkpoint loading/saving
- Ray cluster communication
- **Minimum**: 100Gbps, **Recommended**: 200Gbps

**Single-node assumption for this plan**: We target a single machine with **8x H100 80GB**. All training and inference engines run on this node. Ray will run in single-node mode; multi-node instructions are optional for future scale-out.

### Local Setup Requirements

If running locally:
- CUDA 12.1+
- NVIDIA Driver 535+
- Docker with NVIDIA Container Toolkit
- 100GB+ free disk space on host
- Cooling solution for 8 GPUs

---

## Model Selection & Compatibility

### Base Model Evaluation

#### Current Recommendation: Qwen2.5-Coder-32B-Instruct

**Why Qwen2.5-Coder-32B?**
```yaml
pros:
  - Excellent coding capabilities (72.9 on HumanEval)
  - Good instruction following
  - 32K context window (fits NAT tool specs)
  - Fast inference with vLLM
  - Permissive license (Apache 2.0)
  - Already proven on OpenHands benchmarks

cons:
  - Large (32B params = needs 64GB VRAM)
  - Slower than smaller models
  - May overfit without enough data

specs:
  architecture: Transformer decoder
  parameters: 32B
  context_length: 32K tokens
  quantization: BF16/FP16
  size_on_disk: ~64GB
  license: Apache 2.0
  huggingface: Qwen/Qwen2.5-Coder-32B-Instruct
```

#### Version Compatibility

**Latest Versions** (as of Dec 2024):
```bash
# Qwen2.5-Coder series (Latest)
Qwen/Qwen2.5-Coder-32B-Instruct    # Released Nov 2024

**OpenHands Compatibility**:
- AgentHands is fork of OpenHands 0.22
- All Qwen2.5 models tested with OpenHands
- vLLM 0.6.0+ required for Qwen2.5
- SkyRL tested with Qwen2.5

**Recommendation**: Use **Qwen2.5-Coder-32B-Instruct** for final training, **14B for experiments**.

**Chosen default for this project**:
- We will use `Qwen/Qwen2.5-Coder-32B-Instruct` as the default everywhere (baseline, training, evaluation).

### Model Loading Configuration

```python
# Training config
model_config = {
    "path": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "dtype": "bfloat16",  # BF16 for better stability
    "trust_remote_code": True,
    "use_flash_attention_2": True,  # 30% faster
    "attn_implementation": "flash_attention_2",
}

# vLLM config for inference
vllm_config = {
    "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "tensor_parallel_size": 1,  # Per engine
    "gpu_memory_utilization": 0.85,
    "max_model_len": 8192,  # Reduce from 32K for speed
    "dtype": "bfloat16",
    "trust_remote_code": True,
    "enable_prefix_caching": True,  # Cache NAT system prompt
}
```

---

## Baseline Evaluation

### Why Baseline is Critical

Before RL training, we need to measure:
1. **Vanilla performance**: How good is Qwen2.5-Coder-32B without training?
2. **Upper bound**: How good is Gemini 2.5 Pro?
3. **Task difficulty**: What's the success rate distribution?
4. **Failure modes**: Where does the base model fail?

This tells us:
- Is RL training necessary? (If baseline is 90%, maybe not)
- What to focus training on
- How much improvement to expect
- If our task formulation is reasonable

### Baseline Evaluation Setup

#### Step 1: Create Evaluation Dataset

```python
# File: openhands/integrations/nat_poc/eval/create_eval_set.py

async def create_evaluation_dataset(
    output_dir: Path,
    num_samples: int = 100
):
    """
    Create diverse evaluation dataset

    Stratified by:
    - Difficulty (easy, medium, hard)
    - Domain (financial, research, coding, etc.)
    - Number of tools (1, 2-3, 4+)
    """

    planner = GeminiPlanner(api_key=os.getenv("GEMINI_API_KEY"))

    # Create stratified samples
    samples = []

    # Easy tasks (30%)
    easy_requests = [
        "I want a simple calculator",
        "Create an agent that tells me the weather",
        "Build a basic note-taking agent",
    ]

    # Medium tasks (50%)
    medium_requests = [
        "I want a financial research agent",
        "Create an agent for web research",
        "Build a customer support agent",
    ]

    # Hard tasks (20%)
    hard_requests = [
        "Create a multi-step data analysis agent with visualization",
        "Build an agent that can research, analyze, and generate reports",
        "Create an automated trading research agent",
    ]

    # Generate plans for each
    for category, requests in [
        ("easy", easy_requests),
        ("medium", medium_requests),
        ("hard", hard_requests)
    ]:
        for request in requests:
            plan = await planner.analyze_intent(request)
            samples.append({
                "task_id": f"{category}_{len(samples):03d}",
                "difficulty": category,
                "user_request": request,
                "plan": plan,
                "num_tools": len(plan.missing_tools),
            })

    # Save
    df = pd.DataFrame(samples)
    df.to_parquet(output_dir / "eval_set.parquet")

    return df
```

#### Step 2: Baseline Evaluation Runner

```python
# File: openhands/integrations/nat_poc/eval/run_baseline.py

class BaselineEvaluator:
    """
    Run baseline evaluation on vanilla Qwen2.5-Coder-32B
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        eval_dataset: str = "data/eval_set.parquet",
        output_dir: str = "results/baseline"
    ):
        self.model_name = model_name
        self.eval_dataset = pd.read_parquet(eval_dataset)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model with vLLM
        self.llm = vLLM(
            model=model_name,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
        )

    async def evaluate_sample(
        self,
        sample: Dict
    ) -> Dict:
        """
        Evaluate one sample

        Returns:
            - success: bool
            - validation_results: Dict
            - time_taken: float
            - tokens_used: int
        """
        start_time = time.time()

        # Create workspace
        workspace = self.output_dir / sample["task_id"]
        workspace.mkdir(exist_ok=True)

        # Create prompt for AgentHands
        prompt = self._create_prompt(sample)

        # Run AgentHands with vanilla model
        env = NATAgentCreationEnv(
            env_config=DictConfig({
                "workspace_dir": str(workspace),
                "max_turns": 30,
            }),
            extras={"agent_spec": sample}
        )

        # Simulate episode
        observation, metadata = env.init(prompt)
        done = False
        turns = 0

        while not done and turns < 30:
            # Get action from model
            action = await self._get_model_action(observation)

            # Step environment
            result = env.step(action)
            done = result["done"]
            turns += 1

        time_taken = time.time() - start_time

        return {
            "task_id": sample["task_id"],
            "success": result["metadata"]["success"],
            "validation": result["metadata"]["validation"],
            "reward": result["reward"],
            "turns": turns,
            "time_taken": time_taken,
        }

    async def run_full_evaluation(self):
        """
        Run on all eval samples
        """
        results = []

        for idx, sample in self.eval_dataset.iterrows():
            print(f"Evaluating {idx+1}/{len(self.eval_dataset)}: {sample['task_id']}")

            result = await self.evaluate_sample(sample)
            results.append(result)

            # Save incrementally
            pd.DataFrame(results).to_json(
                self.output_dir / "results.jsonl",
                orient="records",
                lines=True
            )

        # Calculate aggregate metrics
        df = pd.DataFrame(results)

        metrics = {
            "overall_success_rate": df["success"].mean(),
            "by_difficulty": df.groupby("difficulty")["success"].mean().to_dict(),
            "avg_turns": df["turns"].mean(),
            "avg_time": df["time_taken"].mean(),
            "avg_reward": df["reward"].mean(),
        }

        # Save metrics
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics


# Usage
if __name__ == "__main__":
    evaluator = BaselineEvaluator()
    metrics = asyncio.run(evaluator.run_full_evaluation())
    print(json.dumps(metrics, indent=2))
```

#### Step 3: Gemini Upper Bound

```python
# Evaluate Gemini 2.5 Pro as upper bound
gemini_evaluator = BaselineEvaluator(
    model_name="gemini-2.5-pro",  # Via API
    eval_dataset="data/eval_set.parquet",
    output_dir="results/gemini_upperbound"
)

gemini_metrics = asyncio.run(gemini_evaluator.run_full_evaluation())
```

### Expected Baseline Results

```yaml
vanilla_qwen_32b:
  overall_success_rate: 0.45  # 45%
  by_difficulty:
    easy: 0.70    # 70% on simple tasks
    medium: 0.45  # 45% on medium tasks
    hard: 0.20    # 20% on hard tasks
  common_failures:
    - Missing YAML config (40%)
    - Incomplete tests (35%)
    - NAT pattern errors (25%)
    - MCP setup missing (20%)
  avg_turns: 25
  avg_time: 180s  # 3 minutes per agent

gemini_2_5_pro_upperbound:
  overall_success_rate: 0.92  # 92%
  by_difficulty:
    easy: 0.98
    medium: 0.92
    hard: 0.85
  avg_turns: 18
  avg_time: 120s
```

### Running Baseline Evaluation

```bash
# 1. Create evaluation dataset
python -m openhands.integrations.nat_poc.eval.create_eval_set \
  --num_samples 100 \
  --output_dir ~/data/nat_eval

# 2. Run baseline
python -m openhands.integrations.nat_poc.eval.run_baseline \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --eval_dataset ~/data/nat_eval/eval_set.parquet \
  --output_dir ~/results/baseline_qwen32b

# 3. Analyze results
python -m openhands.integrations.nat_poc.eval.analyze_baseline \
  --results_dir ~/results/baseline_qwen32b

# Expected output:
# ========================================
# Baseline Evaluation Results
# ========================================
# Model: Qwen2.5-Coder-32B-Instruct
# Samples: 100
#
# Overall Success Rate: 45.0%
# By Difficulty:
#   Easy (30 samples):   70.0%
#   Medium (50 samples): 45.0%
#   Hard (20 samples):   20.0%
#
# Average Turns: 25.3
# Average Time: 182.4s
# Average Reward: 0.52
#
# Common Failure Modes:
#   1. Missing YAML config: 40 cases
#   2. Incomplete tests: 35 cases
#   3. NAT pattern errors: 25 cases
#
# Recommendation: RL training should focus on:
#   - YAML config generation
#   - Comprehensive test writing
#   - NAT pattern compliance
```

---

## Evaluation Harness

### Why We Need an Evaluation Harness

The evaluation harness is critical for:
1. **Consistent measurement** across baseline, training, and post-training
2. **Automated validation** of complete NAT agents
3. **Detailed failure analysis** to guide training
4. **Regression testing** to ensure improvements don't break things

### Evaluation Harness Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   EVALUATION HARNESS                         │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  1. Test Case Generator                             │    │
│  │     • Load evaluation dataset                       │    │
│  │     • Stratify by difficulty                        │    │
│  │     • Create reproducible splits                    │    │
│  └────────────────────────────────────────────────────┘    │
│                      ↓                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │  2. Agent Executor                                  │    │
│  │     • Run model on task                             │    │
│  │     • Capture all outputs                           │    │
│  │     • Track resource usage                          │    │
│  │     • Handle timeouts/errors                        │    │
│  └────────────────────────────────────────────────────┘    │
│                      ↓                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │  3. Validator                                       │    │
│  │     • Check tool files exist                        │    │
│  │     • Validate NAT pattern compliance               │    │
│  │     • Run pytest                                    │    │
│  │     • Check YAML config                             │    │
│  │     • Validate MCP setup                            │    │
│  │     • Run end-to-end tests                          │    │
│  └────────────────────────────────────────────────────┘    │
│                      ↓                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │  4. Metrics Calculator                              │    │
│  │     • Success rate                                  │    │
│  │     • Component-wise scores                         │    │
│  │     • Failure mode analysis                         │    │
│  │     • Statistical significance                      │    │
│  └────────────────────────────────────────────────────┘    │
│                      ↓                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │  5. Reporter                                        │    │
│  │     • Generate HTML report                          │    │
│  │     • Create visualizations                         │    │
│  │     • Export to JSON/CSV                            │    │
│  │     • Compare to baseline                           │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Implementation

```python
# File: openhands/integrations/nat_poc/eval/harness.py

class EvaluationHarness:
    """
    Complete evaluation harness for NAT agent creation
    """

    def __init__(
        self,
        eval_dataset: str,
        model_path: str,
        output_dir: str,
        num_workers: int = 8,
        timeout_per_task: int = 600,  # 10 minutes
    ):
        self.eval_dataset = pd.read_parquet(eval_dataset)
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.timeout_per_task = timeout_per_task

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.executor = AgentExecutor(model_path)
        self.validator = NATValidator()
        self.metrics_calculator = MetricsCalculator()

    async def run_evaluation(
        self,
        subset: Optional[str] = None,  # "easy", "medium", "hard", or None for all
    ) -> Dict:
        """
        Run complete evaluation
        """
        # Filter dataset if needed
        if subset:
            eval_data = self.eval_dataset[
                self.eval_dataset["difficulty"] == subset
            ]
        else:
            eval_data = self.eval_dataset

        print(f"Evaluating on {len(eval_data)} samples...")

        # Run in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._evaluate_single, row)
                for _, row in eval_data.iterrows()
            ]

            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results.append(result)

                # Save incrementally
                self._save_incremental(results)

        # Calculate metrics
        metrics = self.metrics_calculator.calculate(results)

        # Generate report
        self._generate_report(results, metrics)

        return metrics

    def _evaluate_single(self, sample: Dict) -> Dict:
        """
        Evaluate single sample with timeout
        """
        try:
            # Run with timeout
            with timeout(self.timeout_per_task):
                # Execute agent
                output = self.executor.run(sample)

                # Validate output
                validation = self.validator.validate(
                    workspace=output["workspace"],
                    expected=sample
                )

                return {
                    "task_id": sample["task_id"],
                    "success": validation["complete"],
                    "validation": validation,
                    "output": output,
                    "error": None,
                }

        except TimeoutError:
            return {
                "task_id": sample["task_id"],
                "success": False,
                "validation": None,
                "output": None,
                "error": "timeout",
            }

        except Exception as e:
            return {
                "task_id": sample["task_id"],
                "success": False,
                "validation": None,
                "output": None,
                "error": str(e),
            }

    def _generate_report(self, results: List[Dict], metrics: Dict):
        """
        Generate comprehensive HTML report
        """
        # Create HTML report
        html = f"""
        <html>
        <head><title>Evaluation Report</title></head>
        <body>
            <h1>NAT Agent Creation Evaluation</h1>
            <h2>Model: {self.model_path}</h2>

            <h3>Overall Metrics</h3>
            <ul>
                <li>Success Rate: {metrics['overall_success_rate']:.1%}</li>
                <li>Samples: {metrics['total_samples']}</li>
                <li>Avg Reward: {metrics['avg_reward']:.3f}</li>
            </ul>

            <h3>By Difficulty</h3>
            <table>
                <tr><th>Difficulty</th><th>Success Rate</th><th>Count</th></tr>
                <tr><td>Easy</td><td>{metrics['by_difficulty']['easy']:.1%}</td><td>{metrics['counts']['easy']}</td></tr>
                <tr><td>Medium</td><td>{metrics['by_difficulty']['medium']:.1%}</td><td>{metrics['counts']['medium']}</td></tr>
                <tr><td>Hard</td><td>{metrics['by_difficulty']['hard']:.1%}</td><td>{metrics['counts']['hard']}</td></tr>
            </table>

            <h3>Component Breakdown</h3>
            <ul>
                <li>Tool Generation: {metrics['components']['tools']:.1%}</li>
                <li>Integration: {metrics['components']['integration']:.1%}</li>
                <li>Workflow: {metrics['components']['workflow']:.1%}</li>
            </ul>

            <h3>Failure Modes</h3>
            <ul>
                {self._format_failure_modes(metrics['failure_modes'])}
            </ul>
        </body>
        </html>
        """

        # Save report
        (self.output_dir / "report.html").write_text(html)

        # Save JSON
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
```

### Evaluation Metrics

```python
class MetricsCalculator:
    """
    Calculate comprehensive evaluation metrics
    """

    def calculate(self, results: List[Dict]) -> Dict:
        """
        Calculate all metrics
        """
        df = pd.DataFrame(results)

        return {
            # Overall
            "overall_success_rate": df["success"].mean(),
            "total_samples": len(df),
            "successful": df["success"].sum(),
            "failed": (~df["success"]).sum(),

            # By difficulty
            "by_difficulty": {
                diff: df[df["difficulty"] == diff]["success"].mean()
                for diff in ["easy", "medium", "hard"]
            },

            # Component-wise
            "components": {
                "tools": self._calc_tool_score(df),
                "integration": self._calc_integration_score(df),
                "workflow": self._calc_workflow_score(df),
            },

            # Rewards
            "avg_reward": df["validation"].apply(
                lambda x: x["reward"] if x else 0
            ).mean(),

            # Efficiency
            "avg_turns": df["output"].apply(
                lambda x: x["turns"] if x else 0
            ).mean(),
            "avg_time": df["output"].apply(
                lambda x: x["time"] if x else 0
            ).mean(),

            # Failure modes
            "failure_modes": self._analyze_failures(df),

            # Statistical
            "confidence_interval_95": self._calc_ci(df["success"]),
        }

    def _analyze_failures(self, df: pd.DataFrame) -> Dict:
        """
        Analyze failure modes
        """
        failures = df[~df["success"]]

        modes = {
            "timeout": (failures["error"] == "timeout").sum(),
            "no_tools_created": 0,
            "missing_yaml": 0,
            "missing_mcp_setup": 0,
            "tests_failed": 0,
            "nat_pattern_errors": 0,
        }

        for _, row in failures.iterrows():
            if row["validation"]:
                val = row["validation"]
                if not val["tools"]["all_valid"]:
                    modes["no_tools_created"] += 1
                if not val["integration"]["yaml_valid"]:
                    modes["missing_yaml"] += 1
                if not val["integration"]["mcp_setup_valid"]:
                    modes["missing_mcp_setup"] += 1
                # ... more analysis

        return modes
```

### Usage

```bash
# Run baseline evaluation
python -m openhands.integrations.nat_poc.eval.harness \
  --eval_dataset ~/data/nat_eval/eval_set.parquet \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --output_dir ~/results/baseline \
  --num_workers 8

# Run after training
python -m openhands.integrations.nat_poc.eval.harness \
  --eval_dataset ~/data/nat_eval/eval_set.parquet \
  --model ~/ckpts/agenthands_nat/checkpoint_epoch_100 \
  --output_dir ~/results/trained \
  --num_workers 8

# Compare
python -m openhands.integrations.nat_poc.eval.compare \
  --baseline ~/results/baseline/metrics.json \
  --trained ~/results/trained/metrics.json
```

---

## Training Infrastructure

### Infrastructure Setup Steps

This is the detailed work required to set up training.

#### Single-node (8xH100) quickstart and Ray rationale

- Run everything on one 8xH100 box. Use FSDP2 for Qwen 32B and vLLM with 8 inference engines.
- Why Ray on a single node:
  - GPU-aware scheduler and placement groups
  - Shared object store for large artifacts
  - Robust actor lifecycles and failure handling
  - Same code scales to multi-node later
- You can initialize Ray implicitly in-process, or explicitly via `ray start`. For debugging, `local_mode=True` runs tasks inline.

```python
# Single-node Ray usage (optional explicit code)
import ray
# For normal execution on one node:
ray.init(num_cpus=None, num_gpus=8)
# For easier debugging (tasks run inline):
# ray.init(local_mode=True, num_gpus=8)
```

#### Step 1: Make sure 8xH100 Brev VM works

# Update system
sudo apt update && sudo apt upgrade -y

# Install basic tools
sudo apt install -y \
  git \
  tmux \
  htop \
  nvtop \
  tree \
  curl \
  wget

# Verify GPUs
nvidia-smi  # Should show 8x H100

# Check CUDA
nvcc --version  # Should be 12.1+
```

#### Step 2: Software Stack Installation (3-4 hours)

**Install Python & uv**:

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Verify
uv --version
```

**Install SkyRL**:

```bash
# Clone SkyRL
cd ~/
git clone --recurse-submodules https://github.com/NovaSky-AI/SkyRL
cd SkyRL/skyrl-train

# Create environment and install
uv sync --extra vllm
source .venv/bin/activate

# Verify installation
python -c "import skyrl_train; print('✅ SkyRL installed')"
```

**Optional: Pre-download Qwen 32B weights (faster first run)**:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct \
  --local-dir $HOME/models/Qwen2.5-Coder-32B-Instruct \
  --local-dir-use-symlinks False
export HF_HUB_ENABLE_HF_TRANSFER=1
```

**Install AgentHands**:

```bash
# Clone AgentHands
cd ~/
git clone https://github.com/athreesh/AgentHands
cd AgentHands

# Install
pip install -e .

# Verify
python -c "import openhands; print('✅ AgentHands installed')"
```

**Install SkyRL-Gym**:

```bash
cd ~/SkyRL/skyrl-gym
pip install -e .

# Verify
python -c "import skyrl_gym; print('✅ SkyRL-Gym installed')"
```

#### Step 3: Environment & Dataset Setup (2-3 hours)

**Copy NAT environment to SkyRL**:

```bash
# Copy environment file
cp ~/AgentHands/openhands/integrations/nat_poc/skyrl_integration/nat_agent_env.py \
   ~/SkyRL/skyrl-gym/skyrl_gym/envs/nat_agent_creation.py

# Register in skyrl-gym
# Edit: ~/SkyRL/skyrl-gym/skyrl_gym/envs/__init__.py
# Add: from skyrl_gym.envs.nat_agent_creation import NATAgentCreationEnv

# Reinstall
cd ~/SkyRL/skyrl-gym
pip install -e .
```

**Generate training dataset**:

```bash
# Set Gemini API key
export GEMINI_API_KEY=your_key_here

# Create dataset directory
mkdir -p ~/data/nat_agent_specs

# Run dataset generation
cd ~/AgentHands
python -m openhands.integrations.nat_poc.skyrl_integration.prepare_dataset \
  --output_dir ~/data/nat_agent_specs \
  --num_samples 1000

# This takes ~2-3 hours (1000 Gemini API calls)
# Cost: ~$10-50 depending on complexity

# Verify dataset
ls -lh ~/data/nat_agent_specs/
# Should see: train.parquet (900 examples), val.parquet (100 examples)
```

#### Step 4: Ray Cluster Setup (1-2 hours)

**Configure Ray**:

```bash
# Install Ray
pip install "ray[default]==2.51.1"

# Configure for uv
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

# Add to bashrc for persistence
echo 'export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook' >> ~/.bashrc

# Option A (recommended single-node simplicity): let SkyRL init Ray in-process.
#   No action needed here; training will call ray.init() internally.
#
# Option B (explicit runtime): start a local Ray head for visibility via `ray status`.
ray start --head --port=6379 --num-cpus=64 --num-gpus=8

# Verify (only if you used Option B)
ray status

# Expected output (Option B):
#   Node status shows 1 node with {'CPU': 64.0, 'GPU': 8.0, ...}
```

**For multi-node setup** (if using 2+ machines):

```bash
# On head node
ray start --head --port=6379 --num-cpus=64 --num-gpus=8

# Get head node IP
HEAD_IP=$(hostname -I | awk '{print $1}')
echo "Head node IP: $HEAD_IP"

# On worker nodes
ray start --address=$HEAD_IP:6379 --num-cpus=64 --num-gpus=8
```

#### Step 5: WandB Setup (15 minutes)

```bash
# Install wandb
pip install wandb

# Login
wandb login

# Create project
wandb init -p agenthands_nat_rl

# Set API key for training
export WANDB_API_KEY=your_wandb_key

# Add to bashrc
echo 'export WANDB_API_KEY=your_wandb_key' >> ~/.bashrc
```

#### Step 6: Create Training Scripts (1-2 hours)

**Create training directory**:

```bash
mkdir -p ~/SkyRL/skyrl-train/examples/nat_agent_creation
cd ~/SkyRL/skyrl-train/examples/nat_agent_creation
```

**Create training script** (`train_agenthands.sh`):

```bash
#!/bin/bash
set -x
set -e

# Configuration
export DATA_DIR="$HOME/data/nat_agent_specs"
export CKPT_DIR="$HOME/ckpts/agenthands_nat"
export NUM_GPUS=8
export MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"

# Logging
export WANDB_API_KEY=${WANDB_API_KEY}
export WANDB_PROJECT="agenthands_nat_rl"
export WANDB_RUN_NAME="qwen32b_nat_$(date +%Y%m%d_%H%M%S)"

# Ray configuration
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

# Verify data exists
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "❌ Error: Training data not found at $DATA_DIR/train.parquet"
    exit 1
fi

echo "✅ Training data found"
echo "✅ Starting training with $NUM_GPUS GPUs"
echo "✅ Model: $MODEL"

# Launch training
cd ~/SkyRL/skyrl-train

uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/val.parquet']" \
  \
  `# Algorithm` \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=true \
  \
  `# Model` \
  trainer.policy.model.path="$MODEL" \
  \
  `# Distributed Training` \
  trainer.placement.colocate_all=false \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  \
  `# Inference (vLLM)` \
  generator.num_inference_engines=$NUM_GPUS \
  generator.backend=vllm \
  generator.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=4 \
  generator.sampling_params.max_generate_length=8192 \
  generator.sampling_params.temperature=0.7 \
  generator.sampling_params.top_p=0.95 \
  generator.gpu_memory_utilization=0.85 \
  \
  `# Training Hyperparameters` \
  trainer.epochs=100 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=32 \
  trainer.eval_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.policy.optimizer_config.lr=5.0e-7 \
  trainer.policy.optimizer_config.weight_decay=0.01 \
  trainer.update_epochs_per_batch=2 \
  \
  `# Evaluation` \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  \
  `# Environment` \
  environment.env_class=nat_agent_creation \
  environment.env_config.workspace_dir="/tmp/nat_workspaces" \
  environment.env_config.max_turns=30 \
  \
  `# Checkpointing` \
  trainer.ckpt_interval=10 \
  trainer.ckpt_path="$CKPT_DIR" \
  \
  `# Logging` \
  trainer.logger=wandb \
  trainer.project_name="$WANDB_PROJECT" \
  trainer.run_name="$WANDB_RUN_NAME" \
  \
  $@

echo "✅ Training complete!"
echo "✅ Checkpoints saved to: $CKPT_DIR"
```

Make executable:

```bash
chmod +x train_agenthands.sh
```

#### Step 7: Pre-flight Checks (1 hour)

**Create validation script** (`preflight_check.sh`):

```bash
#!/bin/bash
set -e

echo "🔍 Running pre-flight checks..."

# Check GPUs
echo "1. Checking GPUs..."
nvidia-smi
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ "$GPU_COUNT" -ne 8 ]; then
    echo "❌ Expected 8 GPUs, found $GPU_COUNT"
    exit 1
fi
echo "✅ Found 8 GPUs"

# Check CUDA
echo "2. Checking CUDA..."
nvcc --version
echo "✅ CUDA OK"

# Check Python packages
echo "3. Checking Python packages..."
python -c "import skyrl_train; print('✅ SkyRL')"
python -c "import openhands; print('✅ AgentHands')"
python -c "import skyrl_gym; print('✅ SkyRL-Gym')"
python -c "import vllm; print('✅ vLLM')"
python -c "import ray; print('✅ Ray')"

# Check dataset
echo "4. Checking dataset..."
if [ ! -f "$HOME/data/nat_agent_specs/train.parquet" ]; then
    echo "❌ Training data not found"
    exit 1
fi
echo "✅ Dataset found"

# Check disk space
echo "5. Checking disk space..."
SPACE=$(df -h $HOME | tail -1 | awk '{print $4}')
echo "Available space: $SPACE"
echo "✅ Disk space OK"

# Check Ray
echo "6. Checking Ray..."
ray status || echo "ℹ️ Ray not started via 'ray start' (OK if using in-process ray.init())"
echo "✅ Ray available"

# Test environment import
echo "7. Testing environment..."
python -c "from skyrl_gym.envs.nat_agent_creation import NATAgentCreationEnv; print('✅ Environment OK')"

# Test model loading (small test)
echo "8. Testing model loading..."
python -c "
from vllm import LLM
llm = LLM(model='Qwen/Qwen2.5-Coder-7B-Instruct', dtype='bfloat16', max_model_len=1024)
output = llm.generate('Hello', max_tokens=10)
print('✅ Model loading OK')
"

echo ""
echo "✅✅✅ All pre-flight checks passed! ✅✅✅"
echo ""
echo "Ready to start training!"
echo ""
echo "To start training, run:"
echo "  bash train_agenthands.sh"
```

Run preflight:

```bash
bash preflight_check.sh
```

#### Step 8: Launch Training (30 minutes setup + 24-48 hours training)

**Start training in tmux** (so it continues if SSH disconnects):

```bash
# Create tmux session
tmux new -s training

# Inside tmux, start training
cd ~/SkyRL/skyrl-train/examples/nat_agent_creation
bash train_agenthands.sh

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

**Monitor training**:

```bash
# Attach to tmux
tmux attach -t training

# In separate terminal, monitor GPUs
watch -n 1 nvidia-smi

# Check WandB
# Open: https://wandb.ai/your-username/agenthands_nat_rl

# Check logs
tail -f /tmp/ray/session_latest/logs/*

# Check disk space
watch -n 60 df -h
```

---

## Implementation Phases

### Phase 1: Setup & Baseline (Week 1)

**Goal**: Infrastructure ready, baseline measured

#### Tasks

**Day 1-2: Environment Setup**
- [ ] Provision cloud instances (8xH100)
- [ ] Install software stack
- [ ] Set up Ray (single-node)
- [ ] Configure WandB

**Day 3-4: Baseline Evaluation**
- [ ] Create evaluation dataset (100 samples)
- [ ] Implement baseline evaluator
- [ ] Run Qwen2.5-Coder-32B baseline
- [ ] Run Gemini 2.5 Pro upper bound
- [ ] Analyze results

**Day 5: Evaluation Harness**
- [ ] Implement evaluation harness
- [ ] Create validation pipeline
- [ ] Test on small scale
- [ ] Document usage

**Deliverables**:
- ✅ Working infrastructure
- ✅ Baseline metrics documented
- ✅ Evaluation harness ready
- ✅ Gap analysis (baseline vs upper bound)

**Estimated Cost**: $500-1000 (instance rental)

---

### Phase 2: Core RL Components (Week 2)

**Goal**: All RL components implemented and tested

#### Tasks

**Day 1-3: SkyRL Environment**
- [ ] Implement `NATAgentCreationEnv`
- [ ] Implement reward function
- [ ] Implement validation logic
- [ ] Write unit tests
- [ ] Register with SkyRL

**Day 4-5: Dataset Preparation**
- [ ] Implement dataset generation script
- [ ] Generate 1000 training examples with Gemini
- [ ] Validate dataset quality
- [ ] Create train/val split

**Day 6-7: Integration & Testing**
- [ ] Test environment with dummy model
- [ ] Test full training loop on 1 GPU
- [ ] Verify checkpointing
- [ ] Verify logging

**Deliverables**:
- ✅ Working SkyRL environment
- ✅ Training dataset (1000 examples)
- ✅ Integration tests passing
- ✅ Small-scale test successful

**Estimated Cost**: $1000-1500

---

### Phase 3: Initial Training (Week 3)

**Goal**: First full-scale RL training run

#### Tasks

**Day 1: Pre-training Preparation**
- [ ] Run pre-flight checks
- [ ] Verify all components
- [ ] Set up monitoring
- [ ] Create backup plan

**Day 2-4: Training Run**
- [ ] Launch training (100 epochs)
- [ ] Monitor training
- [ ] Track GPU utilization
- [ ] Handle any issues

**Day 5-7: Evaluation & Analysis**
- [ ] Run trained model on validation set
- [ ] Compare to baseline
- [ ] Analyze failure modes
- [ ] Identify improvements

**Deliverables**:
- ✅ Trained model checkpoint
- ✅ Training metrics logged
- ✅ Evaluation results
- ✅ Analysis report

**Estimated Cost**: $2000-3000 (24-48 hours training)

---

### Phase 4: Iteration & Optimization (Week 4)

**Goal**: Improve based on initial results

#### Tasks

**Day 1-2: Reward Function Optimization**
- [ ] Analyze which reward components work
- [ ] Test alternative formulations
- [ ] Implement curriculum learning (optional)
- [ ] Add shaped rewards

**Day 3-5: Second Training Run**
- [ ] Launch training with improved config
- [ ] Monitor improvements
- [ ] Track metrics

**Day 6-7: Comparative Analysis**
- [ ] Compare run 1 vs run 2
- [ ] Statistical significance testing
- [ ] Document improvements

**Deliverables**:
- ✅ Improved model
- ✅ Comparative analysis
- ✅ Best practices documented

**Estimated Cost**: $2000-3000

---

### Phase 5: Deployment Preparation (Week 5)

**Goal**: Production-ready deployment

#### Tasks

**Day 1-3: Deployment Scripts**
- [ ] Implement model loading
- [ ] Create inference wrapper
- [ ] Integrate with Gemini planner
- [ ] Add logging/monitoring

**Day 4-5: Production Testing**
- [ ] Test on real scenarios
- [ ] Stress testing
- [ ] Latency optimization
- [ ] Error handling

**Day 6-7: Documentation**
- [ ] Write deployment guide
- [ ] API documentation
- [ ] User guide
- [ ] Troubleshooting guide

**Deliverables**:
- ✅ Production deployment scripts
- ✅ Complete documentation
- ✅ Monitoring dashboard

**Estimated Cost**: $500-1000

---

### Phase 6: Production & Monitoring (Week 6+)

**Goal**: Deploy and monitor in production

#### Tasks

**Week 6: Deployment**
- [ ] Deploy to production
- [ ] Set up monitoring
- [ ] Create dashboards
- [ ] Configure alerts

**Ongoing: Monitoring & Iteration**
- [ ] Collect user feedback
- [ ] Monitor success rates
- [ ] Identify edge cases
- [ ] Plan improvements

**Deliverables**:
- ✅ Production deployment
- ✅ Monitoring active
- ✅ Feedback loop established

---

## Testing & Validation

### Unit Tests

```python
# tests/test_nat_agent_env.py

def test_environment_initialization():
    """Test environment can be created"""
    env = NATAgentCreationEnv(
        env_config=DictConfig({"workspace_dir": "/tmp", "max_turns": 30}),
        extras={"agent_spec": sample_spec}
    )
    assert env is not None

def test_reward_calculation():
    """Test reward function"""
    validation = {
        "tools": {"all_valid": True},
        "integration": {"complete": True},
        "workflow": {"success": True}
    }
    reward = env._calculate_reward(validation)
    assert reward > 1.0  # Should get bonus for complete success

def test_validation():
    """Test validation logic"""
    # Create workspace with valid NAT agent
    # Run validation
    # Check all components validated correctly
    pass
```

### Integration Tests

```python
# tests/integration/test_training_pipeline.py

def test_full_episode():
    """Test complete episode execution"""
    # Load environment
    # Run episode with dummy model
    # Verify reward is calculated
    # Check all outputs exist

def test_dataset_loading():
    """Test dataset can be loaded"""
    dataset = load_dataset("train.parquet")
    assert len(dataset) > 0

def test_checkpoint_saving():
    """Test checkpoints save correctly"""
    # Run 1 epoch
    # Save checkpoint
    # Load checkpoint
    # Verify weights match
```

---

## Deployment Architecture

### Production Deployment

```
┌──────────────────────────────────────────────────────┐
│              PRODUCTION DEPLOYMENT                    │
│                                                       │
│  ┌────────────────────────────────────────────┐     │
│  │  FastAPI Server                             │     │
│  │  • REST API                                 │     │
│  │  • Request validation                       │     │
│  │  • Rate limiting                            │     │
│  └─────────────────┬──────────────────────────┘     │
│                    ↓                                  │
│  ┌────────────────────────────────────────────┐     │
│  │  Agent Creation Pipeline                    │     │
│  │  1. Gemini planning                         │     │
│  │  2. AgentHands generation (trained model)  │     │
│  │  3. Validation                              │     │
│  │  4. Package & return                        │     │
│  └────────────────────────────────────────────┘     │
│                                                       │
│  ┌────────────────────────────────────────────┐     │
│  │  Model Serving (vLLM)                      │     │
│  │  • Loaded trained checkpoint                │     │
│  │  • GPU inference                            │     │
│  │  • Batch processing                         │     │
│  └────────────────────────────────────────────┘     │
│                                                       │
│  ┌────────────────────────────────────────────┐     │
│  │  Monitoring                                 │     │
│  │  • Request metrics                          │     │
│  │  • Success rates                            │     │
│  │  • Latency tracking                         │     │
│  │  • Error logging                            │     │
│  └────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────┘
```

---

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Training doesn't converge | Medium | High | Start with small-scale test, monitor closely |
| Out of memory errors | High | Medium | Tune batch sizes, use gradient checkpointing |
| Hardware failures | Low | High | Use cloud with automatic failover |
| Dataset quality issues | Medium | High | Manual review of 100 examples |
| Model doesn't improve | Low | High | Have baseline fallback |

### Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Cost overrun | Medium | Medium | Set budget alerts, use spot instances |
| Timeline slippage | Medium | Medium | Build in 20% buffer |
| Key person unavailable | Low | High | Document everything |

---

## Success Metrics

### Training Success Criteria

| Metric | Baseline | Target | Stretch Goal |
|--------|---------|--------|--------------|
| Overall Success Rate | 45% | 70%+ | 85%+ |
| Easy Tasks | 70% | 90%+ | 95%+ |
| Medium Tasks | 45% | 70%+ | 85%+ |
| Hard Tasks | 20% | 50%+ | 70%+ |
| Tool Generation | 60% | 85%+ | 95%+ |
| Integration | 40% | 80%+ | 90%+ |
| Workflow | 30% | 70%+ | 80%+ |

### Cost Metrics

| Item | Budget | Actual |
|------|--------|--------|
| Instance rental | $6,000 | TBD |
| Gemini API calls | $100 | TBD |
| Storage | $50 | TBD |
| **Total** | **$6,150** | TBD |

---

## Appendix

### Glossary

- **GRPO**: Group Relative Policy Optimization
- **FSDP2**: Fully Sharded Data Parallel v2
- **vLLM**: Fast LLM inference engine
- **NAT**: NeMo Agent Toolkit
- **MCP**: Model Context Protocol

### References

- SkyRL: https://skyrl.readthedocs.io
- OpenHands: https://docs.openhands.dev
- Qwen2.5-Coder: https://huggingface.co/Qwen
- NAT: https://github.com/NVIDIA/NeMo-Agent-Toolkit

---

**Document End**

Questions? Contact: [Team lead]
Last Updated: December 2024
