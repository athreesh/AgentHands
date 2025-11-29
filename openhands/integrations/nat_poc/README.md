# NAT Meta-Agent PoC

This PoC demonstrates an AI system that can automatically create new NAT agents based on natural language descriptions.

## Architecture

```
User Input (Natural Language)
    ↓
Gemini 2.5 Pro (Reasoning & Planning)
  • Analyzes user intent
  • Selects appropriate NAT scaffold
  • Identifies required tools
  • Generates test cases
    ↓
AgentHands (Tool Generation)
  • Creates NAT-compatible tools
  • Follows NAT patterns exactly
  • Includes tests and validation
    ↓
NAT Agent (Assembly & Deployment)
  • Combines existing + generated tools
  • Creates YAML configuration
  • Validates through test cases
```

## Simple Scenario: Weather-Aware Calculator

**User Request:** "I want a calculator that can also tell me the current weather"

**What Gemini Does:**
1. Analyzes: User wants calculation + weather functionality
2. Scaffold: `tool_calling_agent` (tool-heavy, fast execution)
3. Existing tools: calculator tools (basic math)
4. Missing tools: `weather_api` (needs to be created)
5. Test case: "Calculate 5+3 and tell me weather in San Francisco"

**What AgentHands Does:**
1. Receives spec for `weather_api` tool
2. Creates `/workspace/nat_tools/weather_api.py` following NAT pattern
3. Creates `/workspace/tests/test_weather_api.py`
4. Runs tests to validate functionality

**Result:** Working NAT agent that can do math AND fetch weather

## Setup

### Prerequisites

1. **Gemini API Key**
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   ```

2. **Python Dependencies**
   ```bash
   pip install aiohttp
   ```

3. **NAT Examples** (already included in `AgentHands/nat_examples/`)

## Usage

### Simple PoC (Pre-defined Scenario)

```bash
cd /Users/anish/Desktop/repos/AgentHands
python -m openhands.integrations.nat_poc.poc_simple_scenario
```

This runs the weather-calculator scenario and shows:
- Gemini's reasoning and plan
- Generated AgentHands prompts
- YAML configuration template
- Test scenarios

### Interactive Mode (Your Own Request)

```bash
python -m openhands.integrations.nat_poc.poc_simple_scenario interactive
```

Then enter your request:
```
What agent do you want to build?
> I want an agent that can search papers on arXiv and summarize them
```

## Output

The PoC generates:

1. **Console Output**: Complete plan with reasoning
2. **Prompt Files**: `/tmp/nat_poc_prompt_*.txt` (for AgentHands)
3. **YAML Config**: `/tmp/nat_poc_config.yml` (NAT configuration)

## Example Output

```
================================================================================
NAT Meta-Agent PoC: Weather-Aware Calculator
================================================================================

📝 User Request: I want a calculator that can also tell me the current weather

🤖 Analyzing intent with Gemini 2.5 Pro...
✅ Plan generated!

--------------------------------------------------------------------------------
PLAN SUMMARY
--------------------------------------------------------------------------------
Scaffold: tool_calling_agent
Reasoning: This task is primarily tool-heavy (calculator + weather API),
requiring fast execution with minimal reasoning. tool_calling_agent is optimal.

Existing Tools: calculator, add, subtract, multiply, divide
Missing Tools (1):
  - weather_api: Fetch current weather data for a city

Test Cases (3):
  - test_1: Calculate 5+3 and get weather for San Francisco
  - test_2: Complex calculation with weather lookup
  - test_3: Weather for multiple cities with math

Implementation Steps:
  1. Create weather_api tool following NAT pattern
  2. Register tool with @register_function decorator
  3. Create YAML config with calculator + weather_api tools
  4. Test end-to-end scenarios
  5. Validate all test cases pass
```

## Next Steps

After running the PoC:

1. **Review Generated Plan**: Check Gemini's reasoning and tool specs
2. **Execute AgentHands**: Use the generated prompts to create tools
3. **Assemble Agent**: Use the YAML config to build the NAT agent
4. **Validate**: Run the test scenarios to verify functionality
5. **Iterate**: If tests fail, use RL to optimize (future work)

## File Structure

```
openhands/integrations/nat_poc/
├── __init__.py
├── README.md
├── gemini_planner.py          # Gemini 2.5 Pro wrapper
└── poc_simple_scenario.py     # Main PoC script

openhands/agenthub/codeact_agent/prompts/
└── system_prompt_nat.j2       # NAT-specific system prompt

nat_examples/                   # NAT agent examples
├── tool_calling/
└── react/
```

## Testing the Full Flow

To test the complete flow:

1. **Run PoC to generate plan**:
   ```bash
   python -m openhands.integrations.nat_poc.poc_simple_scenario
   ```

2. **Copy prompt to AgentHands** (once AgentHands server is running):
   ```bash
   cat /tmp/nat_poc_prompt_weather_api.txt
   # Use this prompt in AgentHands interface
   ```

3. **Verify tool creation**:
   ```bash
   # Check if tool was created
   ls /workspace/nat_tools/weather_api.py

   # Run tests
   pytest /workspace/tests/test_weather_api.py
   ```

4. **Assemble NAT agent**:
   ```bash
   # Use the generated YAML config
   nat run --config /tmp/nat_poc_config.yml
   ```

5. **Test the agent**:
   ```bash
   # Run test scenarios
   nat eval --config /tmp/nat_poc_config.yml
   ```

## Future Work

### Phase 2: RL Optimization
- Reward function based on test passage rates
- SkyRL integration for policy optimization
- Experience replay for learning from past successes

### Phase 3: GUI
- Gradio interface for user input
- Real-time progress visualization
- Agent testing playground
- Feedback collection

### Phase 4: Production
- Multi-agent orchestration
- Template library
- Auto-documentation
- Deployment automation

## API Reference

### GeminiPlanner

```python
from openhands.integrations.nat_poc import GeminiPlanner

planner = GeminiPlanner(api_key="your_key", model="gemini-2.5-pro")

# Analyze user intent
plan = await planner.analyze_intent(
    user_request="I want a calculator that can tell me the weather"
)

# Generate AgentHands prompt for a specific tool
prompt = planner.generate_openhands_prompt(plan, tool_spec)
```

### AgentPlan

```python
@dataclass
class AgentPlan:
    user_request: str
    scaffold_type: str              # tool_calling_agent, react_agent, etc.
    scaffold_reasoning: str         # Why this scaffold was chosen
    existing_tools: list[str]       # Tools already in NAT
    missing_tools: list[ToolSpec]   # Tools to generate
    tool_reasoning: str             # Why these tools
    test_cases: list[TestCase]      # Validation scenarios
    implementation_steps: list[str] # Step-by-step guide
    yaml_config_template: str       # NAT YAML config
```

## Troubleshooting

### "GEMINI_API_KEY not set"
```bash
export GEMINI_API_KEY=your_api_key_here
```

### "Gemini API error"
Check that you're using the correct API key and model name:
- Model: `gemini-2.5-pro`
- Ensure API key has access to Gemini 2.5 Pro

### "Failed to parse Gemini response"
Gemini might wrap JSON in markdown code blocks. The parser handles this, but if it still fails, check the raw response.

## Contributing

This is a PoC for the meta-agent system. To extend:

1. Add more sophisticated test generation
2. Integrate with actual AgentHands execution
3. Implement RL reward functions
4. Build the GUI interface

## License

Same as parent AgentHands project.
