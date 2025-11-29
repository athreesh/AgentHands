"""
Simple PoC: Weather-Aware Calculator Agent

This demonstrates the complete flow:
1. User request → Gemini 2.5 Pro (plan generation)
2. Gemini plan → AgentHands (tool generation)
3. AgentHands → New NAT agent (with working tools)
4. Validation through test cases

Simple scenario:
- User wants: "A calculator that can also tell me the current weather"
- Existing: calculator tools (basic math)
- Missing: weather API tool
- Agent type: tool_calling_agent (fast, tool-heavy)
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from openhands.integrations.nat_poc.gemini_planner import (
    GeminiPlanner,
    AgentPlan,
    ToolSpec,
)


async def run_simple_poc():
    """Run the simple weather-calculator PoC"""

    print("=" * 80)
    print("NAT Meta-Agent PoC: Weather-Aware Calculator")
    print("=" * 80)
    print()

    # Step 1: User Input
    user_request = "I want a calculator that can also tell me the current weather"

    print(f"📝 User Request: {user_request}")
    print()

    # Step 2: Gemini 2.5 Pro Planning
    print("🤖 Analyzing intent with Gemini 2.5 Pro...")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("Please set it with: export GEMINI_API_KEY=your_api_key")
        return

    planner = GeminiPlanner(api_key=api_key, model="gemini-2.5-pro")

    try:
        plan = await planner.analyze_intent(user_request)
    except Exception as e:
        print(f"❌ Error calling Gemini: {e}")
        return

    print("✅ Plan generated!")
    print()
    print("-" * 80)
    print("PLAN SUMMARY")
    print("-" * 80)
    print(f"Scaffold: {plan.scaffold_type}")
    print(f"Reasoning: {plan.scaffold_reasoning}")
    print()
    print(f"Existing Tools: {', '.join(plan.existing_tools)}")
    print()
    print(f"Missing Tools ({len(plan.missing_tools)}):")
    for tool in plan.missing_tools:
        print(f"  - {tool.name}: {tool.description}")
    print()
    print(f"Test Cases ({len(plan.test_cases)}):")
    for tc in plan.test_cases:
        print(f"  - {tc.test_id}: {tc.description}")
    print()
    print("Implementation Steps:")
    for i, step in enumerate(plan.implementation_steps, 1):
        print(f"  {i}. {step}")
    print("-" * 80)
    print()

    # Step 3: Generate AgentHands prompts for missing tools
    if plan.missing_tools:
        print(f"🔧 Need to generate {len(plan.missing_tools)} tool(s) using AgentHands...")
        print()

        for tool_spec in plan.missing_tools:
            print(f"📄 Generating prompt for: {tool_spec.name}")
            print("-" * 80)

            prompt = planner.generate_openhands_prompt(plan, tool_spec)
            print(prompt)
            print("-" * 80)
            print()

            # Save prompt to file for manual execution
            prompt_file = f"/tmp/nat_poc_prompt_{tool_spec.name}.txt"
            with open(prompt_file, "w") as f:
                f.write(prompt)

            print(f"💾 Prompt saved to: {prompt_file}")
            print()

    # Step 4: Display YAML config template
    print("📋 NAT Agent Configuration (YAML)")
    print("-" * 80)
    print(plan.yaml_config_template)
    print("-" * 80)
    print()

    # Step 5: Display test scenarios
    print("🧪 Test Scenarios for Validation")
    print("-" * 80)
    for tc in plan.test_cases:
        print(f"Test: {tc.test_id}")
        print(f"  Description: {tc.description}")
        print(f"  Input: {tc.input_query}")
        print(f"  Expected: {tc.expected_behavior}")
        print(f"  Success Criteria: {tc.success_criteria}")
        print()
    print("-" * 80)
    print()

    # Summary
    print("✅ PoC Complete!")
    print()
    print("NEXT STEPS:")
    print("1. Review the generated plan and prompts above")
    print("2. Use AgentHands to execute the tool generation prompts")
    print("3. The prompts have been saved to /tmp/nat_poc_prompt_*.txt")
    print("4. Once tools are created, use the YAML config to assemble the agent")
    print("5. Run the test scenarios to validate the agent works")
    print()


async def run_interactive_poc():
    """Interactive version where user provides their own request"""

    print("=" * 80)
    print("NAT Meta-Agent PoC: Interactive Mode")
    print("=" * 80)
    print()

    user_request = input("What agent do you want to build?\n> ").strip()

    if not user_request:
        print("❌ No request provided")
        return

    print()
    print(f"📝 User Request: {user_request}")
    print()

    # Step 2: Gemini 2.5 Pro Planning
    print("🤖 Analyzing intent with Gemini 2.5 Pro...")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("Please set it with: export GEMINI_API_KEY=your_api_key")
        return

    planner = GeminiPlanner(api_key=api_key, model="gemini-2.5-pro")

    try:
        plan = await planner.analyze_intent(user_request)
    except Exception as e:
        print(f"❌ Error calling Gemini: {e}")
        import traceback
        traceback.print_exc()
        return

    print("✅ Plan generated!")
    print()
    print("-" * 80)
    print("PLAN SUMMARY")
    print("-" * 80)
    print(f"Scaffold: {plan.scaffold_type}")
    print(f"Reasoning: {plan.scaffold_reasoning}")
    print()
    print(f"Existing Tools: {', '.join(plan.existing_tools) if plan.existing_tools else 'None'}")
    print()
    print(f"Missing Tools ({len(plan.missing_tools)}):")
    for tool in plan.missing_tools:
        print(f"  - {tool.name}: {tool.description}")
        print(f"    Input: {tool.input_schema}")
        print(f"    Output: {tool.output_schema}")
        print(f"    Dependencies: {', '.join(tool.dependencies) if tool.dependencies else 'None'}")
        print()
    print()
    print(f"Test Cases ({len(plan.test_cases)}):")
    for tc in plan.test_cases:
        print(f"  - {tc.test_id}: {tc.description}")
        print(f"    Query: \"{tc.input_query}\"")
        print(f"    Expected: {tc.expected_behavior}")
        print()
    print()
    print("Implementation Steps:")
    for i, step in enumerate(plan.implementation_steps, 1):
        print(f"  {i}. {step}")
    print("-" * 80)
    print()

    # Generate AgentHands prompts
    if plan.missing_tools:
        print(f"🔧 Generating prompts for {len(plan.missing_tools)} tool(s)...")
        print()

        for tool_spec in plan.missing_tools:
            prompt = planner.generate_openhands_prompt(plan, tool_spec)

            # Save prompt to file
            prompt_file = f"/tmp/nat_poc_prompt_{tool_spec.name}.txt"
            with open(prompt_file, "w") as f:
                f.write(prompt)

            print(f"💾 Prompt for {tool_spec.name} saved to: {prompt_file}")

        print()

    # Display YAML config
    print("📋 NAT Agent Configuration (YAML)")
    print("-" * 80)
    print(plan.yaml_config_template)
    print("-" * 80)
    print()

    # Save YAML to file
    yaml_file = "/tmp/nat_poc_config.yml"
    with open(yaml_file, "w") as f:
        f.write(plan.yaml_config_template)
    print(f"💾 YAML config saved to: {yaml_file}")
    print()

    print("✅ PoC Complete!")
    print()


def main():
    """Main entry point"""
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "simple"

    if mode == "interactive":
        asyncio.run(run_interactive_poc())
    else:
        asyncio.run(run_simple_poc())


if __name__ == "__main__":
    main()
