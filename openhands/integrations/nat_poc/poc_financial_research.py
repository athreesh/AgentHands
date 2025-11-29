"""
Financial Research Agent PoC with MCP Integration

This demonstrates the enhanced flow:
1. User request → Gemini 2.5 Pro (considers MCP servers)
2. Gemini decides: Use MCP server OR create custom tool OR both
3. Generate plan with test cases
4. (Next) AgentHands executes the plan
5. (Next) Validate with test cases

Scenario: "I want an agent that can research a company's stock performance and analyze their latest financial statements"
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from openhands.integrations.nat_poc.gemini_planner import GeminiPlanner
from openhands.integrations.nat_poc.mcp_registry import MCPRegistry


async def run_financial_research_poc():
    """Run the financial research agent PoC"""

    print("=" * 80)
    print("NAT Meta-Agent PoC: Financial Research Agent with MCP Integration")
    print("=" * 80)
    print()

    # Step 1: User Input
    user_request = "I want an agent that can research a company's stock performance and analyze their latest financial statements"

    print(f"📝 User Request: {user_request}")
    print()

    # Step 2: Show available MCP servers
    print("🔍 Available MCP Servers (smithery.ai):")
    print("-" * 80)

    financial_servers = MCPRegistry.FINANCIAL_SERVERS
    for server in financial_servers:
        print(f"  • {server.name}")
        print(f"    ID: {server.smithery_id}")
        print(f"    Capabilities: {', '.join(server.capabilities)}")
        print()

    research_servers = MCPRegistry.RESEARCH_SERVERS
    for server in research_servers[:2]:  # Just show top 2
        print(f"  • {server.name}")
        print(f"    ID: {server.smithery_id}")
        print(f"    Capabilities: {', '.join(server.capabilities)}")
        print()

    print("-" * 80)
    print()

    # Step 3: Gemini 2.5 Pro Planning
    print("🤖 Analyzing intent with Gemini 2.5 Pro...")
    print("   (Gemini will decide: MCP servers vs custom tools)")
    print()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
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
    print("=" * 80)
    print("COMPREHENSIVE PLAN")
    print("=" * 80)
    print()

    # Scaffold
    print(f"🏗️  SCAFFOLD: {plan.scaffold_type}")
    print(f"Reasoning: {plan.scaffold_reasoning}")
    print()

    # Existing Tools
    if plan.existing_tools:
        print(f"✅ EXISTING NAT TOOLS ({len(plan.existing_tools)}):")
        for tool in plan.existing_tools:
            print(f"  • {tool}")
        print()

    # MCP Servers
    if plan.mcp_servers:
        print(f"📡 MCP SERVERS TO USE ({len(plan.mcp_servers)}):")
        for server in plan.mcp_servers:
            print(f"  • {server.name} ({server.smithery_id})")
            print(f"    Capabilities: {', '.join(server.capabilities)}")
            print(f"    Install: {server.install_command}")
            print()
    else:
        print("📡 MCP SERVERS: None selected")
        print()

    # Custom Tools to Create
    if plan.missing_tools:
        print(f"🔧 CUSTOM TOOLS TO CREATE ({len(plan.missing_tools)}):")
        for tool in plan.missing_tools:
            print(f"  • {tool.name}")
            print(f"    Purpose: {tool.purpose}")
            print(f"    Input: {tool.input_schema}")
            print(f"    Output: {tool.output_schema}")
            print(f"    Dependencies: {', '.join(tool.dependencies)}")
            print()
    else:
        print("🔧 CUSTOM TOOLS: None needed (using MCP servers)")
        print()

    # Reasoning
    print("💡 TOOL REASONING:")
    print(f"{plan.tool_reasoning}")
    print()

    # Test Cases
    print(f"🧪 TEST CASES ({len(plan.test_cases)}):")
    for tc in plan.test_cases:
        print(f"  {tc.test_id}: {tc.description}")
        print(f"    Query: \"{tc.input_query}\"")
        print(f"    Expected: {tc.expected_behavior}")
        print(f"    Success: {tc.success_criteria}")
        print()

    # Implementation Steps
    print("📋 IMPLEMENTATION STEPS:")
    for i, step in enumerate(plan.implementation_steps, 1):
        print(f"  {i}. {step}")
    print()

    print("=" * 80)
    print()

    # Step 4: Generate AgentHands prompts (if needed)
    if plan.missing_tools:
        print(f"🔧 Generating AgentHands prompts for {len(plan.missing_tools)} custom tool(s)...")
        print()

        for tool_spec in plan.missing_tools:
            prompt = planner.generate_openhands_prompt(plan, tool_spec)

            # Save prompt
            prompt_file = f"/tmp/nat_financial_prompt_{tool_spec.name}.txt"
            with open(prompt_file, "w") as f:
                f.write(prompt)

            print(f"💾 Prompt for '{tool_spec.name}' saved to: {prompt_file}")

        print()

    # Step 5: Save MCP server instructions
    if plan.mcp_servers:
        print("📡 Generating MCP Server Installation Instructions...")
        print()

        instructions = "# MCP Server Installation Instructions\n\n"
        for server in plan.mcp_servers:
            instructions += f"## {server.name}\n"
            instructions += f"Install command:\n```bash\n{server.install_command}\n```\n\n"
            instructions += f"Capabilities: {', '.join(server.capabilities)}\n\n"

        mcp_file = "/tmp/nat_financial_mcp_servers.md"
        with open(mcp_file, "w") as f:
            f.write(instructions)

        print(f"💾 MCP instructions saved to: {mcp_file}")
        print()

    # Step 6: Save YAML config
    yaml_file = "/tmp/nat_financial_config.yml"
    with open(yaml_file, "w") as f:
        f.write(plan.yaml_config_template)

    print(f"📋 NAT YAML config saved to: {yaml_file}")
    print()

    # Step 7: Save test cases
    test_file = "/tmp/nat_financial_tests.json"
    import json
    test_data = [
        {
            "test_id": tc.test_id,
            "description": tc.description,
            "input_query": tc.input_query,
            "expected_behavior": tc.expected_behavior,
            "success_criteria": tc.success_criteria
        }
        for tc in plan.test_cases
    ]
    with open(test_file, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"🧪 Test cases saved to: {test_file}")
    print()

    print("=" * 80)
    print("✅ PoC Complete!")
    print("=" * 80)
    print()

    # Summary
    print("📊 SUMMARY:")
    print(f"  Scaffold: {plan.scaffold_type}")
    print(f"  Existing Tools: {len(plan.existing_tools)}")
    print(f"  MCP Servers: {len(plan.mcp_servers)}")
    print(f"  Custom Tools: {len(plan.missing_tools)}")
    print(f"  Test Cases: {len(plan.test_cases)}")
    print()

    print("🎯 KEY INSIGHT:")
    if plan.mcp_servers and not plan.missing_tools:
        print("  Gemini chose to use ONLY MCP servers (no custom tools needed)")
    elif not plan.mcp_servers and plan.missing_tools:
        print("  Gemini chose to create ONLY custom tools (no MCP servers used)")
    elif plan.mcp_servers and plan.missing_tools:
        print("  Gemini chose a HYBRID approach (MCP servers + custom tools)")
    print()

    print("📁 NEXT STEPS:")
    print("  1. Review the plan above")
    if plan.mcp_servers:
        print("  2. Install MCP servers (see /tmp/nat_financial_mcp_servers.md)")
    if plan.missing_tools:
        print("  3. Use AgentHands to generate custom tools (prompts in /tmp)")
    print("  4. Use the YAML config to assemble the NAT agent")
    print("  5. Run test cases to validate")
    print()


def main():
    """Main entry point"""
    asyncio.run(run_financial_research_poc())


if __name__ == "__main__":
    main()
