"""
REAL End-to-End PoC with Actual OpenHands Execution

This uses:
1. Gemini 2.5 Pro for planning
2. REAL OpenHands (Gemini 2.5 Pro) for tool generation
3. REAL test validation

No simulation - this is the real deal!
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from openhands.integrations.nat_poc.gemini_planner import GeminiPlanner
from openhands.integrations.nat_poc.real_openhands_executor import RealOpenHandsExecutor


async def run_real_e2e():
    """Run the REAL end-to-end flow with actual OpenHands"""

    print("=" * 80)
    print("REAL END-TO-END: Financial Research Agent")
    print("Gemini Planning → OpenHands (Gemini) → NAT Tool → Validation")
    print("=" * 80)
    print()

    # ========================================================================
    # STEP 1: Gemini Planning
    # ========================================================================

    user_request = "I want an agent that can research a company's stock performance and analyze their latest financial statements"

    print("STEP 1: GEMINI 2.5 PRO PLANNING")
    print("-" * 80)
    print(f"User Request: {user_request}")
    print()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not set")
        return

    planner = GeminiPlanner(api_key=api_key, model="gemini-2.5-pro")

    print("🤖 Analyzing with Gemini 2.5 Pro...")
    plan = await planner.analyze_intent(user_request)

    print(f"✅ Plan Complete!")
    print(f"   Scaffold: {plan.scaffold_type}")
    print(f"   MCP Servers: {len(plan.mcp_servers)}")
    print(f"   Custom Tools: {len(plan.missing_tools)}")
    print(f"   Test Cases: {len(plan.test_cases)}")
    print()

    if plan.mcp_servers:
        print("📡 MCP Servers Selected:")
        for server in plan.mcp_servers:
            print(f"   • {server.name}")
        print()

    if plan.missing_tools:
        print("🔧 Custom Tools to Generate:")
        for tool in plan.missing_tools:
            print(f"   • {tool.name}: {tool.description}")
        print()

    # ========================================================================
    # STEP 2: REAL OpenHands Tool Generation
    # ========================================================================

    print("STEP 2: OPENHANDS (GEMINI 2.5 PRO) TOOL GENERATION")
    print("-" * 80)

    if not plan.missing_tools:
        print("✅ No custom tools needed (using MCP servers only)")
        print()
    else:
        executor = RealOpenHandsExecutor()

        for tool_spec in plan.missing_tools:
            print(f"\n{'=' * 80}")
            print(f"Generating Tool: {tool_spec.name}")
            print(f"{'=' * 80}\n")

            # Generate the prompt
            prompt = planner.generate_openhands_prompt(plan, tool_spec)

            # Save prompt for reference
            prompt_file = f"/tmp/nat_openhands_prompt_{tool_spec.name}.txt"
            with open(prompt_file, "w") as f:
                f.write(prompt)
            print(f"💾 Prompt saved: {prompt_file}")
            print()

            # ACTUALLY RUN OPENHANDS
            print("🚀 Launching OpenHands with Gemini 2.5 Pro...")
            print()

            result = await executor.generate_tool(
                prompt=prompt,
                tool_name=tool_spec.name,
                max_iterations=30
            )

            print()
            print("-" * 80)
            print(f"Generation Result for {tool_spec.name}:")
            print("-" * 80)

            if result["success"]:
                print(f"✅ SUCCESS!")
                print(f"   Tool file: {result['tool_file']}")
                print(f"   Test file: {result['test_file']}")

                if result.get("test_results"):
                    test_res = result["test_results"]
                    if test_res.get("passed"):
                        print(f"   ✅ Tests PASSED")
                    else:
                        print(f"   ❌ Tests FAILED")
                        if "error" in test_res:
                            print(f"      Error: {test_res['error']}")
            else:
                print(f"❌ FAILED")
                if "error" in result:
                    print(f"   Error: {result['error']}")
                return None

            print()

    # ========================================================================
    # STEP 3: Save Agent Configuration
    # ========================================================================

    print("STEP 3: NAT AGENT ASSEMBLY")
    print("-" * 80)

    config_file = "/tmp/nat_workspace/financial_research_agent_real.yml"
    with open(config_file, "w") as f:
        f.write(plan.yaml_config_template)

    print(f"✅ NAT Config saved: {config_file}")

    if plan.mcp_servers:
        mcp_file = "/tmp/nat_workspace/install_mcp_servers.sh"
        mcp_content = "#!/bin/bash\n# MCP Server Installation\n\n"
        for server in plan.mcp_servers:
            mcp_content += f"# {server.name}\n{server.install_command}\n\n"

        with open(mcp_file, "w") as f:
            f.write(mcp_content)

        print(f"✅ MCP setup script: {mcp_file}")

    print()

    # ========================================================================
    # STEP 4: Summary
    # ========================================================================

    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()

    print("✅ REAL End-to-End Execution Complete!")
    print()
    print(f"📊 What was created:")
    print(f"   • Gemini generated plan with {len(plan.test_cases)} test cases")
    print(f"   • OpenHands generated {len(plan.missing_tools)} NAT tool(s)")
    print(f"   • Identified {len(plan.mcp_servers)} MCP servers to use")
    print()

    print(f"📁 Generated Files:")
    print(f"   • Config: {config_file}")
    if plan.mcp_servers:
        print(f"   • MCP setup: {mcp_file}")
    for tool in plan.missing_tools:
        print(f"   • Tool: /tmp/nat_workspace/nat_tools/{tool.name}.py")
        print(f"   • Tests: /tmp/nat_workspace/tests/test_{tool.name}.py")
    print()

    print("🎯 Next Steps:")
    print("   1. Install MCP servers (run install_mcp_servers.sh)")
    print("   2. Install NAT: pip install nemo-agent-toolkit")
    print("   3. Run the agent: nat run --config financial_research_agent_real.yml")
    print("   4. Test with queries like: 'What's NVIDIA's stock performance?'")
    print()

    print("=" * 80)

    return plan


def main():
    """Main entry point"""
    result = asyncio.run(run_real_e2e())

    if result:
        print("\n✅ Real end-to-end execution completed successfully!")
        exit(0)
    else:
        print("\n❌ Real end-to-end execution failed")
        exit(1)


if __name__ == "__main__":
    main()
