"""
Real OpenHands Executor for NAT Tool Generation

This module provides programmatic execution of OpenHands using Gemini 2.5 Pro
to generate actual NAT-compatible tools (not simulation).
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# Add OpenHands to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from openhands.core.config import OpenHandsConfig
from openhands.core.main import run_controller
from openhands.events.action import MessageAction
from openhands.events import EventStreamSubscriber
from openhands.core.logger import openhands_logger as logger


class RealOpenHandsExecutor:
    """Executes OpenHands with Gemini 2.5 Pro to generate NAT tools"""

    def __init__(self, config_path: str = None):
        """
        Initialize executor.

        Args:
            config_path: Path to OpenHands config file (defaults to config_nat_gemini.toml)
        """
        if config_path is None:
            config_path = str(Path(__file__).parent.parent.parent.parent / "config_nat_gemini.toml")

        self.config_path = config_path
        self.workspace_dir = Path("/tmp/nat_workspace")
        self.workspace_dir.mkdir(exist_ok=True, parents=True)

    async def generate_tool(
        self,
        prompt: str,
        tool_name: str,
        max_iterations: int = 30
    ) -> dict:
        """
        Generate a NAT tool using actual OpenHands with Gemini 2.5 Pro.

        Args:
            prompt: The detailed generation prompt
            tool_name: Name of the tool to generate
            max_iterations: Max iterations for OpenHands

        Returns:
            Dictionary with generation results
        """

        print(f"🤖 Executing OpenHands (Gemini 2.5 Pro) to generate: {tool_name}")
        print("-" * 80)

        # Load config
        config = OpenHandsConfig.load_from_toml(self.config_path)
        config.max_iterations = max_iterations
        config.workspace_base = str(self.workspace_dir)

        # Override with NAT-specific system prompt
        config.agent.system_prompt_filename = "system_prompt_nat.j2"

        print(f"📋 Config loaded:")
        print(f"   Model: {config.llm.model}")
        print(f"   Max iterations: {config.max_iterations}")
        print(f"   Workspace: {config.workspace_base}")
        print(f"   System prompt: {config.agent.system_prompt_filename}")
        print()

        # Create initial action with the prompt
        initial_action = MessageAction(content=prompt, source="user")

        print(f"📝 Sending prompt to OpenHands...")
        print(f"   Prompt length: {len(prompt)} characters")
        print()

        # Run OpenHands
        try:
            final_state = await run_controller(
                config=config,
                initial_user_action=initial_action,
                exit_on_message=False,
                headless_mode=True
            )

            if final_state is None:
                print("❌ OpenHands execution failed (final_state is None)")
                return {
                    "success": False,
                    "error": "Execution failed"
                }

            print(f"✅ OpenHands completed!")
            print(f"   Final state: {final_state.agent_state}")
            print(f"   Iterations: {final_state.iteration}")
            print()

            # Check for generated files
            tools_dir = self.workspace_dir / "nat_tools"
            tests_dir = self.workspace_dir / "tests"

            tool_file = tools_dir / f"{tool_name}.py"
            test_file = tests_dir / f"test_{tool_name}.py"

            if tool_file.exists():
                print(f"✅ Tool file created: {tool_file}")
            else:
                print(f"⚠️  Tool file not found: {tool_file}")

            if test_file.exists():
                print(f"✅ Test file created: {test_file}")
            else:
                print(f"⚠️  Test file not found: {test_file}")

            print()

            # Run tests if they exist
            test_result = None
            if test_file.exists():
                print("🧪 Running tests...")
                test_result = await self._run_pytest(test_file)

            return {
                "success": tool_file.exists(),
                "tool_file": str(tool_file) if tool_file.exists() else None,
                "test_file": str(test_file) if test_file.exists() else None,
                "test_results": test_result,
                "final_state": final_state
            }

        except Exception as e:
            print(f"❌ Error during OpenHands execution: {e}")
            import traceback
            traceback.print_exc()

            return {
                "success": False,
                "error": str(e)
            }

    async def _run_pytest(self, test_file: Path) -> dict:
        """Run pytest on the generated test file"""

        try:
            import subprocess

            result = subprocess.run(
                ["pytest", str(test_file), "-v"],
                capture_output=True,
                text=True,
                timeout=60
            )

            # Parse output
            passed = "passed" in result.stdout.lower()
            output_lines = result.stdout.split("\n")

            return {
                "passed": result.returncode == 0,
                "output": result.stdout,
                "details": output_lines[-10:] if len(output_lines) > 10 else output_lines
            }

        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "error": "Tests timed out after 60 seconds"
            }
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }


async def execute_tool_generation_real(prompt: str, tool_name: str) -> dict:
    """
    Execute OpenHands to generate a tool (real, not simulated).

    Args:
        prompt: The generation prompt
        tool_name: Name of the tool

    Returns:
        Generation results
    """

    executor = RealOpenHandsExecutor()
    result = await executor.generate_tool(
        prompt=prompt,
        tool_name=tool_name
    )

    return result


if __name__ == "__main__":
    # Test with a simple prompt
    test_prompt = """
    Create a simple NAT tool that adds two numbers.

    Follow the NAT pattern exactly:
    1. Create FinancialTrendAnalyzerConfig class
    2. Use @register_function decorator
    3. Implement async _arun function
    4. Yield FunctionInfo

    Create the file at /workspace/nat_tools/simple_adder.py
    """

    result = asyncio.run(execute_tool_generation_real(test_prompt, "simple_adder"))
    print(f"\nResult: {result}")
