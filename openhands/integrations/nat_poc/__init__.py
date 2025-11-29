"""
NeMo Agent Toolkit (NAT) Integration PoC

This module demonstrates the integration between AgentHands and NAT:
1. Gemini 2.5 Pro analyzes user intent and creates a plan
2. AgentHands generates NAT-compatible tools based on the plan
3. Tools are assembled into a working NAT agent
4. Agent is validated through test cases

Key Components:
- gemini_planner.py: Gemini 2.5 Pro wrapper for intent analysis
- poc_simple_scenario.py: Simple demo (weather-aware calculator)

Usage:
    # Set API key
    export GEMINI_API_KEY=your_key_here

    # Run simple PoC
    python -m openhands.integrations.nat_poc.poc_simple_scenario

    # Run interactive mode
    python -m openhands.integrations.nat_poc.poc_simple_scenario interactive
"""

from .gemini_planner import GeminiPlanner, AgentPlan, ToolSpec, TestCase

__all__ = ["GeminiPlanner", "AgentPlan", "ToolSpec", "TestCase"]
__version__ = "0.1.0"
