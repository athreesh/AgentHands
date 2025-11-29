"""
Gemini 2.5 Pro Planning Agent for NAT Tool/Agent Generation

This module uses Gemini 2.5 Pro to analyze user intent and generate
comprehensive plans for creating NAT agents.

Now supports MCP server integration from smithery.ai - can choose between
using existing MCP servers vs creating custom NAT tools.
"""

import json
from dataclasses import dataclass
from typing import Any
import aiohttp
from .mcp_registry import MCPRegistry, MCPServer


@dataclass
class ToolSpec:
    """Specification for a tool that needs to be created"""
    name: str
    description: str
    purpose: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    dependencies: list[str]
    example_usage: str


@dataclass
class TestCase:
    """Test case for validating the created agent"""
    test_id: str
    description: str
    input_query: str
    expected_behavior: str
    success_criteria: str


@dataclass
class AgentPlan:
    """Complete plan for creating a NAT agent"""
    user_request: str
    scaffold_type: str  # tool_calling_agent, react_agent, etc.
    scaffold_reasoning: str
    existing_tools: list[str]
    mcp_servers: list[MCPServer]  # MCP servers to use
    missing_tools: list[ToolSpec]  # Custom tools to create
    tool_reasoning: str
    test_cases: list[TestCase]
    implementation_steps: list[str]
    yaml_config_template: str


class GeminiPlanner:
    """Uses Gemini 2.5 Pro to plan NAT agent creation"""

    def __init__(self, api_key: str, model: str = "gemini-2.5-pro"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def analyze_intent(
        self,
        user_request: str,
        nat_examples_path: str = "/workspace/nat_examples"
    ) -> AgentPlan:
        """
        Analyze user intent and generate a comprehensive plan.

        Args:
            user_request: The user's natural language request
            nat_examples_path: Path to NAT examples directory

        Returns:
            AgentPlan with all details for creating the agent
        """

        prompt = self._build_planning_prompt(user_request, nat_examples_path)

        # Call Gemini 2.5 Pro
        response_text = await self._call_gemini(prompt)

        # Parse response into structured plan
        plan = self._parse_plan(user_request, response_text)

        return plan

    def _build_planning_prompt(self, user_request: str, nat_examples_path: str) -> str:
        """Build comprehensive prompt for Gemini to analyze intent"""

        # Find potentially relevant MCP servers based on user request
        keywords = user_request.lower().split()
        all_servers = MCPRegistry.get_all_servers()

        # Filter servers that might be relevant
        relevant_servers = []
        if any(word in user_request.lower() for word in ['stock', 'finance', 'financial', 'market', 'company']):
            relevant_servers.extend(MCPRegistry.FINANCIAL_SERVERS)
        if any(word in user_request.lower() for word in ['search', 'research', 'find', 'web']):
            relevant_servers.extend(MCPRegistry.RESEARCH_SERVERS)
        if any(word in user_request.lower() for word in ['data', 'analyze', 'spreadsheet']):
            relevant_servers.extend(MCPRegistry.DATA_SERVERS)

        # Format MCP servers for prompt
        mcp_servers_info = MCPRegistry.format_servers_for_prompt(relevant_servers) if relevant_servers else "No specific MCP servers identified for this request."

        return f"""You are an expert AI agent architect specializing in the NeMo Agent Toolkit (NAT).

USER REQUEST:
{user_request}

Your task is to analyze this request and create a comprehensive plan for building a NAT agent that fulfills it.

IMPORTANT: You can choose to use existing MCP (Model Context Protocol) servers from smithery.ai OR create custom NAT tools OR use a combination of both. Consider the trade-offs:
- **MCP Servers**: Pre-built, maintained, often more reliable, but less customizable
- **Custom NAT Tools**: Fully customizable, optimized for your use case, but need to be built and maintained

NAT ARCHITECTURE OVERVIEW:
1. **Agent Types**:
   - tool_calling_agent: Best for tool-heavy tasks, fast execution, minimal reasoning
   - react_agent: Best for research/analysis tasks requiring multi-step reasoning
   - rewoo_agent: Best for tasks with parallelizable operations

2. **Common Existing NAT Tools**:
   - wiki_search: Wikipedia search
   - current_datetime: Get current date/time
   - code_generation: Generate code using LLM
   - document_search_tool: Search through documents
   - nvidia_rag: RAG-based retrieval
   - server_tool: Make HTTP API calls

3. **Tool Pattern**:
   - Tools are async functions registered with @register_function
   - Tools have a Config class defining parameters
   - Tools are referenced in YAML configs

AVAILABLE MCP SERVERS (from smithery.ai):
{mcp_servers_info}

ANALYSIS REQUIRED:

1. **Scaffold Selection** (think deeply):
   - What cognitive pattern is needed? (reasoning vs execution)
   - How many tools will be involved?
   - Is parallel execution beneficial?
   - Does the task require iterative refinement?
   - **Recommendation**: Which agent type (tool_calling_agent, react_agent, rewoo_agent)?
   - **Reasoning**: Why this choice?

2. **Tool Requirements** (be specific):
   - What specific capabilities are needed?
   - Which existing NAT tools can be reused?
   - **DECISION**: Should we use MCP servers or create custom tools?
     * Check the available MCP servers above
     * If an MCP server provides the needed capability, prefer it (pre-built, maintained)
     * If no MCP server exists OR customization is needed, create a custom NAT tool
   - For each MCP server to use:
     * Server name and smithery ID
     * What capabilities it provides
     * Why it's a good fit
   - For each NEW custom tool to create:
     * Tool name (snake_case)
     * Purpose and functionality
     * Input schema (what parameters)
     * Output schema (what it returns)
     * External dependencies (APIs, libraries)
     * Example usage

3. **Test Cases** (concrete and executable):
   - Create 3-5 test cases that validate the agent works
   - Each test should have:
     * Test ID
     * Description
     * Input query (actual user query to test)
     * Expected behavior (what should happen)
     * Success criteria (how to judge success)

4. **Implementation Steps**:
   - Step-by-step guide for creating this agent
   - Include tool creation, config setup, testing

5. **YAML Config Template**:
   - Provide a complete NAT YAML config that would implement this agent
   - Include LLM config, tool definitions, workflow setup

OUTPUT FORMAT (respond with valid JSON):
{{
  "scaffold_type": "tool_calling_agent|react_agent|rewoo_agent",
  "scaffold_reasoning": "Detailed explanation of why this scaffold was chosen",
  "existing_tools": ["tool1", "tool2"],
  "mcp_servers": [
    {{
      "name": "Server Name",
      "smithery_id": "@user/server-name",
      "capabilities_used": ["capability1", "capability2"],
      "reasoning": "Why we're using this MCP server"
    }}
  ],
  "missing_tools": [
    {{
      "name": "tool_name",
      "description": "What the tool does",
      "purpose": "Why we need this tool",
      "input_schema": {{"param1": "type", "param2": "type"}},
      "output_schema": {{"field1": "type", "field2": "type"}},
      "dependencies": ["library1", "library2"],
      "example_usage": "Example of calling this tool"
    }}
  ],
  "tool_reasoning": "Explanation of tool choices (both MCP servers and custom tools)",
  "test_cases": [
    {{
      "test_id": "test_1",
      "description": "Test description",
      "input_query": "Actual user query",
      "expected_behavior": "What should happen",
      "success_criteria": "How to judge success"
    }}
  ],
  "implementation_steps": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ],
  "yaml_config_template": "Complete YAML config as a string"
}}

Think deeply about the user's request and provide a comprehensive, well-reasoned plan.
"""

    async def _call_gemini(self, prompt: str) -> str:
        """Call Gemini 2.5 Pro API"""

        url = f"{self.base_url}/models/{self.model}:generateContent"

        headers = {
            "Content-Type": "application/json",
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,  # Lower for more deterministic planning
                "maxOutputTokens": 8192,
            }
        }

        # For PoC: disable SSL verification to avoid certificate issues
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                f"{url}?key={self.api_key}",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Gemini API error ({response.status}): {error_text}")

                result = await response.json()

                # Extract text from response
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                return text

    def _parse_plan(self, user_request: str, response_text: str) -> AgentPlan:
        """Parse Gemini's response into structured AgentPlan"""

        # Extract JSON from response (Gemini might wrap it in markdown)
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]  # Remove ```json
        if response_text.startswith("```"):
            response_text = response_text[3:]  # Remove ```
        if response_text.endswith("```"):
            response_text = response_text[:-3]  # Remove ```
        response_text = response_text.strip()

        try:
            plan_dict = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Gemini response as JSON: {e}\n\nResponse:\n{response_text}")

        # Parse MCP servers
        mcp_servers = []
        for mcp_data in plan_dict.get("mcp_servers", []):
            # Find the full MCPServer object from registry
            matching_servers = [s for s in MCPRegistry.get_all_servers()
                              if s.smithery_id == mcp_data["smithery_id"]]
            if matching_servers:
                mcp_servers.append(matching_servers[0])
            else:
                # Create a basic MCPServer if not found in registry
                mcp_servers.append(MCPServer(
                    name=mcp_data["name"],
                    smithery_id=mcp_data["smithery_id"],
                    description=mcp_data.get("reasoning", ""),
                    capabilities=mcp_data.get("capabilities_used", []),
                    install_command=f"npx @smithery/cli install {mcp_data['smithery_id']} --client claude"
                ))

        # Convert to structured objects
        missing_tools = [
            ToolSpec(
                name=t["name"],
                description=t["description"],
                purpose=t["purpose"],
                input_schema=t["input_schema"],
                output_schema=t["output_schema"],
                dependencies=t["dependencies"],
                example_usage=t["example_usage"]
            )
            for t in plan_dict.get("missing_tools", [])
        ]

        test_cases = [
            TestCase(
                test_id=tc["test_id"],
                description=tc["description"],
                input_query=tc["input_query"],
                expected_behavior=tc["expected_behavior"],
                success_criteria=tc["success_criteria"]
            )
            for tc in plan_dict.get("test_cases", [])
        ]

        return AgentPlan(
            user_request=user_request,
            scaffold_type=plan_dict["scaffold_type"],
            scaffold_reasoning=plan_dict["scaffold_reasoning"],
            existing_tools=plan_dict.get("existing_tools", []),
            mcp_servers=mcp_servers,
            missing_tools=missing_tools,
            tool_reasoning=plan_dict["tool_reasoning"],
            test_cases=test_cases,
            implementation_steps=plan_dict["implementation_steps"],
            yaml_config_template=plan_dict["yaml_config_template"]
        )

    def generate_openhands_prompt(self, plan: AgentPlan, tool_spec: ToolSpec) -> str:
        """
        Generate a detailed prompt for AgentHands to create a NAT tool.

        Args:
            plan: The overall agent plan
            tool_spec: Specification for the specific tool to create

        Returns:
            Comprehensive prompt for AgentHands
        """

        return f"""Create a NAT (NeMo Agent Toolkit) compatible tool based on the following specification:

CONTEXT:
User wants to build: {plan.user_request}
This tool is part of a {plan.scaffold_type} agent.

TOOL SPECIFICATION:
Name: {tool_spec.name}
Purpose: {tool_spec.purpose}
Description: {tool_spec.description}

Input Schema: {json.dumps(tool_spec.input_schema, indent=2)}
Output Schema: {json.dumps(tool_spec.output_schema, indent=2)}
Dependencies: {', '.join(tool_spec.dependencies) if tool_spec.dependencies else 'None'}

Example Usage:
{tool_spec.example_usage}

REQUIREMENTS:
1. Create the tool file at: /workspace/nat_tools/{tool_spec.name}.py
2. Follow the exact NAT tool pattern (see system prompt)
3. Include proper error handling
4. Add type hints for all parameters
5. Create tests at: /workspace/tests/test_{tool_spec.name}.py
6. Run tests to verify functionality

TEST CASES:
The tool will be tested as part of these agent scenarios:
{self._format_test_cases(plan.test_cases)}

DELIVERABLES:
1. /workspace/nat_tools/{tool_spec.name}.py - Tool implementation
2. /workspace/tests/test_{tool_spec.name}.py - Pytest tests
3. Run pytest and confirm all tests pass
4. Report file locations and test results

Begin implementation now. Follow NAT patterns exactly as described in your system prompt.
"""

    def _format_test_cases(self, test_cases: list[TestCase]) -> str:
        """Format test cases for inclusion in prompt"""
        formatted = []
        for tc in test_cases:
            formatted.append(f"- {tc.description}: \"{tc.input_query}\"")
        return "\n".join(formatted)
