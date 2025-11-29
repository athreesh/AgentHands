"""
MCP Server Registry for NAT Integration

This module maintains a registry of available MCP servers from smithery.ai
and helps Gemini decide when to use MCP vs create custom tools.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class MCPServer:
    """Represents an MCP server from smithery.ai"""
    name: str
    smithery_id: str
    description: str
    capabilities: list[str]
    install_command: str
    uses: int = 0
    last_updated: str = ""


class MCPRegistry:
    """Registry of known MCP servers from smithery.ai"""

    # Financial & Stock Data MCP Servers
    FINANCIAL_SERVERS = [
        MCPServer(
            name="Yahoo Finance Server",
            smithery_id="@hwangwoohyun-nav/yahoo-finance-mcp",
            description="Comprehensive Yahoo Finance data including stock prices, company info, financial statements, options, and market news",
            capabilities=[
                "get_stock_price",
                "get_historical_data",
                "get_company_info",
                "get_financial_statements",
                "get_options_data",
                "get_market_news"
            ],
            install_command="npx @smithery/cli install @hwangwoohyun-nav/yahoo-finance-mcp --client claude",
            uses=0,
            last_updated="4 months ago"
        ),
        MCPServer(
            name="Yahoo Finance (marcus-nascimento98)",
            smithery_id="@marcus-nascimento98/yahoo-finance-mcp",
            description="Market data from Yahoo Finance: historical prices, fundamentals, financial statements, options chains, holders, corporate actions, news",
            capabilities=[
                "get_historical_prices",
                "get_fundamentals",
                "get_financial_statements",
                "get_options_chains",
                "get_holders",
                "get_corporate_actions",
                "get_news"
            ],
            install_command="npx @smithery/cli install @marcus-nascimento98/yahoo-finance-mcp --client claude",
            uses=0,
            last_updated="3 days ago"
        ),
        MCPServer(
            name="StockScreen",
            smithery_id="mcp-stockscreen",
            description="Comprehensive stock screening capabilities through Yahoo Finance",
            capabilities=[
                "screen_stocks",
                "filter_by_criteria",
                "compare_stocks"
            ],
            install_command="npx @smithery/cli install mcp-stockscreen --client claude",
            uses=0,
            last_updated="Dec 20, 2024"
        ),
        MCPServer(
            name="Financial Modeling Prep",
            smithery_id="@vijitdaroch/financial-modeling-prep-mcp-server",
            description="Financial data from Financial Modeling Prep API",
            capabilities=[
                "get_company_profile",
                "get_financial_ratios",
                "get_stock_quotes",
                "get_market_data"
            ],
            install_command="npx @smithery/cli install @vijitdaroch/financial-modeling-prep-mcp-server --client claude",
            uses=0,
            last_updated="recent"
        )
    ]

    # Research & Search MCP Servers
    RESEARCH_SERVERS = [
        MCPServer(
            name="Exa Search",
            smithery_id="exa",
            description="Fast, intelligent web search and web crawling - reduces hallucinations with fresh API/library info",
            capabilities=[
                "web_search",
                "web_crawl",
                "get_fresh_data"
            ],
            install_command="npx @smithery/cli install exa --client claude",
            uses=785049,
            last_updated="active"
        ),
        MCPServer(
            name="Linkup",
            smithery_id="linkup",
            description="Real-time web search providing trustworthy, source-backed answers and latest news",
            capabilities=[
                "real_time_search",
                "get_news",
                "source_backed_answers"
            ],
            install_command="npx @smithery/cli install linkup --client claude",
            uses=3900,
            last_updated="active"
        )
    ]

    # Data Analysis MCP Servers
    DATA_SERVERS = [
        MCPServer(
            name="Google Sheets",
            smithery_id="google-sheets",
            description="Real-time collaboration, data analysis, and integration with cloud spreadsheets",
            capabilities=[
                "read_sheets",
                "write_sheets",
                "analyze_data",
                "collaborate"
            ],
            install_command="npx @smithery/cli install google-sheets --client claude",
            uses=77,
            last_updated="active"
        ),
        MCPServer(
            name="Airtable",
            smithery_id="airtable",
            description="Combines spreadsheet and database for organizing and tracking data",
            capabilities=[
                "database_operations",
                "organize_data",
                "track_records"
            ],
            install_command="npx @smithery/cli install airtable --client claude",
            uses=859,
            last_updated="active"
        )
    ]

    @classmethod
    def find_servers_for_capabilities(cls, required_capabilities: list[str]) -> list[MCPServer]:
        """Find MCP servers that match required capabilities"""
        all_servers = cls.FINANCIAL_SERVERS + cls.RESEARCH_SERVERS + cls.DATA_SERVERS

        matching_servers = []
        for server in all_servers:
            # Check if server has any of the required capabilities
            if any(cap.lower() in ' '.join(server.capabilities).lower()
                   or cap.lower() in server.description.lower()
                   for cap in required_capabilities):
                matching_servers.append(server)

        return matching_servers

    @classmethod
    def find_servers_by_category(cls, category: str) -> list[MCPServer]:
        """Find servers by category: financial, research, data"""
        if category.lower() == "financial":
            return cls.FINANCIAL_SERVERS
        elif category.lower() == "research":
            return cls.RESEARCH_SERVERS
        elif category.lower() == "data":
            return cls.DATA_SERVERS
        else:
            return []

    @classmethod
    def get_all_servers(cls) -> list[MCPServer]:
        """Get all registered MCP servers"""
        return cls.FINANCIAL_SERVERS + cls.RESEARCH_SERVERS + cls.DATA_SERVERS

    @classmethod
    def format_servers_for_prompt(cls, servers: list[MCPServer]) -> str:
        """Format servers for inclusion in Gemini prompt"""
        if not servers:
            return "No matching MCP servers found."

        formatted = []
        for server in servers:
            formatted.append(f"""
**{server.name}** (smithery.ai/{server.smithery_id})
  - Description: {server.description}
  - Capabilities: {', '.join(server.capabilities)}
  - Install: `{server.install_command}`
  - Usage: {server.uses:,} uses
""")
        return "\n".join(formatted)
