#!/usr/bin/env python3
"""
DuckDuckGo MCP Server - Free web search for Claude Code
No API keys required!
"""

import asyncio
import json
from typing import Any, Optional
from mcp.server import Server
from mcp.types import Tool, TextContent
from duckduckgo_search import DDGS

app = Server("duckduckgo-search")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="web_search", description="Search the web using DuckDuckGo. Returns relevant search results with titles, URLs, and snippets.", inputSchema={
                "type": "object", "properties": {
                    "query": {
                        "type": "string", "description": "The search query to execute"
                    }, "max_results": {
                        "type": "integer", "description": "Maximum number of results to return (default: 10)", "default": 10, "minimum": 1, "maximum": 20
                    }, "region": {
                        "type": "string", "description": "Search region (default: 'us-en')", "default": "us-en"
                    }
                }, "required": ["query"]
            }
        ), Tool(
            name="news_search", description="Search for recent news using DuckDuckGo. Returns news articles with titles, URLs, dates, and summaries.", inputSchema={
                "type": "object", "properties": {
                    "query": {
                        "type": "string", "description": "The news search query"
                    }, "max_results": {
                        "type": "integer", "description": "Maximum number of news results (default: 10)", "default": 10, "minimum": 1, "maximum": 20
                    }
                }, "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    if name == "web_search":
        return await handle_web_search(arguments)
    elif name == "news_search":
        return await handle_news_search(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def handle_web_search(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle web search requests."""
    query = arguments["query"]
    max_results = arguments.get("max_results", 10)
    region = arguments.get("region", "us-en")
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region=region, max_results=max_results))
        
        if not results:
            return [TextContent(type="text", text=f"No search results found for: {query}")]
        
        # Format results for LLM consumption
        formatted_results = []
        formatted_results.append(f"🔍 **Web Search Results for: {query}**\n")
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('href', 'No URL')
            snippet = result.get('body', 'No description available')
            
            formatted_results.append(f"**{i}. {title}**")
            formatted_results.append(f"URL: {url}")
            formatted_results.append(f"Description: {snippet}")
            formatted_results.append("")
        
        return [TextContent(type="text", text="\n".join(formatted_results))]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error performing web search: {str(e)}")]

async def handle_news_search(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle news search requests."""
    query = arguments["query"]
    max_results = arguments.get("max_results", 10)
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
        
        if not results:
            return [TextContent(type="text", text=f"No news results found for: {query}")]
        
        # Format results for LLM consumption
        formatted_results = []
        formatted_results.append(f"📰 **News Results for: {query}**\n")
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', 'No URL')
            snippet = result.get('body', 'No description available')
            date = result.get('date', 'No date')
            source = result.get('source', 'Unknown source')
            
            formatted_results.append(f"**{i}. {title}**")
            formatted_results.append(f"Source: {source} | Date: {date}")
            formatted_results.append(f"URL: {url}")
            formatted_results.append(f"Summary: {snippet}")
            formatted_results.append("")
        
        return [TextContent(type="text", text="\n".join(formatted_results))]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error performing news search: {str(e)}")]

if __name__ == "__main__":
    import sys
    asyncio.run(app.run(sys.stdin, sys.stdout))