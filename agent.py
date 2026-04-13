import logging
import os
import re
from datetime import datetime
from typing import Dict, Any, List

from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self):
        """Initialize the research agent with Tavily search and Claude LLM"""
        logger.info("Initializing ResearchAgent")
        
        # Check for required API keys
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        
        if not anthropic_key:
            logger.warning("ANTHROPIC_API_KEY not found in environment variables")
        
        if not tavily_key:
            logger.warning("TAVILY_API_KEY not found in environment variables")
        
        # Initialize Tavily search tool only if API key is available
        if tavily_key:
            self.search_tool = TavilySearch(
                max_results=5,
                description="Search the web for current information on any topic"
            )
        else:
            logger.warning("Tavily search tool not initialized - missing API key")
            self.search_tool = None
        
        # Initialize Claude LLM only if API key is available
        if anthropic_key:
            self.llm = ChatAnthropic(
                model="claude-haiku-4-5-20251001",
                temperature=0.1,
                anthropic_api_key=anthropic_key
            )
        else:
            logger.warning("Claude LLM not initialized - missing API key")
            self.llm = None
        
        # Create the agent only if both tools are available
        if self.search_tool and self.llm:
            self.agent = self._create_agent()
        else:
            logger.warning("Agent not created - missing API keys")
            self.agent = None
        
    def _create_agent(self):
        """Create the LangChain agent with research tools"""
        
        # System prompt for the agent
        system_prompt = """You are an expert research analyst. Your job is to research topics thoroughly and produce structured, accurate reports. When researching:

- Search for the topic from multiple angles
- Look for recent data and statistics
- Find expert opinions and credible sources
- Synthesize information critically
- Always cite your sources

Use the search tool to gather comprehensive information, then provide a detailed analysis of your findings."""
        
        # Create agent with tools using create_agent (LangChain 1.2.15 API)
        tools = [self.search_tool]
        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt
        )
        
        return agent
    
    async def research(self, topic: str, num_searches: int = 3) -> Dict[str, Any]:
        """
        Research a topic and generate a structured report
        
        Args:
            topic: The research topic
            num_searches: Number of searches to perform (3 for quick, 7 for deep)
            
        Returns:
            Structured research report as JSON
        """
        logger.info(f"Starting research on topic: {topic} with {num_searches} searches")
        
        # Check if search tool is initialized
        if not self.search_tool:
            logger.error("Search tool not initialized - missing TAVILY_API_KEY")
            return {
                "title": f"Research Report: {topic}",
                "summary": "Research cannot be performed due to missing TAVILY_API_KEY. Please set TAVILY_API_KEY environment variable.",
                "key_findings": ["Tavily API key missing", "Cannot perform web search"],
                "sections": [
                    {
                        "heading": "Configuration Error",
                        "content": "The research agent requires TAVILY_API_KEY environment variable to be set. Please configure this and restart the server."
                    }
                ],
                "sources": [],
                "generated_at": datetime.now().isoformat(),
                "topic": topic
            }
        
        sources = []
        all_results = []
        
        search_queries = self._generate_search_queries(topic, num_searches)
        
        for i, query in enumerate(search_queries):
            logger.info(f"Performing search {i+1}/{num_searches}: {query}")
            try:
                # Call Tavily directly — this returns structured results
                raw_results = await self.search_tool.ainvoke({"query": query})
                
                # raw_results is a list of dicts with url and content
                if isinstance(raw_results, list):
                    for item in raw_results:
                        if isinstance(item, dict):
                            url = item.get("url", "")
                            title = item.get("title", "") or url
                            content = item.get("content", "")
                            
                            # Add to sources if URL is new
                            if url and url not in [s["url"] for s in sources]:
                                sources.append({
                                        "title": title,
                                        "url": url
                                    })
                            
                            # Add content to research data
                            if content:
                                all_results.append(f"Source: {url}\n{content}")
                
                elif isinstance(raw_results, str):
                    all_results.append(raw_results)
                    
            except Exception as e:
                logger.error(f"Search error for query '{query}': {e}")
                continue
        
        logger.info(f"Collected {len(sources)} unique sources")
        logger.info(f"Sources collected: {sources}")
        
        combined_research = "\n\n".join(all_results)
        
        # Generate structured report using Claude
        report = await self._generate_structured_report(
            topic=topic,
            research_data=combined_research,
            sources=sources
        )
        
        logger.info(f"Research completed for topic: {topic}")
        return report
    
    def _generate_search_queries(self, topic: str, num_searches: int) -> List[str]:
        """Generate diverse search queries for comprehensive research"""
        
        base_queries = [
            f"{topic} overview and key facts",
            f"{topic} recent developments and trends 2024",
            f"{topic} expert analysis and opinions",
            f"{topic} statistics and data",
            f"{topic} challenges and opportunities",
            f"{topic} future outlook and predictions",
            f"{topic} case studies and real-world examples"
        ]
        
        return base_queries[:num_searches]
    
    async def _generate_structured_report(
        self, 
        topic: str, 
        research_data: str, 
        sources: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Generate a structured research report using Claude"""
        
        report_prompt = f"""
Based on the following research data, generate a structured research report in JSON format with this exact structure:

{{
    "title": "Research Report: {topic}",
    "summary": "2-3 sentence executive summary",
    "key_findings": ["finding 1", "finding 2", "finding 3"],
    "sections": [
        {{
            "heading": "section title",
            "content": "detailed paragraph"
        }},
        {{
            "heading": "another section title", 
            "content": "another detailed paragraph"
        }}
    ],
    "sources": [],
    "generated_at": "{{datetime.now().isoformat()}}",
    "topic": "{topic}"
}}

Research Data:
{research_data}

Please generate the complete JSON report following this exact structure. Ensure all quotes are properly escaped and the JSON is valid.
"""
        
        # Use Claude to generate the structured report
        response = await self.llm.ainvoke([
            SystemMessage(content="You are an expert research analyst who creates structured JSON reports. Always return valid JSON."),
            ("human", report_prompt)
        ])
        
        # Parse the response to ensure it's valid JSON
        try:
            import json
            report_content = response.content.strip()
            
            # Remove any markdown code blocks if present
            if report_content.startswith("```json"):
                report_content = report_content[7:]
            if report_content.endswith("```"):
                report_content = report_content[:-3]
            report_content = report_content.strip()
            
            report = json.loads(report_content)
            logger.info("Successfully generated structured report")
            
            # Add real sources to the report
            report["sources"] = sources[:8]  # Limit to 8 sources
            
            # Deduplicate sources by URL
            seen_urls = set()
            unique_sources = []
            for s in sources:
                if s["url"] and s["url"] not in seen_urls:
                    seen_urls.add(s["url"])
                    unique_sources.append(s)
            report["sources"] = unique_sources
            
            return report
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            # Return a basic structure if JSON parsing fails
            return {
                "title": f"Research Report: {topic}",
                "summary": "Research completed but report generation encountered an error.",
                "key_findings": ["Research data collected", "Analysis performed"],
                "sections": [
                    {
                        "heading": "Research Overview",
                        "content": "Research was conducted on the requested topic. Please try again for detailed results."
                    }
                ],
                "sources": sources[:8],  # Limit to 8 sources
                "generated_at": datetime.now().isoformat(),
                "topic": topic
            }
