import logging
import os
from datetime import datetime
from typing import Dict, Any, List

from langchain_anthropic import ChatAnthropic
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self):
        """Initialize the research agent with Tavily search and Claude LLM"""
        logger.info("Initializing ResearchAgent")
        
        # Initialize Tavily search tool
        self.search_tool = TavilySearchResults(
            max_results=5,
            description="Search the web for current information on any topic"
        )
        
        # Initialize Claude LLM
        self.llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=0.1,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        # Create the agent
        self.agent = self._create_agent()
        
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
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create agent with tools
        tools = [self.search_tool]
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        return agent_executor
    
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
        
        try:
            # Generate search queries for different angles
            search_queries = self._generate_search_queries(topic, num_searches)
            
            # Gather information from multiple searches
            all_results = []
            sources = []
            
            for i, query in enumerate(search_queries):
                logger.info(f"Performing search {i+1}/{num_searches}: {query}")
                
                # Use the agent to search
                result = await self.agent.ainvoke({
                    "input": f"Search for: {query}. Find recent, credible information with sources."
                })
                
                all_results.append(result.get('output', ''))
                
                # Extract sources from the search results
                if 'intermediate_steps' in result:
                    for step in result['intermediate_steps']:
                        if hasattr(step[1], 'get') and step[1].get('url'):
                            sources.append({
                                "title": step[1].get('title', 'Unknown'),
                                "url": step[1].get('url')
                            })
            
            # Combine all research results
            combined_research = "\n\n".join(all_results)
            
            # Generate structured report using Claude
            report = await self._generate_structured_report(
                topic=topic,
                research_data=combined_research,
                sources=sources
            )
            
            logger.info(f"Research completed for topic: {topic}")
            return report
            
        except Exception as e:
            logger.error(f"Error during research: {str(e)}")
            raise e
    
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
    "sources": {sources},
    "generated_at": "{datetime.now().isoformat()}",
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
                "sources": sources,
                "generated_at": datetime.now().isoformat(),
                "topic": topic
            }
