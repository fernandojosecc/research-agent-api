import logging
import os
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from agent import ResearchAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Research Agent API",
    description="An AI-powered research agent that generates structured reports",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ResearchRequest(BaseModel):
    topic: str
    depth: str  # "quick" or "deep"

class HealthResponse(BaseModel):
    status: str

# Initialize research agent
research_agent = ResearchAgent()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return {"status": "ok"}

@app.post("/research")
async def research_topic(request: ResearchRequest) -> Dict[str, Any]:
    """
    Research a topic and return a structured report
    
    Args:
        request: ResearchRequest containing topic and depth
        
    Returns:
        Structured research report as JSON
    """
    logger.info(f"Research request received for topic: {request.topic}, depth: {request.depth}")
    
    try:
        # Validate depth parameter
        if request.depth not in ["quick", "deep"]:
            raise HTTPException(
                status_code=400,
                detail="Depth must be either 'quick' or 'deep'"
            )
        
        # Set number of searches based on depth
        num_searches = 3 if request.depth == "quick" else 7
        
        logger.info(f"Starting research with {num_searches} searches")
        
        # Generate research report
        report = await research_agent.research(
            topic=request.topic,
            num_searches=num_searches
        )
        
        logger.info(f"Research completed for topic: {request.topic}")
        return report
        
    except Exception as e:
        logger.error(f"Error during research: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during research: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
