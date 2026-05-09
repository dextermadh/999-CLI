from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict): 
    # Maintains the conversation history (Standard LangGraph)
    messages: Annotated[List[BaseMessage], operator.add]

    # The current local directory context (Essential for local-first)
    current_dir: str

    # Track which tools are enabled (Good for your Ethics logic)
    allowed_tools: List[str]

    # The reasoning phase - Gemma 4's <|think|> output goes here
    internal_monologue: str

    # --- Safety & Control ---
    
    # Stores the Risk Assessment object from your Ethics layer
    risk_assessment: Optional[dict]

    # The specific file the agent is currently focused on
    active_file: Optional[str]

    # A flag to indicate if a human needs to approve the next step
    requires_approval: bool
    
    # Approval mode: "default" | "yolo" | "safe"
    approval_mode: str

    # --- Context & Analysis ---
    
    # Context gathered by the CodeAnalyzer
    analysis_result: Optional[str]
    
    # Output from the Terminal verification step
    verification_result: Optional[str]
    
    # --- Configuration ---
    
    # The name of the LLM model to use
    model_name: str
    
    # --- Telemetry ---
    
    # Token usage tracking per turn
    token_usage: Optional[dict]