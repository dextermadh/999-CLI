import os
import sys
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any, List

# Add parent directory to path so we can import tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.file_manager import LocalFileManager
from tools.rag_engine import RAGEngine
from tools.knowledge_base import KnowledgeBase
from tools.artifact_manager import ArtifactManager
from tools.web_tools import WebTools

app = FastAPI(title="999-CLI MCP Server")

workspace_path = os.getcwd()
file_manager = LocalFileManager(project_root=workspace_path)
rag_engine = RAGEngine(workspace_root=workspace_path)
knowledge_base = KnowledgeBase(workspace_root=workspace_path)
artifact_manager = ArtifactManager(workspace_root=workspace_path)
web_tools = WebTools()

class ToolCall(BaseModel):
    tool: str
    arguments: Dict[str, Any] = {}

@app.get("/")
def read_root():
    return {"message": "999-CLI MCP Server is running", "workspace": workspace_path}

@app.get("/tools")
def list_tools():
    """Lists available tools (MCP standard)."""
    return {
        "tools": [
            {"name": "read_file", "description": "Read content of a file", "parameters": {"path": "string"}},
            {"name": "write_file", "description": "Write content to a file", "parameters": {"path": "string", "content": "string"}},
            {"name": "list_files", "description": "List files in directory", "parameters": {"path": "string"}},
            {"name": "semantic_search", "description": "Search codebase semantically", "parameters": {"query": "string"}},
            {"name": "save_knowledge", "description": "Save a fact to knowledge base", "parameters": {"topic": "string", "content": "string"}},
            {"name": "read_knowledge", "description": "Read a fact from knowledge base", "parameters": {"topic": "string"}},
            {"name": "create_artifact", "description": "Create a persistent document", "parameters": {"title": "string", "content": "string"}}
        ]
    }

@app.post("/tools/execute")
def execute_tool(call: ToolCall):
    """Executes a tool (MCP standard)."""
    name = call.tool
    args = call.arguments
    
    try:
        if name == "read_file":
            return {"result": file_manager.read_file(args.get("path"))}
        elif name == "write_file":
            return {"result": file_manager.write_file(args.get("path"), args.get("content", ""))}
        elif name == "list_files":
            return {"result": file_manager.list_files(args.get("path", "."))}
        elif name == "semantic_search":
            return {"result": rag_engine.semantic_search(args.get("query", ""))}
        elif name == "save_knowledge":
            return {"result": knowledge_base.save_knowledge(args.get("topic", ""), args.get("content", ""))}
        elif name == "read_knowledge":
            return {"result": knowledge_base.read_knowledge(args.get("topic", ""))}
        elif name == "create_artifact":
            return {"result": artifact_manager.create_artifact(args.get("title", ""), args.get("content", ""))}
        else:
            raise HTTPException(status_code=404, detail=f"Tool {name} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
