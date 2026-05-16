import os
from pathlib import Path
import json

class KnowledgeBase:
    def __init__(self, workspace_root: str):
        self.workspace = Path(workspace_root).resolve()
        self.knowledge_dir = self.workspace / ".999" / "knowledge"
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    def save_knowledge(self, topic: str, content: str) -> str:
        """Saves a piece of knowledge/fact about the project to a markdown file."""
        # Sanitize filename
        safe_topic = "".join([c for c in topic if c.isalpha() or c.isdigit() or c in (' ', '_', '-')]).rstrip()
        safe_topic = safe_topic.replace(' ', '_').lower()
        
        if not safe_topic:
            return "Error: Invalid topic name."
            
        file_path = self.knowledge_dir / f"{safe_topic}.md"
        try:
            file_path.write_text(content, encoding='utf-8')
            return f"Successfully saved knowledge about '{topic}'."
        except Exception as e:
            return f"Error saving knowledge: {str(e)}"

    def read_knowledge(self, topic: str) -> str:
        """Reads a stored piece of knowledge."""
        safe_topic = topic.replace(' ', '_').lower()
        file_path = self.knowledge_dir / f"{safe_topic}.md"
        
        if not file_path.exists():
            return f"No knowledge found about '{topic}'."
            
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception as e:
            return f"Error reading knowledge: {str(e)}"

    def list_knowledge(self) -> str:
        """Lists all stored knowledge topics."""
        if not self.knowledge_dir.exists():
            return "No knowledge stored yet."
            
        files = list(self.knowledge_dir.glob("*.md"))
        if not files:
            return "No knowledge stored yet."
            
        topics = [f.stem.replace('_', ' ') for f in files]
        return "Stored Knowledge Topics:\n" + "\n".join([f"- {t}" for t in topics])
