import os
from pathlib import Path
import json
import time
import re

class ArtifactManager:
    def __init__(self, workspace_root: str):
        self.workspace = Path(workspace_root).resolve()
        self.artifact_dir = self.workspace / ".999" / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_id(self, title: str) -> str:
        """Converts a title into a safe filename slug."""
        slug = "".join([c for c in title if c.isalpha() or c.isdigit() or c in (' ', '_', '-')]).rstrip()
        return slug.replace(' ', '_').lower()

    def create_artifact(self, title: str, content: str, artifact_type: str = "other") -> str:
        """Creates a new persistent document (artifact)."""
        artifact_id = self._sanitize_id(title)
        if not artifact_id:
            return "Error: Invalid artifact title."
            
        file_path = self.artifact_dir / f"{artifact_id}.md"
        
        if file_path.exists():
            return f"Error: Artifact with ID '{artifact_id}' already exists."
            
        header = f"""---
id: {artifact_id}
title: {title}
type: {artifact_type}
created_at: {time.strftime('%Y-%m-%d %H:%M:%S')}
---

"""
        try:
            file_path.write_text(header + content, encoding='utf-8')
            return f"Successfully created artifact '{title}' (ID: {artifact_id})."
        except Exception as e:
            return f"Error creating artifact: {str(e)}"

    def update_artifact(self, artifact_id: str, content: str) -> str:
        """Updates an existing artifact. Overwrites content but preserves header if possible."""
        file_path = self.artifact_dir / f"{artifact_id}.md"
        
        if not file_path.exists():
            return f"Error: Artifact '{artifact_id}' not found."
            
        try:
            old_content = file_path.read_text(encoding='utf-8')
            # Try to preserve header
            match = re.match(r'^(---\n.*?\n---\n)', old_content, re.DOTALL)
            header = match.group(1) if match else ""
            
            if not header:
                header = f"""---
id: {artifact_id}
updated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}
---

"""
            file_path.write_text(header + content, encoding='utf-8')
            return f"Successfully updated artifact '{artifact_id}'."
        except Exception as e:
            return f"Error updating artifact: {str(e)}"

    def read_artifact(self, artifact_id: str) -> str:
        """Reads the content of an artifact."""
        file_path = self.artifact_dir / f"{artifact_id}.md"
        
        if not file_path.exists():
            return f"Error: Artifact '{artifact_id}' not found."
            
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception as e:
            return f"Error reading artifact: {str(e)}"

    def list_artifacts(self) -> str:
        """Lists all artifacts with their summaries."""
        if not self.artifact_dir.exists():
            return "No artifacts created yet."
            
        files = list(self.artifact_dir.glob("*.md"))
        if not files:
            return "No artifacts created yet."
            
        results = ["### Available Artifacts\n"]
        for f in files:
            try:
                content = f.read_text(encoding='utf-8')
                # Extract title and type from header
                title = f.stem.replace('_', ' ')
                a_type = "other"
                match = re.search(r'title:\s*(.*?)\n', content)
                if match: title = match.group(1).strip()
                match = re.search(r'type:\s*(.*?)\n', content)
                if match: a_type = match.group(1).strip()
                
                results.append(f"- **{title}** (ID: `{f.stem}`) | Type: {a_type}")
            except Exception:
                continue
                
        return "\n".join(results)
