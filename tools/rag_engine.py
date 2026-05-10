import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress OneDNN logs
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

class RAGEngine:
    def __init__(self, workspace_root: str):
        self.workspace = Path(workspace_root).resolve()
        self.ignore_dirs = {'.git', '__pycache__', 'venv', 'env', 'node_modules', '.999'}
        self.index_dir = self.workspace / ".999" / "vector_store"
        self.model = None
        self.index = None
        self.metadata = []

    def _lazy_load(self):
        """Lazy load heavy ML models only when needed."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import faiss
                # Use a very small, fast model for local CPU execution
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.faiss = faiss
            except ImportError:
                raise ImportError("Please install sentence-transformers and faiss-cpu")

    def _chunk_code(self, content: str, file_path: Path) -> List[str]:
        """Smart chunking that respects function and class boundaries."""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        max_chunk_size = 50
        
        # Simple heuristic: split on class/def at the start of a line
        for line in lines:
            # If we see a new top-level definition and we already have some content, start a new chunk
            if (line.startswith('def ') or line.startswith('class ') or line.startswith('async def ')) and current_size > 20:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(line)
            current_size += 1
            
            if current_size >= max_chunk_size:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return [c for c in chunks if c.strip()]

    def index_workspace(self) -> str:
        """Chunks and indexes the entire workspace, including documents and spreadsheets."""
        self._lazy_load()
        
        chunks = []
        metadata = []
        
        # Extended list of supported extensions
        code_exts = {'.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css', '.md', '.go', '.rs', '.java', '.c', '.cpp', '.h', '.sql', '.sh'}
        doc_exts = {'.pdf', '.docx', '.xlsx', '.csv'}
        
        from tools.file_manager import LocalFileManager
        fm = LocalFileManager(str(self.workspace))

        for file_path in self.workspace.rglob("*.*"):
            # Skip ignored directories
            if any(part in self.ignore_dirs for part in file_path.parts):
                continue
            if not file_path.is_file():
                continue
            
            ext = file_path.suffix.lower()
            if ext not in code_exts and ext not in doc_exts:
                continue
                
            try:
                # Use LocalFileManager's unified reader
                content = fm.read_file(str(file_path.relative_to(self.workspace)))
                if content.startswith("Error:"):
                    continue

                if ext in code_exts:
                    file_chunks = self._chunk_code(content, file_path)
                else:
                    # For documents, use paragraph-based chunking with a length-based fallback
                    paragraphs = [c.strip() for c in content.split('\n\n') if c.strip()]
                    file_chunks = []
                    for p in paragraphs:
                        if len(p) > 2000:
                            # Split large paragraphs into smaller chunks
                            file_chunks.extend([p[i:i+2000] for i in range(0, len(p), 2000)])
                        else:
                            file_chunks.append(p)
                
                start_line = 1
                for chunk in file_chunks:
                    # Truncate extremely large chunks to avoid embedding issues
                    if len(chunk) > 4000:
                        chunk = chunk[:4000]
                        
                    chunk_lines = chunk.count('\n') + 1
                    chunks.append(chunk)
                    metadata.append({
                        "file": str(file_path.relative_to(self.workspace)),
                        "start_line": start_line,
                        "end_line": start_line + chunk_lines - 1
                    })
                    start_line += chunk_lines
            except Exception:
                continue 

        if not chunks:
            return "Workspace is empty or has no supported code files."

        # Generate embeddings
        embeddings = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = self.faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.metadata = metadata
        
        # Save to disk
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, str(self.index_dir / "code_index.faiss"))
        with open(self.index_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f)
            
        return f"Successfully indexed {len(chunks)} chunks across the workspace."

    def semantic_search(self, query: str, top_k: int = 10) -> str:
        """Searches the vector index for code snippets matching the natural language query."""
        self._lazy_load()
        
        # Try loading from disk if not in memory
        if self.index is None:
            if not (self.index_dir / "code_index.faiss").exists():
                return "Error: Index not found. Please run 'index_workspace' first."
            self.index = self.faiss.read_index(str(self.index_dir / "code_index.faiss"))
            with open(self.index_dir / "metadata.json", 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.metadata)))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            meta = self.metadata[idx]
            file_path = self.workspace / meta['file']
            try:
                # Read the actual lines from the file to ensure freshness
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                snippet = "".join(lines[meta['start_line']-1:meta['end_line']])
                results.append(f"### File: {meta['file']} (Lines {meta['start_line']}-{meta['end_line']})\n```\n{snippet.strip()}\n```\n")
            except Exception:
                results.append(f"### File: {meta['file']} (Unreadable or modified)\n")
                
        return "\n".join(results)
