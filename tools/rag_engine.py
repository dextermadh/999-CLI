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

    def index_workspace(self) -> str:
        """Chunks and indexes the entire workspace for semantic search."""
        self._lazy_load()
        
        chunks = []
        metadata = []
        
        for file_path in self.workspace.rglob("*.*"):
            # Skip ignored directories and binary files
            if any(part in self.ignore_dirs for part in file_path.parts):
                continue
            if not file_path.is_file():
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8')
                # Simple chunking by lines (e.g. 50 lines per chunk)
                lines = content.split('\n')
                chunk_size = 50
                for i in range(0, len(lines), chunk_size):
                    chunk = '\n'.join(lines[i:i+chunk_size])
                    if chunk.strip():
                        chunks.append(chunk)
                        metadata.append({
                            "file": str(file_path.relative_to(self.workspace)),
                            "start_line": i + 1,
                            "end_line": min(i + chunk_size, len(lines))
                        })
            except Exception:
                continue # Skip binary/unreadable files

        if not chunks:
            return "Workspace is empty or unreadable."

        # Generate embeddings
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        
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

    def semantic_search(self, query: str, top_k: int = 3) -> str:
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
