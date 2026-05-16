import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress OneDNN logs
import ast
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

class RAGEngine:
    def __init__(self, workspace_root: str):
        self.workspace = Path(workspace_root).resolve()
        self.ignore_dirs = {'.git', '__pycache__', 'venv', 'env', 'node_modules', '.999'}
        self.index_dir = self.workspace / ".999" / "vector_store"
        self.model = None
        self.index = None
        self.metadata = []
        self.bm25 = None  # Cached BM25 index (Phase 6)
        self._indexed_at = 0  # Timestamp of last index build
        self._dep_graph = {}  # Dependency graph: file -> [imported files]
        # Eager load models to avoid cold start latency
        try:
            self._lazy_load()
        except Exception:
            pass

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

    # ================================================================
    #  CHUNKING STRATEGIES
    # ================================================================

    def _chunk_python_ast(self, content: str, file_path: Path) -> List[str]:
        """AST-aware chunking for Python files (Phase 5).
        Extracts each top-level function/class as its own chunk, including
        decorators and docstrings. Falls back to heuristic chunking on parse failure.
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._chunk_code(content, file_path)

        lines = content.split('\n')
        chunks = []
        used_lines = set()

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Get start line (including decorators)
                start = node.lineno
                if node.decorator_list:
                    start = node.decorator_list[0].lineno
                end = node.end_lineno or node.lineno

                chunk_lines = lines[start - 1:end]
                chunk_text = '\n'.join(chunk_lines)

                if chunk_text.strip():
                    chunks.append(chunk_text)
                    used_lines.update(range(start, end + 1))

                # For classes, also create sub-chunks per method
                if isinstance(node, ast.ClassDef):
                    for item in ast.iter_child_nodes(node):
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            m_start = item.lineno
                            if item.decorator_list:
                                m_start = item.decorator_list[0].lineno
                            m_end = item.end_lineno or item.lineno
                            method_lines = lines[m_start - 1:m_end]
                            method_text = '\n'.join(method_lines)
                            if method_text.strip() and len(method_text) > 50:
                                chunks.append(method_text)

        # Capture module-level code (imports, constants, etc.) that aren't in any def/class
        module_lines = []
        for i, line in enumerate(lines, 1):
            if i not in used_lines:
                module_lines.append(line)
            else:
                if module_lines and any(l.strip() for l in module_lines):
                    chunk_text = '\n'.join(module_lines)
                    if len(chunk_text.strip()) > 20:
                        chunks.append(chunk_text)
                    module_lines = []

        if module_lines and any(l.strip() for l in module_lines):
            chunk_text = '\n'.join(module_lines)
            if len(chunk_text.strip()) > 20:
                chunks.append(chunk_text)

        return [c for c in chunks if c.strip()] if chunks else self._chunk_code(content, file_path)

    def _chunk_code(self, content: str, file_path: Path) -> List[str]:
        """Smart chunking that respects function and class boundaries with overlap."""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        max_chunk_size = 50
        overlap_size = 10
        
        # Simple heuristic: split on class/def at the start of a line
        for line in lines:
            # If we see a new top-level definition and we already have some content, start a new chunk
            if (line.startswith('def ') or line.startswith('class ') or line.startswith('async def ')) and current_size > 20:
                chunks.append('\n'.join(current_chunk))
                # Keep overlap
                current_chunk = current_chunk[-overlap_size:] if len(current_chunk) >= overlap_size else current_chunk
                current_size = len(current_chunk)
            
            current_chunk.append(line)
            current_size += 1
            
            if current_size >= max_chunk_size:
                chunks.append('\n'.join(current_chunk))
                # Keep overlap
                current_chunk = current_chunk[-overlap_size:] if len(current_chunk) >= overlap_size else current_chunk
                current_size = len(current_chunk)
        
        if current_chunk and current_size > overlap_size:
            chunks.append('\n'.join(current_chunk))
        
        return [c for c in chunks if c.strip()]

    # ================================================================
    #  FILE-LEVEL SUMMARY GENERATION (Phase 7)
    # ================================================================

    def _generate_file_summary(self, content: str, file_path: Path) -> str:
        """Generates a synthetic summary chunk for a file.
        Includes the file path, imports, and a list of all classes/functions.
        This enables the system to answer 'what does X file do?' queries.
        """
        ext = file_path.suffix.lower()
        rel_path = str(file_path.relative_to(self.workspace))
        parts = [f"FILE SUMMARY: {rel_path}"]

        if ext == '.py':
            try:
                tree = ast.parse(content)
                # Extract imports
                imports = []
                symbols = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ''
                        imports.append(module)
                    elif isinstance(node, ast.ClassDef):
                        symbols.append(f"class {node.name}")
                    elif isinstance(node, ast.FunctionDef):
                        symbols.append(f"def {node.name}()")
                    elif isinstance(node, ast.AsyncFunctionDef):
                        symbols.append(f"async def {node.name}()")

                if imports:
                    parts.append("Imports: " + ", ".join(imports[:20]))
                if symbols:
                    parts.append("Symbols: " + ", ".join(symbols))
            except SyntaxError:
                pass

            # Get module docstring
            lines = content.split('\n')
            docstring_lines = []
            in_docstring = False
            for line in lines[:30]:
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    if in_docstring:
                        docstring_lines.append(stripped)
                        break
                    elif stripped.endswith('"""') and len(stripped) > 3:
                        docstring_lines.append(stripped[3:-3])
                        break
                    else:
                        in_docstring = True
                        docstring_lines.append(stripped[3:])
                elif in_docstring:
                    docstring_lines.append(stripped)
                elif stripped.startswith('#'):
                    docstring_lines.append(stripped[1:].strip())

            if docstring_lines:
                parts.append("Description: " + " ".join(docstring_lines[:5]))
        else:
            # For non-Python files, use the first few meaningful lines as description
            lines = [l.strip() for l in content.split('\n') if l.strip()][:10]
            if lines:
                parts.append("Preview: " + " ".join(lines)[:300])

        return "\n".join(parts)

    # ================================================================
    #  INDEXING
    # ================================================================

    def index_workspace(self) -> str:
        """Chunks and indexes the entire workspace, including documents and spreadsheets."""
        self._lazy_load()
        
        chunks = []
        metadata = []
        
        # Extended list of supported extensions
        code_exts = {'.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css', '.md', '.go', '.rs', '.java', '.c', '.cpp', '.h', '.sql', '.sh', '.json', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.env', '.bat', '.ps1'}
        doc_exts = {'.pdf', '.docx', '.xlsx', '.csv'}
        
        from tools.file_manager import LocalFileManager
        fm = LocalFileManager(str(self.workspace))

        file_count = 0
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

                file_count += 1

                # --- Phase 7: Generate a file-level summary chunk ---
                summary = self._generate_file_summary(content, file_path)
                if summary:
                    chunks.append(summary)
                    metadata.append({
                        "file": str(file_path.relative_to(self.workspace)),
                        "start_line": 1,
                        "end_line": 1,
                        "type": "summary",
                        "text": summary
                    })

                # --- Chunk the file content ---
                if ext == '.py':
                    # Phase 5: AST-aware chunking for Python
                    file_chunks = self._chunk_python_ast(content, file_path)
                elif ext in code_exts:
                    file_chunks = self._chunk_code(content, file_path)
                else:
                    # For documents, use paragraph-based chunking with a length-based fallback
                    paragraphs = [c.strip() for c in content.split('\n\n') if c.strip()]
                    file_chunks = []
                    for p in paragraphs:
                        if len(p) > 2000:
                            # Split large paragraphs into smaller chunks with overlap
                            start = 0
                            chunk_size = 2000
                            overlap = 400
                            while start < len(p):
                                end = start + chunk_size
                                file_chunks.append(p[start:end])
                                start += (chunk_size - overlap)
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
                        "end_line": start_line + chunk_lines - 1,
                        "type": "code",
                        "text": chunk
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
        
        # --- Phase 6: Build and cache BM25 index ---
        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [m['text'].split() for m in self.metadata]
            self.bm25 = BM25Okapi(tokenized_corpus)
        except Exception:
            self.bm25 = None

        # Save to disk with timestamp
        self._indexed_at = time.time()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, str(self.index_dir / "code_index.faiss"))
        with open(self.index_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump({"indexed_at": self._indexed_at, "entries": self.metadata}, f)
            
        return f"Successfully indexed {len(chunks)} chunks ({file_count} files) across the workspace."

    def incremental_index(self) -> str:
        """Re-indexes only files that have changed since the last full index.
        Falls back to full index if no previous index exists.
        """
        self._lazy_load()

        # If no index exists, do a full index
        if not (self.index_dir / "code_index.faiss").exists():
            return self.index_workspace()

        # Load existing index
        if self.index is None:
            if not self._load_index_from_disk():
                return self.index_workspace()

        if self._indexed_at == 0:
            return self.index_workspace()

        # Find files changed since last index
        code_exts = {'.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css', '.md', '.go', '.rs', '.java', '.c', '.cpp', '.h', '.sql', '.sh', '.json', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.env', '.bat', '.ps1'}
        doc_exts = {'.pdf', '.docx', '.xlsx', '.csv'}
        changed_files = []

        for file_path in self.workspace.rglob("*.*"):
            if any(part in self.ignore_dirs for part in file_path.parts):
                continue
            if not file_path.is_file():
                continue
            ext = file_path.suffix.lower()
            if ext not in code_exts and ext not in doc_exts:
                continue
            try:
                if file_path.stat().st_mtime > self._indexed_at:
                    changed_files.append(file_path)
            except Exception:
                continue

        if not changed_files:
            return "Index is up to date. No files changed since last index."

        # For now, if files changed, do a full re-index.
        # A true incremental approach (patching the FAISS index) is complex
        # because removing/updating vectors in FAISS IndexFlatL2 requires rebuilding.
        return self.index_workspace()

    # ================================================================
    #  DEPENDENCY GRAPH
    # ================================================================

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Builds a dependency graph mapping each Python file to its local imports.
        This enables the system to understand which files are connected.
        """
        graph = {}
        all_files = set()

        # Collect all Python files
        for file_path in self.workspace.rglob("*.py"):
            if any(part in self.ignore_dirs for part in file_path.parts):
                continue
            if file_path.is_file():
                rel = str(file_path.relative_to(self.workspace)).replace('\\', '/')
                all_files.add(rel)

        # Parse imports from each file
        for file_path in self.workspace.rglob("*.py"):
            if any(part in self.ignore_dirs for part in file_path.parts):
                continue
            if not file_path.is_file():
                continue

            rel = str(file_path.relative_to(self.workspace)).replace('\\', '/')
            deps = []

            try:
                content = file_path.read_text(encoding='utf-8')
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    module = None
                    if isinstance(node, ast.ImportFrom) and node.module:
                        module = node.module
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            module = alias.name

                    if module:
                        # Convert module path to file path
                        mod_path = module.replace('.', '/') + '.py'
                        mod_init = module.replace('.', '/') + '/__init__.py'
                        if mod_path in all_files:
                            deps.append(mod_path)
                        elif mod_init in all_files:
                            deps.append(mod_init)
            except Exception:
                pass

            if deps:
                graph[rel] = list(set(deps))

        self._dep_graph = graph
        return graph

    def get_dependency_graph(self) -> str:
        """Returns the dependency graph as a formatted string."""
        graph = self._build_dependency_graph()
        if not graph:
            return "No import dependencies found between workspace files."

        lines = ["### Dependency Graph (Local Imports)\n"]
        for file, deps in sorted(graph.items()):
            lines.append(f"**{file}**")
            for d in sorted(deps):
                lines.append(f"  -> {d}")
        lines.append(f"\n*{len(graph)} files with local dependencies mapped.*")
        return "\n".join(lines)

    # ================================================================
    #  SEARCH
    # ================================================================

    def _get_global_summary(self) -> str:
        """Returns a high-level summary of the codebase by reading README and structure."""
        results = ["### Codebase Overview (Global Summary)\n"]
        
        # 1. Read README.md
        readme_path = self.workspace / "README.md"
        if readme_path.exists():
            try:
                content = readme_path.read_text(encoding='utf-8')
                results.append(f"#### README.md\n```markdown\n{content[:3000]}\n```\n")
            except Exception:
                results.append("#### README.md (Error reading file)\n")
        
        # 2. Get structure from CodeAnalyzer
        try:
            from tools.code_analyzer import CodeAnalyzer
            analyzer = CodeAnalyzer(str(self.workspace))
            structure = analyzer.map_codebase(max_depth=3)
            results.append(f"#### Directory Structure\n```\n{structure}\n```\n")
        except Exception:
            results.append("#### Directory Structure (Error generating structure)\n")

        # 3. Pull file summaries from the index if available
        if self.metadata:
            summary_chunks = [m for m in self.metadata if m.get("type") == "summary"]
            if summary_chunks:
                results.append("#### File Summaries\n")
                for sc in summary_chunks[:20]:
                    results.append(f"- {sc['text'][:200]}")
            
        return "\n\n".join(results)

    def _load_index_from_disk(self) -> bool:
        """Loads the FAISS index and metadata from disk. Returns True on success."""
        if not (self.index_dir / "code_index.faiss").exists():
            return False
        try:
            self.index = self.faiss.read_index(str(self.index_dir / "code_index.faiss"))
            with open(self.index_dir / "metadata.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Support both old format (list) and new format (dict with timestamp)
            if isinstance(data, dict):
                self.metadata = data.get("entries", [])
                self._indexed_at = data.get("indexed_at", 0)
            else:
                self.metadata = data
                self._indexed_at = 0

            # Rebuild BM25 from cached text
            if self.metadata and 'text' in self.metadata[0]:
                try:
                    from rank_bm25 import BM25Okapi
                    tokenized_corpus = [m['text'].split() for m in self.metadata]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                except Exception:
                    self.bm25 = None
            return True
        except Exception:
            return False

    def semantic_search(self, query: str, top_k: int = 10) -> str:
        """Searches the vector index for code snippets matching the natural language query."""
        self._lazy_load()
        
        # Intent detection for global queries
        global_keywords = [
            "about this codebase", "what is this codebase", "summary of the project",
            "overview of the project", "what does this project do", "tell me about",
            "describe this project", "what is this project", "codebase summary",
            "project overview", "what's in this repo"
        ]
        if any(kw in query.lower() for kw in global_keywords):
            return self._get_global_summary()
            
        # Phase 8: Auto-index if no index exists
        if self.index is None:
            if not self._load_index_from_disk():
                # Auto-index the workspace
                index_result = self.index_workspace()
                if self.index is None:
                    return f"Auto-indexing attempted but failed: {index_result}"

        # Staleness warning: check if any file is newer than the index
        if self._indexed_at > 0:
            stale = False
            for file_path in self.workspace.rglob("*.py"):
                if any(part in self.ignore_dirs for part in file_path.parts):
                    continue
                try:
                    if file_path.stat().st_mtime > self._indexed_at:
                        stale = True
                        break
                except Exception:
                    continue
            # If stale, silently re-index for accuracy
            if stale:
                self.index_workspace()

        # 1. FAISS Search
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        # Retrieve more than top_k for fusion
        faiss_k = min(top_k * 3, len(self.metadata))
        distances, indices = self.index.search(query_embedding, faiss_k)
        
        faiss_results = list(zip(indices[0], distances[0]))
        
        # 2. BM25 Search (using cached index — Phase 6)
        bm25_results = []
        if self.bm25 is not None:
            tokenized_query = query.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_indices = np.argsort(bm25_scores)[::-1][:faiss_k]
            bm25_results = [(idx, bm25_scores[idx]) for idx in bm25_indices]

        # 3. Reciprocal Rank Fusion (RRF)
        scores = {}
        for rank, (idx, dist) in enumerate(faiss_results):
            scores[idx] = scores.get(idx, 0) + 1 / (rank + 60)
            
        for rank, (idx, score) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0) + 1 / (rank + 60)

        # Sort by RRF score
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

        results = []
        for idx in sorted_indices:
            meta = self.metadata[idx]
            chunk_type = meta.get("type", "code")
            file_path = self.workspace / meta['file']
            try:
                if 'text' in meta:
                    snippet = meta['text']
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    snippet = "".join(lines[meta['start_line']-1:meta['end_line']])
                
                type_label = "📄 Summary" if chunk_type == "summary" else f"Lines {meta['start_line']}-{meta['end_line']}"
                results.append(f"### File: {meta['file']} ({type_label})\n```\n{snippet.strip()}\n```\n")
            except Exception:
                results.append(f"### File: {meta['file']} (Unreadable or modified)\n")
                
        return "\n".join(results)
