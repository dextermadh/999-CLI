import os
import re
from pathlib import Path
from typing import List

class CodeAnalyzer:
    def __init__(self, workspace_root: str):
        self.workspace = Path(workspace_root).resolve()
        self.ignore_dirs = {'.git', '__pycache__', 'venv', 'env', 'node_modules', '.next', '.999', 'dist', 'build'}
        self._tree_cache = None
        self._cache_time = 0
        self._cache_expiry = 300 # 5 minutes

    def _get_ignored_patterns(self):
        patterns = list(self.ignore_dirs)
        gitignore_path = self.workspace / '.gitignore'
        if gitignore_path.exists():
            try:
                with open(gitignore_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            patterns.append(line.replace('/', '').replace('*', ''))
            except:
                pass
        return set(p for p in patterns if p)

    def map_codebase(self, max_depth: int = 2, refresh: bool = False) -> str:
        """Returns a string representation of the directory tree with depth limits."""
        import time
        now = time.time()
        
        if not refresh and self._tree_cache and (now - self._cache_time < self._cache_expiry):
            return self._tree_cache
            
        self._tree_cache = self.list_dir_tree(".", max_depth)
        self._cache_time = now
        return self._tree_cache

    def list_dir_tree(self, path: str = ".", max_depth: int = 2) -> str:
        target_dir = (self.workspace / path).resolve()
        if self.workspace not in target_dir.parents and target_dir != self.workspace:
            return "Error: Access Denied. Path is outside sandbox."
            
        tree = []
        ignored = self._get_ignored_patterns()
        
        for root, dirs, files in os.walk(target_dir):
            level = str(Path(root).relative_to(target_dir)).count(os.sep)
            
            if level >= max_depth:
                dirs[:] = []
                
            dirs[:] = [d for d in dirs if not any(ign in d for ign in ignored)]
            
            indent = ' ' * 4 * level
            name = Path(root).name if Path(root) != self.workspace else self.workspace.name
            tree.append(f"{indent}{name}/")
            
            if len(tree) > 100:
                tree.append("... (tree truncated for context)")
                break
                
            if level < max_depth:
                sub_indent = ' ' * 4 * (level + 1)
                for f in files:
                    if not any(ign in f for ign in ignored):
                        tree.append(f"{sub_indent}{f}")
                        if len(tree) > 100:
                            tree.append("... (tree truncated for context)")
                            break
            if len(tree) > 100:
                break
                
        return '\n'.join(tree)

    def search_code(self, pattern: str, file_pattern: str = "*.*", case_sensitive: bool = True) -> str:
        """Searches for a regex pattern in files matching the file_pattern."""
        if not pattern:
            return "Error: Search pattern cannot be empty."
            
        results = []
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern '{pattern}' - {str(e)}"

        for file_path in self.workspace.rglob(file_pattern):
            if any(part in self.ignore_dirs for part in file_path.parts):
                continue
            if not file_path.is_file():
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8')
                matches = list(regex.finditer(content))
                if not matches:
                    continue
                    
                file_results = []
                lines = content.split('\n')
                # Limit to first 10 matches per file to avoid bloat
                for match in matches[:10]:
                    start_char = match.start()
                    line_no = content.count('\n', 0, start_char) + 1
                    line_content = lines[line_no-1].strip()
                    file_results.append(f"  Line {line_no}: {line_content}")
                
                if len(matches) > 10:
                    file_results.append(f"  ... and {len(matches) - 10} more matches")
                    
                if file_results:
                    results.append(f"{file_path.relative_to(self.workspace)}:")
                    results.extend(file_results)
            except Exception:
                continue
                
        if not results:
            return f"No matches found for pattern '{pattern}'"
        return '\n'.join(results)

    def view_file_lines(self, file_path: str, start_line: int, end_line: int) -> str:
        """Returns specific lines of a file."""
        target_file = (self.workspace / file_path).resolve()
        
        # Security check
        if self.workspace not in target_file.parents and target_file != self.workspace:
            return "Error: Access Denied. Path is outside sandbox."
            
        if not target_file.exists():
            return f"Error: File {file_path} does not exist."
            
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            
            if start_idx >= len(lines):
                return f"Error: start_line {start_line} is beyond the end of the file (total lines: {len(lines)})."
                
            selected = lines[start_idx:end_idx]
            result = []
            for i, line in enumerate(selected, start=start_idx + 1):
                result.append(f"{i}: {line.rstrip()}")
            return '\n'.join(result)
        except Exception as e:
            return f"Error reading file lines: {str(e)}"

    def extract_symbols(self, file_path: str) -> str:
        """Extracts classes, functions, and methods from a file using AST or Regex."""
        target_file = (self.workspace / file_path).resolve()
        if not target_file.exists():
            return f"Error: File {file_path} not found."

        if target_file.suffix == '.py':
            return self._extract_python_symbols(target_file)
        else:
            return self._extract_generic_symbols(target_file)

    def _extract_python_symbols(self, file_path: Path) -> str:
        import ast
        try:
            tree = ast.parse(file_path.read_text(encoding='utf-8'))
            symbols = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    symbols.append(f"CLASS: {node.name} (Line {node.lineno})")
                elif isinstance(node, ast.FunctionDef):
                    symbols.append(f"FUNC: {node.name} (Line {node.lineno})")
            return "\n".join(symbols) if symbols else "No symbols found."
        except Exception as e:
            return f"Error parsing Python AST: {str(e)}"

    def _extract_generic_symbols(self, file_path: Path) -> str:
        # Fallback for JS/TS/Go/etc.
        patterns = [
            r'(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z0-9_]+)', # JS/TS Functions
            r'(?:export\s+)?class\s+([a-zA-Z0-9_]+)',                 # JS/TS Classes
            r'func\s+([a-zA-Z0-9_]+)',                                # Go Functions
            r'fn\s+([a-zA-Z0-9_]+)',                                  # Rust Functions
        ]
        symbols = []
        try:
            content = file_path.read_text(encoding='utf-8')
            for pattern in patterns:
                matches = re.finditer(pattern, content)
                for m in matches:
                    symbols.append(f"SYMBOL: {m.group(1)}")
            return "\n".join(set(symbols)) if symbols else "No symbols found."
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def get_codebase_summary(self) -> str:
        """Synthesizes a high-level summary of the entire codebase."""
        summary = []
        # 1. README
        readme_path = self.workspace / "README.md"
        if readme_path.exists():
            content = readme_path.read_text(encoding='utf-8')
            summary.append(f"--- README OVERVIEW ---\n{content[:500]}...")
        
        # 2. Structure Overview (Depth 2)
        summary.append(f"\n--- DIRECTORY STRUCTURE ---\n{self.map_codebase(max_depth=2)}")
        
        # 3. Key Files Analysis
        key_files = ["main.py", "app.py", "index.js", "package.json", "requirements.txt"]
        key_symbols = []
        for kf in key_files:
            if (self.workspace / kf).exists():
                syms = self.extract_symbols(kf)
                key_symbols.append(f"Symbols in {kf}:\n{syms}")
        
        if key_symbols:
            summary.append("\n--- CORE COMPONENT SYMBOLS ---\n" + "\n".join(key_symbols))
            
        return "\n\n".join(summary)
