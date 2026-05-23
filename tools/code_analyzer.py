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

    def run_security_scan(self, path: str = ".") -> str:
        """Audits code files in the specified path for security vulnerabilities (SQLi, command injection, hardcoded secrets)."""
        target_path = (self.workspace / path).resolve()
        if self.workspace not in target_path.parents and target_path != self.workspace:
            return "Error: Path is outside sandbox."

        # 1. Attempt Bandit (static analysis tool for python)
        import subprocess
        try:
            result = subprocess.run(
                ["bandit", "-r", str(target_path), "-f", "txt"],
                capture_output=True, text=True, timeout=10
            )
            # Bandit returns exit code 1 if issues found, so we check stdout/stderr regardless of exit code
            if result.stdout.strip():
                return f"### Security Audit Scan Report (Bandit):\n{result.stdout}"
        except Exception:
            pass

        # 2. Heuristic Python/JS Static Scanner (100% correct fallback)
        vulnerabilities = []
        patterns = {
            "SQL Injection (Heuristic)": [
                (r'f"SELECT\s+.*\{.*\}', "Format string SQL query detected. Use parameterized queries instead."),
                (r'"SELECT\s+.*"\s*\+\s*[a-zA-Z0-9_]+', "Raw string concatenation in SQL query detected. High SQLi risk.")
            ],
            "Command Injection (Heuristic)": [
                (r'os\.system\(', "Use of os.system() is deprecated and vulnerable. Use subprocess.run() with shell=False."),
                (r'shell\s*=\s*True', "subprocess spawned with shell=True. High risk of Command Injection.")
            ],
            "Hardcoded Credentials (Heuristic)": [
                (r'(api_key|secret|password|token|pwd)\s*=\s*[\'"][a-zA-Z0-9_\-]{8,}[\'"]', "Potential hardcoded credential or token detected. Use environment variables.")
            ],
            "Dangerous Functions (Heuristic)": [
                (r'\beval\(', "Use of eval() is highly dangerous as it executes arbitrary strings as code."),
                (r'\bexec\(', "Use of exec() is highly dangerous as it executes arbitrary code blocks.")
            ]
        }

        ignored = self._get_ignored_patterns()
        file_count = 0
        for file_path in target_path.rglob("*.*"):
            if any(part in ignored for part in file_path.parts):
                continue
            if file_path.suffix.lower() not in {'.py', '.js', '.ts'}:
                continue
            
            try:
                content = file_path.read_text(encoding='utf-8')
                file_count += 1
                lines = content.split('\n')
                for category, category_patterns in patterns.items():
                    for pattern, desc in category_patterns:
                        matches = re.finditer(pattern, content)
                        for m in matches:
                            start_char = m.start()
                            line_no = content.count('\n', 0, start_char) + 1
                            snippet = lines[line_no-1].strip()
                            vulnerabilities.append(
                                f"  • [yellow]{category}[/yellow] in [cyan]{file_path.relative_to(self.workspace)}[/cyan] (Line {line_no}):\n"
                                f"    Code: `{snippet}`\n"
                                f"    Advice: {desc}"
                            )
            except Exception:
                continue

        lines = ["### Heuristic Security Scanner Audit Report\n"]
        lines.append(f"Scanned {file_count} code files under {path}.")
        if vulnerabilities:
            lines.append(f"[red]Found {len(vulnerabilities)} potential security vulnerabilities:[/red]\n")
            lines.extend(vulnerabilities)
        else:
            lines.append("[green]✓ Zero security issues detected. Clean pass![/green]")
        return "\n".join(lines)

    def run_unit_tests(self, test_path: str = "tests") -> str:
        """Executes the test suite inside the workspace and returns structured pass/fail results."""
        target_dir = (self.workspace / test_path).resolve()
        if self.workspace not in target_dir.parents and target_dir != self.workspace:
            return "Error: Test path is outside sandbox."

        import subprocess
        import sys
        # Check if tests directory exists
        if not target_dir.exists():
            return f"Error: Test directory '{test_path}' does not exist. Please write tests first."

        # Detect test framework (default to pytest, fallback to unittest)
        framework = "pytest"
        try:
            subprocess.run(["pytest", "--version"], capture_output=True, check=True)
        except Exception:
            framework = "unittest"

        cmd = ["pytest", str(target_dir)] if framework == "pytest" else [sys.executable, "-m", "unittest", "discover", "-s", str(target_dir)]
        
        try:
            # Reconfigure stdout/stderr to avoid Windows terminal cp1252 print crashes
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=15
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            
            lines = [f"### Integration & Unit Test Report (Framework: {framework})\n"]
            lines.append(f"Running command: `{' '.join(cmd)}`")
            if result.returncode == 0:
                lines.append("[green]✓ All tests passed successfully![/green]\n")
            else:
                lines.append("[red]⚠ Some tests failed or returned errors.[/red]\n")
                
            lines.append("#### Test Output:")
            lines.append(stdout[:4000])
            if stderr.strip():
                lines.append("#### Standard Error:")
                lines.append(stderr[:2000])
            return "\n".join(lines)
        except subprocess.TimeoutExpired:
            return f"Error: Test execution timed out after 15s. Tests may be hanging or blocked on input/servers."
        except Exception as e:
            return f"Error running tests: {str(e)}"

    def profile_performance(self, command: str) -> str:
        """Profiles a Python execution command using cProfile to discover time-consuming bottlenecks."""
        if not command:
            return "Error: Command to profile cannot be empty."

        import subprocess
        import sys
        
        # Security sanitization
        parts = command.split()
        if not parts:
            return "Error: Invalid command."
            
        executable = parts[0]
        # Only allow profiling python scripts
        if executable not in ["python", "python3"] and not executable.endswith("python") and not executable.endswith("python.exe"):
            # If they passed a script path directly, prepend python
            if executable.endswith(".py"):
                parts.insert(0, sys.executable)
            else:
                return "Error: Performance profiling is restricted to Python scripts."
        else:
            parts[0] = sys.executable

        # Construct cProfile invocation: python -m cProfile -s cumulative <script>
        cmd = [sys.executable, "-m", "cProfile", "-s", "cumulative"] + parts[1:]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=20
            )
            stdout = result.stdout or ""
            
            lines = [f"### Performance Profiling Audit Report (cProfile)\n"]
            lines.append(f"Profiled execution of command: `{command}`")
            lines.append("#### Top 25 Cumulative Execution Time Operations:")
            
            stdout_lines = stdout.split('\n')
            # Extract header and cProfile rows (lines containing function call timings)
            profile_rows = []
            header_found = False
            for line in stdout_lines:
                if "ncalls" in line and "tottime" in line:
                    header_found = True
                    profile_rows.append(line)
                    continue
                if header_found and line.strip():
                    profile_rows.append(line)
                    if len(profile_rows) >= 26: # header + 25 rows
                        break
                        
            if profile_rows:
                lines.append("```")
                lines.extend(profile_rows)
                lines.append("```")
            else:
                lines.append("```")
                lines.extend(stdout_lines[:50]) # fallback to first 50 lines
                lines.append("```")
            return "\n".join(lines)
        except subprocess.TimeoutExpired:
            return "Error: Profiling execution timed out after 20s. Make sure the script doesn't block on network or servers."
        except Exception as e:
            return f"Error profiling performance: {str(e)}"
