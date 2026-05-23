import os
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional
from tools.knowledge_graph import CodeKnowledgeGraph

class KnowledgeGraphBuilder:
    def __init__(self, workspace_root: str):
        self.workspace = Path(workspace_root).resolve()
        self.ignore_dirs = {'.git', '__pycache__', 'venv', 'env', 'node_modules', '.next', '.999', 'dist', 'build'}
        self.graph = CodeKnowledgeGraph()
        # Track defined classes and functions globally to resolve inheritances and calls
        # mapping: class_name -> node_id
        self.global_classes: Dict[str, str] = {}
        # mapping: function_name -> List[node_id] (since same function name can exist in different files/classes)
        self.global_functions: Dict[str, List[str]] = {}
        
        # Save path
        self.save_path = self.workspace / ".999" / "knowledge_graph.json"

    def _get_ignored_patterns(self) -> Set[str]:
        patterns = list(self.ignore_dirs)
        gitignore_path = self.workspace / '.gitignore'
        if gitignore_path.exists():
            try:
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Strip leading/trailing slashes and wildcards
                            clean = line.replace('/', '').replace('*', '')
                            if clean:
                                patterns.append(clean)
            except:
                pass
        return set(patterns)

    def resolve_local_file_path(self, current_file: Path, import_str: str) -> Optional[Path]:
        """Resolves an import string (e.g. 'core.graph' or 'tools.terminal') to a local file path in the workspace."""
        # Standardize relative paths and absolute-like imports
        parts = import_str.split('.')
        
        # 1. Try relative to workspace root (absolute-like local imports)
        candidate1 = self.workspace / "/".join(parts)
        if (candidate1.with_suffix('.py')).exists():
            return candidate1.with_suffix('.py')
        if (candidate1 / "__init__.py").exists():
            return candidate1 / "__init__.py"

        # 2. Try relative to the current file's directory
        candidate2 = current_file.parent / "/".join(parts)
        if (candidate2.with_suffix('.py')).exists():
            return candidate2.with_suffix('.py')
        if (candidate2 / "__init__.py").exists():
            return candidate2 / "__init__.py"

        return None

    def parse_python_file(self, file_path: Path) -> None:
        """Parses a Python file using AST to extract files, classes, methods, imports, and calls."""
        rel_path = str(file_path.relative_to(self.workspace)).replace('\\', '/')
        file_node_id = f"file:{rel_path}"
        
        # Add file node
        self.graph.add_node(
            node_id=file_node_id,
            node_type="file",
            properties={
                "path": rel_path,
                "name": file_path.name,
                "extension": ".py",
                "size_bytes": file_path.stat().st_size
            }
        )

        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
        except Exception as e:
            # SyntaxError or encoding issues, skip deep AST but keep the file node
            return

        # Track local imports to resolve later
        local_imports: List[Tuple[str, int]] = []
        
        # Inner scope tracker variables
        current_class: Optional[str] = None
        current_class_node_id: Optional[str] = None
        current_function_node_id: Optional[str] = None

        # Custom recursive walker to keep track of lexical scope (classes and functions)
        def walk_ast(node: ast.AST, current_class_info: Optional[Tuple[str, str]] = None, current_func_id: Optional[str] = None):
            nonlocal file_node_id
            
            c_class_name, c_class_id = current_class_info if current_class_info else (None, None)
            c_func_id = current_func_id

            if isinstance(node, ast.Import):
                for alias in node.names:
                    local_imports.append((alias.name, node.lineno))
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    local_imports.append((node.module, node.lineno))

            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                class_node_id = f"class:{rel_path}:{class_name}"
                
                # Register globally
                self.global_classes[class_name] = class_node_id
                
                # Fetch docstring
                docstring = ast.get_docstring(node) or ""
                
                # Add class node
                self.graph.add_node(
                    node_id=class_node_id,
                    node_type="class",
                    properties={
                        "name": class_name,
                        "docstring": docstring,
                        "start_line": node.lineno,
                        "end_line": getattr(node, 'end_lineno', node.lineno)
                    }
                )
                
                # Establish DEFINES relation from File to Class
                self.graph.add_edge(file_node_id, class_node_id, "DEFINES")

                # Parse base classes (inheritance)
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_name = base.id
                        # We will link inheritance edges after collecting all global classes in a second pass
                        self.graph.add_node(class_node_id, "class") # ensure node exists in edge mapping
                
                # Update current class scope and traverse children
                c_class_name, c_class_id = class_name, class_node_id
                for child in node.body:
                    walk_ast(child, (c_class_name, c_class_id), c_func_id)
                return # skip default walk to avoid re-traversing body

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                is_async = isinstance(node, ast.AsyncFunctionDef)
                docstring = ast.get_docstring(node) or ""
                
                # Build unique node ID
                if c_class_id:
                    func_node_id = f"method:{rel_path}:{c_class_name}:{func_name}"
                    relation = "DEFINES_METHOD"
                    source_node = c_class_id
                else:
                    func_node_id = f"func:{rel_path}:{func_name}"
                    relation = "DEFINES"
                    source_node = file_node_id

                # Register globally
                if func_name not in self.global_functions:
                    self.global_functions[func_name] = []
                self.global_functions[func_name].append(func_node_id)

                # Get function signature arguments
                args = [arg.arg for arg in node.args.args]
                
                # Add function node
                self.graph.add_node(
                    node_id=func_node_id,
                    node_type="function",
                    properties={
                        "name": func_name,
                        "docstring": docstring,
                        "is_async": is_async,
                        "args": args,
                        "start_line": node.lineno,
                        "end_line": getattr(node, 'end_lineno', node.lineno)
                    }
                )
                
                # Establish edge from file or class defining the function
                self.graph.add_edge(source_node, func_node_id, relation)

                # Update current function scope and traverse children
                c_func_id = func_node_id
                for child in node.body:
                    walk_ast(child, (c_class_name, c_class_id), c_func_id)
                return # skip default walk to avoid re-traversing body

            elif isinstance(node, ast.Call):
                # We found a function/method call!
                target_func = None
                
                if isinstance(node.func, ast.Name):
                    target_func = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    target_func = node.func.attr
                
                if target_func and c_func_id:
                    # We will resolve and map these calls in a second pass once all nodes are collected!
                    # For now, store the raw call information in the source function's properties to process later
                    if "raw_calls" not in self.graph.nodes[c_func_id]["properties"]:
                        self.graph.nodes[c_func_id]["properties"]["raw_calls"] = []
                    self.graph.nodes[c_func_id]["properties"]["raw_calls"].append(target_func)

            # Recurse through all child elements
            for child in ast.iter_child_nodes(node):
                walk_ast(child, (c_class_name, c_class_id), c_func_id)

        # Start walk
        walk_ast(tree)

        # Resolve imports immediately for this file
        for imp_name, line in local_imports:
            target_path = self.resolve_local_file_path(file_path, imp_name)
            if target_path:
                target_rel = str(target_path.relative_to(self.workspace)).replace('\\', '/')
                target_node_id = f"file:{target_rel}"
                
                # Add target node placeholder to guarantee it exists in graph
                self.graph.add_node(target_node_id, "file", {"path": target_rel, "name": target_path.name})
                # Add IMPORTS edge
                self.graph.add_edge(file_node_id, target_node_id, "IMPORTS")
            else:
                # External dependency
                dep_node_id = f"dependency:{imp_name}"
                self.graph.add_node(dep_node_id, "dependency", {"name": imp_name})
                self.graph.add_edge(file_node_id, dep_node_id, "DEPENDS_ON")

    def parse_generic_file(self, file_path: Path) -> None:
        """Heuristic regex-based parsing for non-Python files to extract structural elements."""
        rel_path = str(file_path.relative_to(self.workspace)).replace('\\', '/')
        file_node_id = f"file:{rel_path}"
        ext = file_path.suffix.lower()

        self.graph.add_node(
            node_id=file_node_id,
            node_type="file",
            properties={
                "path": rel_path,
                "name": file_path.name,
                "extension": ext,
                "size_bytes": file_path.stat().st_size
            }
        )

        # Simple patterns for JS/TS, Go, Rust functions and classes
        patterns = [
            (r'(?:export\s+)?class\s+([a-zA-Z0-9_]+)', 'class'),
            (r'(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z0-9_]+)', 'function'),
            (r'func\s+([a-zA-Z0-9_]+)', 'function'), # Go
            (r'fn\s+([a-zA-Z0-9_]+)', 'function'),   # Rust
        ]

        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            for pattern, node_type in patterns:
                matches = re.finditer(pattern, content)
                for m in matches:
                    name = m.group(1)
                    start_char = m.start()
                    line_no = content.count('\n', 0, start_char) + 1
                    
                    if node_type == 'class':
                        node_id = f"class:{rel_path}:{name}"
                        self.global_classes[name] = node_id
                        self.graph.add_node(node_id, "class", {"name": name, "start_line": line_no})
                        self.graph.add_edge(file_node_id, node_id, "DEFINES")
                    else:
                        node_id = f"func:{rel_path}:{name}"
                        if name not in self.global_functions:
                            self.global_functions[name] = []
                        self.global_functions[name].append(node_id)
                        self.graph.add_node(node_id, "function", {"name": name, "start_line": line_no})
                        self.graph.add_edge(file_node_id, node_id, "DEFINES")
                        
            # Extract basic imports (e.g. require or ES6 imports in JS/TS)
            js_imports = re.finditer(r'(?:import|from)\s+[\'"]([^\'"]+)[\'"]', content)
            for m in js_imports:
                imp = m.group(1)
                if imp.startswith('.'):
                    # Local relative import
                    # Strip relative prefixes to find candidate filenames
                    clean_imp = imp.lstrip('./').replace('/', '.')
                    # For simplicity, add dependency or local link if file exists
                    target_node_id = f"dependency:{imp}"
                    self.graph.add_node(target_node_id, "dependency", {"name": imp})
                    self.graph.add_edge(file_node_id, target_node_id, "DEPENDS_ON")
                else:
                    # NPM dependency
                    dep_node_id = f"dependency:{imp}"
                    self.graph.add_node(dep_node_id, "dependency", {"name": imp})
                    self.graph.add_edge(file_node_id, dep_node_id, "DEPENDS_ON")
        except Exception:
            pass

    def build_graph(self) -> CodeKnowledgeGraph:
        """Scans the directory, parses all code files, resolves cross-references, and returns the finished graph."""
        ignored = self._get_ignored_patterns()
        
        code_exts = {'.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs'}
        
        # Pass 1: Add all nodes (files, classes, functions) and initial definitions/local imports
        all_code_files: List[Path] = []
        for file_path in self.workspace.rglob("*.*"):
            if any(part in ignored for part in file_path.parts):
                continue
            if not file_path.is_file():
                continue
            
            ext = file_path.suffix.lower()
            if ext not in code_exts:
                continue

            all_code_files.append(file_path)
            
            if ext == '.py':
                self.parse_python_file(file_path)
            else:
                self.parse_generic_file(file_path)

        # Pass 2: Resolve cross-references (inheritances and function calls)
        for node_id, node_data in list(self.graph.nodes.items()):
            node_type = node_data["type"]
            properties = node_data.get("properties", {})
            
            # 1. Resolve Inheritances
            if node_type == "class":
                # Check for inherited classes defined in AST bases
                # (we re-parse or extract from bases)
                # Since we want to map relationships, we can check AST code
                # For simplicity, we can do it by checking base name matches in global_classes
                rel_path = properties.get("path")
                start_line = properties.get("start_line")
                if rel_path and start_line:
                    # Let's see if this class inherits from a known class
                    # (Quick check: search file contents around definition)
                    pass

            # 2. Resolve Function/Method Calls (Call Graph Mapping)
            elif node_type == "function" and "raw_calls" in properties:
                raw_calls = properties["raw_calls"]
                for called_name in raw_calls:
                    # Heuristic 1: Call matches a globally defined function
                    if called_name in self.global_functions:
                        for target_id in self.global_functions[called_name]:
                            # Heuristic optimization: If in Python, prefer linking call to target function defined in the same file or imported files
                            src_file = node_id.split(':')[1] if len(node_id.split(':')) > 1 else ""
                            tgt_file = target_id.split(':')[1] if len(target_id.split(':')) > 1 else ""
                            
                            # Add CALLS relation
                            self.graph.add_edge(node_id, target_id, "CALLS")
                            
                    # Heuristic 2: Call matches a class instantiation
                    elif called_name in self.global_classes:
                        class_node_id = self.global_classes[called_name]
                        self.graph.add_edge(node_id, class_node_id, "INSTANTIATES")
                
                # Cleanup raw_calls property to keep JSON clean
                del properties["raw_calls"]

        # Ensure all links are valid and target nodes exist in the graph
        for node_id, node_data in list(self.graph.nodes.items()):
            for edge_type, targets in list(node_data["edges"].items()):
                valid_targets = []
                for target_id in targets:
                    if target_id in self.graph.nodes:
                        valid_targets.append(target_id)
                node_data["edges"][edge_type] = valid_targets

        # Save to disk
        self.graph.save_to_disk(str(self.save_path))
        return self.graph
