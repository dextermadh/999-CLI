import os
import re
import json
import time
from typing import Literal, List
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from core.state import AgentState
from core.prompts import ARCHITECT_SYSTEM_PROMPT
from ethics.risk_classifier import RiskClassifier
from tools.file_manager import LocalFileManager
from tools.terminal import LocalTerminal
from tools.code_analyzer import CodeAnalyzer
from tools.web_tools import WebTools
from tools.rag_engine import RAGEngine
from tools.git_tools import GitTools

# --- Setup ---
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
classifier = RiskClassifier()

workspace_path = os.getcwd()
file_manager = LocalFileManager(project_root=workspace_path)
terminal = LocalTerminal(workspace_root=workspace_path)
analyzer = CodeAnalyzer(workspace_root=workspace_path)
web_tools = WebTools()
rag_engine = RAGEngine(workspace_root=workspace_path)
git_tools = GitTools(workspace_root=workspace_path)


# ============================================================
#  UTILITY: Extract tool calls from any format the model uses
# ============================================================

def _extract_tool_calls(text: str) -> list:
    """
    Robust tool extractor using balanced-brace parsing. 
    Can handle complex JSON with nested braces (like React code).
    """
    if not text or not isinstance(text, str):
        return []

    parsed = []
    seen_ids = set()

    def _add_safe(data):
        if not isinstance(data, dict): return
        t_name = data.get("tool") or data.get("tool_name")
        if not t_name and "command" in data: t_name = "run_terminal"
        
        if t_name:
            data["tool"] = t_name 
            if "file" in data and "path" not in data: data["path"] = data["file"]
            if "text" in data and "content" not in data: data["content"] = data["text"]
            
            call_id = json.dumps(data, sort_keys=True)
            if call_id not in seen_ids:
                parsed.append(data)
                seen_ids.add(call_id)

    # Balanced brace parsing to find potential JSON blocks
    start_idx = -1
    depth = 0
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start_idx != -1:
                candidate = text[start_idx:i+1]
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict) and ("tool" in data or "command" in data):
                        _add_safe(data)
                except:
                    pass
                start_idx = -1

    return parsed


def _has_tool_calls(text: str) -> bool:
    """Quick check: does the text contain any tool call indicator?"""
    if not text: return False
    indicators = ["<tool_call>", "<|tool_call", "```json", '"tool":']
    return any(ind in text for ind in indicators)


def _get_display_text(text: str) -> str:
    """Surgical removal of all technical noise, leaving only the model's thoughts."""
    if not text: return ""
    
    clean = text
    # Remove all known tags and their closing counterparts
    noise = [
        "<|thought|>", "</|thought|>", "<|thought>", "</|thought>",
        "<|channel>thought", "<channel|>",
        "<tool_call>", "</tool_call>",
        "<|tool_call|>", "</|tool_call|>",
        "call:", "FINISHED"
    ]
    for n in noise:
        clean = clean.replace(n, "")
    
    # Remove any residual JSON blocks
    clean = re.sub(r'\{[^{}]*"tool"[^{}]*\}', '', clean, flags=re.DOTALL)
    # Remove any hanging closing tags from malformed output
    clean = re.sub(r'</?[a-z_]+_call/?>', '', clean)
    
    return clean.strip()


# ============================================================
#  GRAPH NODES
# ============================================================

def analyze_node(state: AgentState):
    """Gathers context before planning. Loads project config if available."""
    map_str = analyzer.map_codebase(max_depth=4)

    # Load project config (.999/config.md) - truncate to save context
    config_path = os.path.join(workspace_path, ".999", "config.md")
    config_content = ""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read(1000)
                if len(config_content) == 1000:
                    config_content += "\n... (config truncated)"
        except Exception:
            pass

    analysis_parts = [
        f"CURRENT DIRECTORY: {state.get('current_dir', os.getcwd())}",
        f"CODEBASE MAP (Depth 4):\n{map_str}"
    ]
    if config_content:
        analysis_parts.append(f"PROJECT CONVENTIONS:\n{config_content}")

    project_type = _detect_project_type()
    if project_type:
        analysis_parts.append(f"PROJECT TYPE: {project_type}")

    return {"analysis_result": "\n\n".join(analysis_parts)}


def _detect_project_type() -> str:
    """Detects the project type from manifest files."""
    indicators = {
        "requirements.txt": "Python", "pyproject.toml": "Python",
        "setup.py": "Python", "package.json": "Node.js",
        "go.mod": "Go", "Cargo.toml": "Rust",
        "pom.xml": "Java (Maven)", "build.gradle": "Java (Gradle)",
    }
    detected = []
    for filename, lang in indicators.items():
        if os.path.exists(os.path.join(workspace_path, filename)):
            detected.append(lang)
    return ", ".join(set(detected)) if detected else ""


def plan_node(state: AgentState):
    """
    Calls the LLM to generate a plan / response.
    Streams tokens live to stdout for real-time feedback.
    Falls back to non-streaming if streaming produces empty output.
    """
    import sys

    # Generate explicit tool descriptions with schemas for the model
    descriptions = {
        "read_file": '{"tool": "read_file", "path": "string"}',
        "write_file": '{"tool": "write_file", "path": "string", "content": "string"}',
        "patch_file": '{"tool": "patch_file", "path": "string", "search_string": "string", "replace_string": "string"}',
        "list_files": '{"tool": "list_files", "path": "string"}',
        "list_dir_tree": '{"tool": "list_dir_tree", "path": "string", "max_depth": int}',
        "delete_file": '{"tool": "delete_file", "path": "string"}',
        "create_dir": '{"tool": "create_dir", "path": "string"}',
        "move_file": '{"tool": "move_file", "source_path": "string", "dest_path": "string"}',
        "run_terminal": '{"tool": "run_terminal", "command": "string"}',
        "search_code": '{"tool": "search_code", "pattern": "string", "file_pattern": "string"}',
        "browse_url": '{"tool": "browse_url", "url": "string"}',
        "index_workspace": '{"tool": "index_workspace"}',
        "semantic_search": '{"tool": "semantic_search", "query": "string"}'
    }
    
    allowed = state.get('allowed_tools', [])
    tool_help = "\n".join([f"- {t}: {descriptions.get(t, 'JSON tool call')}" for t in allowed])

    sys_prompt = ARCHITECT_SYSTEM_PROMPT.format(
        tool_descriptions=tool_help,
        current_dir=state.get('current_dir', workspace_path),
        allowed_tools=", ".join(allowed),
        analysis_result=state.get('analysis_result', ''),
        verification_result=state.get('verification_result', '')
    )

    history = state['messages']
    # Keep a much larger context for complex engineering tasks
    if len(history) > 50:
        history = history[-50:]

    def _role(m):
        if m.type == "human": return "user"
        if m.type == "ai": return "assistant"
        return "user" # Treat tools as user feedback for maximum compatibility

    # Clean the history to remove leaked tags or redundant technical noise
    formatted_history = []
    for m in history:
        content = m.content
        
        # Truncate large results (like directory lists or file reads)
        if len(content) > 1500:
            content = content[:1500] + "... (truncated)"

        if m.type == "ai":
            # Strip technical tags to keep the reasoning thread clean
            content = re.sub(r'<(tool_call|\|tool_call\|).*?</(tool_call|\|tool_call\|)>', '[Action Taken]', content, flags=re.DOTALL)
        
        role = _role(m)
        formatted_history.append({"role": role, "content": content})

    messages = [
        {"role": "system", "content": sys_prompt},
        *formatted_history
    ]

    start_time = time.time()
    thought_and_action = ""
    did_stream = False
    
    # Use Rich for clean formatting
    from rich.console import Console
    from rich.text import Text
    stream_console = Console()
    
    # ---- Step 1: Attempt Streaming ----
    try:
        print("\n[bold green]999-CLI[/bold green] [dim]is thinking...[/dim]\n")
        
        stream = client.chat.completions.create(
            model=state.get("model_name", "gemma-4-e4b"),
            messages=messages,
            temperature=0.1,
            max_tokens=4096,
            stream=True
        )

        silence_active = False
        silence_patterns = ["<tool_call", "<|tool_call", "call:tool_call", "```json", '{"tool"']
        
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except:
                continue
                
            thought_and_action += delta
            
            # AGGRESSIVE SHIELD: If a tool pattern is detected in the tail, kill the display
            if not silence_active:
                if any(p in thought_and_action[-50:] for p in silence_patterns):
                    silence_active = True
                else:
                    # Scrub partial tags from the delta
                    clean_delta = delta
                    for p in silence_patterns:
                        clean_delta = clean_delta.replace(p[:5], "")
                    
                    if clean_delta:
                        stream_console.print(Text(clean_delta, style="italic dim"), end="")
                        did_stream = True
            else:
                # Shield is active: check if we've reached a closing indicator
                if any(p in thought_and_action[-20:] for p in ["</tool_call>", "</|tool_call|>", "}"]):
                    # Wait for a potential newline after a closing brace
                    if thought_and_action.endswith("\n") or "FINISHED" in thought_and_action[-10:]:
                        silence_active = False
        
        print("\n")
    except Exception as e:
        thought_and_action = f"(Streaming error: {str(e)})"

    # ---- Step 2: Fallback & Recovery ----
    if not thought_and_action.strip() or len(thought_and_action.strip()) < 5:
        try:
            # If the model is silent, try a HIGH-TEMP nudge
            messages.append({"role": "user", "content": "Please continue. Use a tool call if necessary."})
            response = client.chat.completions.create(
                model=state.get("model_name", "gemma-4-e4b"),
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            thought_and_action = response.choices[0].message.content or "(Model remains silent. Check server logs.)"
        except Exception as e:
            thought_and_action = f"(Fatal Error: {str(e)})"

    # ---- Step 3: Final Logic & Tool Fallback ----
    state['internal_monologue'] = thought_and_action
    
    clean_text = _get_display_text(thought_and_action).strip()
    if not clean_text:
        if _extract_tool_calls(thought_and_action):
            thought_and_action = "Executing tools...\n" + thought_and_action
        elif not thought_and_action.strip():
            thought_and_action = "(Model returned no text content.)"
    
    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "internal_monologue": thought_and_action,
        "did_stream": did_stream,
        "messages": [AIMessage(content=thought_and_action)],
        "token_usage": {
            "prompt": 0,
            "completion": 0,
            "time_ms": elapsed_ms
        }
    }

    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "internal_monologue": thought_and_action,
        "did_stream": did_stream,
        "messages": [AIMessage(content=thought_and_action)],
        "token_usage": {
            "prompt": 0,
            "completion": 0,
            "time_ms": elapsed_ms
        }
    }


def safety_gate_node(state: AgentState):
    """
    Lightweight safety check.
    Skips the expensive LLM call for read-only operations.
    Only calls the classifier when write/delete/terminal operations are detected.
    """
    monologue = state['internal_monologue']
    tool_calls = _extract_tool_calls(monologue)

    # Check if any tool is potentially dangerous
    dangerous_tools = {"write_file", "patch_file", "delete_file", "move_file", "run_terminal"}
    has_dangerous = any(t.get("tool") in dangerous_tools for t in tool_calls)

    if not has_dangerous:
        # Auto-allow read-only operations
        return {
            "risk_assessment": {
                "risk_score": 0.0,
                "category": "Safe",
                "reasoning": "Read-only or conversational operation.",
                "action": "ALLOW"
            },
            "requires_approval": False
        }

    # Only call the LLM classifier for potentially dangerous operations
    try:
        user_task = state['messages'][-1].content if state['messages'] else ""
        model = state.get("model_name")
        assessment = classifier.analyze_instruction(user_task, monologue, model=model)
        return {
            "risk_assessment": assessment.model_dump(),
            "requires_approval": assessment.risk_score > 0.7 or assessment.action == "BLOCK"
        }
    except Exception:
        # If classifier fails, default to ALLOW with monitoring
        return {
            "risk_assessment": {
                "risk_score": 0.3,
                "category": "Classifier Error",
                "reasoning": "Safety classifier failed. Defaulting to ALLOW.",
                "action": "ALLOW"
            },
            "requires_approval": False
        }


def execute_node(state: AgentState):
    """
    Executes ALL tool calls found in the agent's response.
    Handles file operations (with approval), read operations, and terminal commands.
    """
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.syntax import Syntax

    c = Console()

    monologue = state['internal_monologue']
    tool_calls = _extract_tool_calls(monologue)

    if not tool_calls:
        return {"verification_result": ""}

    approval_mode = state.get("approval_mode", "default")
    all_results = []

    # --- Separate write ops (need approval) from other ops ---
    write_ops = []
    other_ops = []

    for tool_data in tool_calls:
        tool_name = tool_data.get("tool")
        if tool_name in ["write_file", "patch_file", "run_terminal"] and approval_mode != "yolo":
            path = tool_data.get("path") or tool_data.get("command", "terminal")
            # For terminal, we don't deduplicate in the same way as files
            if tool_name == "run_terminal":
                write_ops.append((tool_data, f"Command: {tool_data.get('command')}"))
            else:
                # Deduplicate file ops: keep only the last operation per file path
                write_ops = [op for op in write_ops if op[0].get("path") != path]
                diff = ""
                try:
                    if tool_name == "write_file":
                        diff = file_manager.preview_write_file(path, tool_data.get("content", ""))
                    else:
                        diff = file_manager.preview_patch_file(
                            path, tool_data.get("search_string", ""),
                            tool_data.get("replace_string", "")
                        )
                except Exception as e:
                    diff = f"Error: {str(e)}"
                write_ops.append((tool_data, diff))
        else:
            other_ops.append(tool_data)

    # --- Show batched approval (Files and Terminal) ---
    if write_ops:
        c.print(f"\n[bold yellow]Agent requests permission for {len(write_ops)} action(s):[/bold yellow]")

        for tool_data, info in write_ops:
            t_name = tool_data.get("tool")
            if t_name == "run_terminal":
                c.print(f"\n[bold magenta]🖥️ Terminal Command[/bold magenta]")
                c.print(f"  [cyan]{tool_data.get('command')}[/cyan]")
            else:
                c.print(f"\n[bold cyan]📄 {tool_data.get('path')}[/bold cyan]")
                if info and not info.startswith("Error:"):
                    c.print(Syntax(info, "diff", theme="monokai", background_color="default"))
                elif info.startswith("Error:"):
                    c.print(f"[red]{info}[/red]")
                else:
                    c.print("[dim]New file (no diff to show)[/dim]")

        ans = Prompt.ask(
            f"\n[bold yellow]Execute all {len(write_ops)} action(s)? (y/n/feedback)[/bold yellow]",
            console=c
        )

        if ans.lower() in ['y', 'yes']:
            # Only stash if there are file changes
            if any(op[0].get("tool") != "run_terminal" for op in write_ops):
                checkpoint = git_tools.git_stash()
                if "Skipped" not in checkpoint and "Error" not in checkpoint:
                    c.print("[dim]📌 Checkpoint saved.[/dim]")

            for tool_data, info in write_ops:
                if isinstance(info, str) and info.startswith("Error:"):
                    all_results.append(info)
                    continue
                result = _dispatch_tool(tool_data, tool_data.get("tool"), state)
                if result:
                    all_results.append(result)
        elif ans.lower() in ['n', 'no']:
            all_results.append("User rejected the requested actions.")
        else:
            # Conversational Feedback
            feedback = f"User feedback/correction: {ans}"
            c.print(f"\n[bold blue]💬 Feedback captured:[/bold blue] {ans}")
            all_results.append(feedback)

    # --- Execute non-write tools (including terminal commands) ---
    for tool_data in other_ops:
        tool_name = tool_data.get("tool")

        # Show progress indicators
        if tool_name == "read_file":
            c.print(f"[dim]📖 Reading {tool_data.get('path')}...[/dim]")
        elif tool_name == "search_code":
            c.print(f"[dim]🔍 Searching for '{tool_data.get('pattern') or tool_data.get('query')}'...[/dim]")
        elif tool_name == "semantic_search":
            c.print(f"[dim]🧠 Semantic search: '{tool_data.get('query')}'...[/dim]")
        elif tool_name == "run_terminal":
            c.print(f"[dim]🖥️ Running: {tool_data.get('command')}...[/dim]")
        elif tool_name == "list_files" or tool_name == "list_dir_tree":
            c.print(f"[dim]📁 Listing {tool_data.get('path') or '.'}...[/dim]")
        elif tool_name == "index_workspace":
            c.print(f"[dim]📦 Indexing workspace...[/dim]")

        result = _dispatch_tool(tool_data, tool_name, state)
        if result is not None:
            all_results.append(result)

    combined = "\n---\n".join(all_results) if all_results else "All tools executed successfully."
    return {"verification_result": f"Execution Result:\n{combined}"}


def _dispatch_tool(tool_data: dict, tool_name: str, state: AgentState) -> str:
    """Routes a tool call to its handler. Executes ALL tools including terminal."""
    approval_mode = state.get("approval_mode", "default")

    try:
        if tool_name == "write_file":
            return file_manager.write_file(tool_data.get("path"), tool_data.get("content", ""))
        elif tool_name == "patch_file":
            return file_manager.patch_file(
                tool_data.get("path"),
                tool_data.get("search_string", ""),
                tool_data.get("replace_string", "")
            )
        elif tool_name == "read_file":
            return file_manager.read_file(tool_data.get("path"))
        elif tool_name == "list_files":
            return str(file_manager.list_files(tool_data.get("path") or "."))
        elif tool_name == "search_code":
            pattern = tool_data.get("pattern") or tool_data.get("query") or ""
            return analyzer.search_code(
                pattern,
                tool_data.get("file_pattern", "*.*"),
                case_sensitive=tool_data.get("case_sensitive", True)
            )
        elif tool_name == "delete_file":
            return file_manager.delete_file(tool_data.get("path"))
        elif tool_name == "move_file":
            return file_manager.move_file(tool_data.get("source_path"), tool_data.get("dest_path"))
        elif tool_name == "create_dir":
            return file_manager.create_dir(tool_data.get("path"))
        elif tool_name == "get_file_info":
            return file_manager.get_file_info(tool_data.get("path"))
        elif tool_name == "view_file_lines":
            return analyzer.view_file_lines(
                tool_data.get("path"),
                tool_data.get("start_line", 1),
                tool_data.get("end_line", 100)
            )
        elif tool_name == "list_dir_tree":
            return analyzer.list_dir_tree(tool_data.get("path", "."), tool_data.get("max_depth", 2))
        elif tool_name == "index_workspace":
            return rag_engine.index_workspace()
        elif tool_name == "semantic_search":
            return rag_engine.semantic_search(tool_data.get("query", ""), tool_data.get("top_k", 3))
        elif tool_name == "browse_url":
            return web_tools.browse_url(tool_data.get("url"))
        elif tool_name == "fetch_url":
            return web_tools.fetch_url_content(tool_data.get("url"))
        # --- Git Tools ---
        elif tool_name == "git_status":
            return git_tools.git_status()
        elif tool_name == "git_diff":
            return git_tools.git_diff(staged=tool_data.get("staged", False))
        elif tool_name == "git_log":
            return git_tools.git_log(n=tool_data.get("n", 10))
        elif tool_name == "git_commit":
            return git_tools.git_commit(tool_data.get("message", "999: auto-commit"))
        elif tool_name == "git_checkout":
            return git_tools.git_checkout(tool_data.get("target", ""))
        elif tool_name == "git_stash":
            return git_tools.git_stash()
        elif tool_name == "git_stash_pop":
            return git_tools.git_stash_pop()
        # --- Terminal ---
        elif tool_name == "run_terminal":
            command = tool_data.get("command", "")
            # Safety check
            forbidden = ["> /dev/", "rm -rf /", "mkfs", "format C:"]
            if any(f in command for f in forbidden):
                return "Error: Blocked potentially destructive system command."
            return terminal.execute(command)
        else:
            return f"Error: Unknown tool '{tool_name}'"
    except Exception as e:
        return f"Tool '{tool_name}' Failed: {str(e)}"


def verify_node(state: AgentState):
    """
    Post-execution verification.
    Auto-runs tests if files were modified.
    NEW: Auto-checks localhost if a dev server was started.
    """
    result_str = state.get('verification_result', '')
    monologue = state.get('internal_monologue', '')

    # 1. Auto-test discovery
    if "Successfully wrote" in result_str or "Successfully patched" in result_str:
        test_result = _auto_run_tests()
        if test_result:
            result_str += f"\n\n--- Auto Test Results ---\n{test_result}"

    # 2. Auto-Heal: Check localhost if dev server started
    if "npm run dev" in monologue or "next dev" in monologue or "npm start" in monologue:
        from rich.console import Console
        c = Console()
        c.print("[dim]🏥 Auto-Heal: Checking localhost for build errors...[/dim]")
        
        # Give the server a few seconds to boot if it's the first time
        time.sleep(3) 
        try:
            url = "http://localhost:3000"
            browser_report = web_tools.browse_url(url)
            
            if "Error" in browser_report or "Build Error" in browser_report or "SyntaxError" in browser_report:
                result_str += f"\n\n[CRITICAL BUILD ERROR DETECTED]\nSource: {url}\nReport:\n{browser_report}\nFIX THIS IMMEDIATELY."
            else:
                result_str += f"\n\n--- Auto-Heal Check ---\n{url} appears to be healthy."
        except Exception as e:
            result_str += f"\n\nAuto-Heal check failed: {str(e)}"

    # Inject results as SystemMessage
    return {
        "verification_result": result_str,
        "messages": [SystemMessage(content=f"Tool execution results:\n{result_str}")] if result_str else []
    }


def _auto_run_tests() -> str:
    """Detects test framework and runs tests automatically."""
    test_commands = []

    if os.path.exists(os.path.join(workspace_path, "pytest.ini")) or \
       os.path.exists(os.path.join(workspace_path, "pyproject.toml")) or \
       os.path.exists(os.path.join(workspace_path, "setup.cfg")):
        test_commands.append("python -m pytest --tb=short -q")

    if os.path.exists(os.path.join(workspace_path, "package.json")):
        try:
            with open(os.path.join(workspace_path, "package.json"), 'r') as f:
                pkg = json.load(f)
            if "scripts" in pkg and "test" in pkg["scripts"]:
                test_commands.append("npm test")
        except Exception:
            pass

    if os.path.exists(os.path.join(workspace_path, "go.mod")):
        test_commands.append("go test ./...")

    if not test_commands:
        return ""

    results = []
    for cmd in test_commands:
        result = terminal.execute(cmd, timeout=60)
        results.append(f"$ {cmd}\n{result}")

    return "\n".join(results)


# ============================================================
#  ROUTING LOGIC
# ============================================================

def route_safety(state: AgentState) -> Literal["execute", "blocked"]:
    risk = state.get("risk_assessment", {})
    if risk.get("action") == "BLOCK":
        return "blocked"
    return "execute"


def should_continue(state: AgentState) -> Literal["analyze", "end"]:
    monologue = state.get("internal_monologue", "")

    if "FINISHED" in monologue:
        return "end"

    # Continue the loop if the model used ANY tool call format
    if _has_tool_calls(monologue):
        return "analyze"

    return "end"


# ============================================================
#  BUILD THE GRAPH
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node("analyze", analyze_node)
workflow.add_node("plan", plan_node)
workflow.add_node("safety_gate", safety_gate_node)
workflow.add_node("execute", execute_node)
workflow.add_node("verify", verify_node)

workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "plan")
workflow.add_edge("plan", "safety_gate")

workflow.add_conditional_edges(
    "safety_gate",
    route_safety,
    {
        "execute": "execute",
        "blocked": END
    }
)

workflow.add_edge("execute", "verify")

workflow.add_conditional_edges(
    "verify",
    should_continue,
    {
        "analyze": "analyze",
        "end": END
    }
)

app = workflow.compile()