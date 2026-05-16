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
from tools.knowledge_base import KnowledgeBase
from tools.artifact_manager import ArtifactManager
from tools.subagent import SubagentManager
from agents.architect import execute_plan
from agents.reviewer import execute_verify
from agents.developer import execute_develop

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
knowledge_base = KnowledgeBase(workspace_root=workspace_path)
artifact_manager = ArtifactManager(workspace_root=workspace_path)
subagent_manager = SubagentManager(client=client, model_name="gemma-4-e4b")
from tools.episode_store import EpisodeStore
episode_store = EpisodeStore(workspace_root=workspace_path)
from tools.browser_tool import BrowserTool
browser_tool = BrowserTool(workspace_root=workspace_path)


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
            if "file_path" in data and "path" not in data: data["path"] = data["file_path"]
            if "filepath" in data and "path" not in data: data["path"] = data["filepath"]
            if "arg" in data and "path" not in data: data["path"] = data["arg"]
            if "text" in data and "content" not in data: data["content"] = data["text"]
            if "args" in data and "command" not in data: data["command"] = data["args"]
            
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
                    if isinstance(data, dict):
                        if "tool" in data or "command" in data:
                            _add_safe(data)
                        else:
                            preceding = text[max(0, start_idx-50):start_idx].strip()
                            match = re.search(r'(?:<\|?tool_call>|call:)\s*([a-zA-Z0-9_]+)$', preceding)
                            if match:
                                data["tool"] = match.group(1)
                                _add_safe(data)
                except:
                    # Fallback: Try to fix missing quotes or double braces (Phase 3 hardening)
                    try:
                        cand_to_fix = candidate
                        if cand_to_fix.startswith("{{") and cand_to_fix.endswith("}}"):
                            cand_to_fix = cand_to_fix[1:-1]
                            
                        fixed = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)\s*:', r'\1"\2":', cand_to_fix)
                        data = json.loads(fixed)
                        if isinstance(data, dict):
                            if "tool" in data or "command" in data:
                                _add_safe(data)
                            else:
                                preceding = text[max(0, start_idx-50):start_idx].strip()
                                match = re.search(r'(?:<\|?tool_call>|call:)\s*([a-zA-Z0-9_]+)$', preceding)
                                if match:
                                    data["tool"] = match.group(1)
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
    # Only refresh the map if we just performed a write/delete operation
    last_verification = state.get("verification_result", "")
    should_refresh = any(phrase in last_verification for phrase in ["Successfully wrote", "Successfully patched", "Successfully deleted", "Successfully moved"])
    
    speed = state.get("speed_mode", "fast")
    depth = 2 if speed == "fast" else 4
    map_str = analyzer.map_codebase(max_depth=depth, refresh=should_refresh)

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
        f"WORKSPACE: {workspace_path}",
        f"CODEBASE MAP (Depth {depth}):\n{map_str}"
    ]
    if config_content:
        analysis_parts.append(f"PROJECT CONVENTIONS:\n{config_content}")

    project_type = _detect_project_type()
    if project_type:
        analysis_parts.append(f"PROJECT TYPE: {project_type}")

    # Load available knowledge topics (Auto-suggest)
    knowledge_list = knowledge_base.list_knowledge()
    if "No knowledge stored yet" not in knowledge_list:
        analysis_parts.append(knowledge_list)

    # Memory Retrieval (Phase 2)
    user_query = ""
    for m in reversed(state.get('messages', [])):
        if m.type == "human":
            user_query = m.content
            break
    if user_query:
        similar_episodes = episode_store.search_similar(user_query, k=2)
        if similar_episodes:
            memory_parts = ["PAST SIMILAR EPISODES (Memory):"]
            for ep in similar_episodes:
                memory_parts.append(f"Request: {ep['request']}\nSolution:\n{ep['solution']}")
            analysis_parts.append("\n".join(memory_parts))

    # Dynamic Model Routing (Phase 4)
    complex_keywords = ["create", "implement", "refactor", "debug", "fix", "write", "patch"]
    is_complex = any(kw in user_query.lower() for kw in complex_keywords) if user_query else False
    
    # Dynamic Model Selection from LM Studio
    try:
        models_response = client.models.list()
        loaded_models = [m.id for m in models_response.data]
        
        preferred_model = "unsloth/gemma-4-e4b-it"
        
        if preferred_model in loaded_models:
            chosen_model = preferred_model
        elif loaded_models:
            # Use the first available loaded model if preferred is not found
            chosen_model = loaded_models[0]
            from rich.console import Console
            Console().print(f"[yellow]Preferred model not found. Using loaded model: {chosen_model}[/yellow]")
        else:
            chosen_model = preferred_model
    except Exception:
        # Fallback if LM Studio API fails
        chosen_model = "unsloth/gemma-4-e4b-it"
    from rich.console import Console
    c = Console()
    c.print(f"[bold magenta]Router:[/bold magenta] Selected model [cyan]{chosen_model}[/cyan] (Complexity: {'High' if is_complex else 'Low'})")

    return {
        "analysis_result": "\n\n".join(analysis_parts),
        "model_name": chosen_model
    }


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
    Delegates to agents/architect.py.
    """
    return execute_plan(state, client, workspace_path)

def develop_node(state: AgentState):
    """
    Calls the Developer LLM to execute coding tasks.
    Delegates to agents/developer.py.
    """
    return execute_develop(state, client, workspace_path)

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
        return {"verification_result": "No tools were needed for this step. Ready to answer."}

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
            c.print(f"[dim]Reading {tool_data.get('path')}...[/dim]")
        elif tool_name == "search_code":
            c.print(f"[dim]Searching for '{tool_data.get('pattern') or tool_data.get('query')}'...[/dim]")
        elif tool_name == "semantic_search":
            c.print(f"[dim]Semantic search: '{tool_data.get('query')}'...[/dim]")
        elif tool_name == "run_terminal":
            c.print(f"[dim]Running: {tool_data.get('command')}...[/dim]")
        elif tool_name == "list_files" or tool_name == "list_dir_tree":
            c.print(f"[dim]Listing {tool_data.get('path') or '.'}...[/dim]")
        elif tool_name == "index_workspace":
            c.print(f"[dim]Indexing workspace...[/dim]")
        elif tool_name == "get_codebase_summary":
            c.print(f"[dim]Synthesizing project summary...[/dim]")
        elif tool_name == "extract_symbols":
            c.print(f"[dim]Extracting symbols from {tool_data.get('path')}...[/dim]")
        elif tool_name == "dependency_graph":
            c.print(f"[dim]Building dependency graph...[/dim]")
        elif tool_name == "incremental_index":
            c.print(f"[dim]Incremental re-indexing...[/dim]")
        elif tool_name == "save_knowledge":
            c.print(f"[dim]Saving knowledge about '{tool_data.get('topic')}'...[/dim]")
        elif tool_name == "read_knowledge":
            c.print(f"[dim]Reading knowledge about '{tool_data.get('topic')}'...[/dim]")
        elif tool_name == "list_knowledge":
            c.print(f"[dim]Listing stored knowledge...[/dim]")
        elif tool_name == "create_artifact":
            c.print(f"[dim]Creating artifact '{tool_data.get('title')}'...[/dim]")
        elif tool_name == "update_artifact":
            c.print(f"[dim]Updating artifact '{tool_data.get('artifact_id')}'...[/dim]")
        elif tool_name == "read_artifact":
            c.print(f"[dim]Reading artifact '{tool_data.get('artifact_id')}'...[/dim]")
        elif tool_name == "list_artifacts":
            c.print(f"[dim]Listing artifacts...[/dim]")

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
            start_line = tool_data.get("start_line", 1)
            end_line = tool_data.get("end_line", start_line + 100)
            return analyzer.view_file_lines(
                tool_data.get("path"),
                start_line,
                end_line
            )
        elif tool_name == "list_dir_tree":
            return analyzer.list_dir_tree(tool_data.get("path", "."), tool_data.get("max_depth", 2))
        elif tool_name == "index_workspace":
            return rag_engine.index_workspace()
        elif tool_name == "semantic_search":
            return rag_engine.semantic_search(tool_data.get("query", ""), tool_data.get("top_k", 10))
        elif tool_name == "get_codebase_summary":
            return analyzer.get_codebase_summary()
        elif tool_name == "extract_symbols":
            return analyzer.extract_symbols(tool_data.get("path", ""))
        elif tool_name == "dependency_graph":
            return rag_engine.get_dependency_graph()
        elif tool_name == "incremental_index":
            return rag_engine.incremental_index()
        elif tool_name == "save_knowledge":
            return knowledge_base.save_knowledge(tool_data.get("topic", ""), tool_data.get("content", ""))
        elif tool_name == "read_knowledge":
            return knowledge_base.read_knowledge(tool_data.get("topic", ""))
        elif tool_name == "list_knowledge":
            return knowledge_base.list_knowledge()
        elif tool_name == "create_artifact":
            return artifact_manager.create_artifact(tool_data.get("title", ""), tool_data.get("content", ""), tool_data.get("artifact_type", "other"))
        elif tool_name == "update_artifact":
            return artifact_manager.update_artifact(tool_data.get("artifact_id", ""), tool_data.get("content", ""))
        elif tool_name == "read_artifact":
            return artifact_manager.read_artifact(tool_data.get("artifact_id", ""))
        elif tool_name == "list_artifacts":
            return artifact_manager.list_artifacts()
        elif tool_name == "delegate_task":
            return subagent_manager.delegate(tool_data.get("task", ""), tool_data.get("context", ""))
        elif tool_name == "browse_url":
            return web_tools.browse_url(tool_data.get("url"))
        elif tool_name == "fetch_url":
            return web_tools.fetch_url_content(tool_data.get("url"))
        elif tool_name == "browser_navigate":
            return browser_tool.navigate(tool_data.get("url"))
        elif tool_name == "browser_click":
            return browser_tool.click(tool_data.get("selector"))
        elif tool_name == "browser_type":
            return browser_tool.type(tool_data.get("selector"), tool_data.get("text"))
        elif tool_name == "browser_screenshot":
            return browser_tool.screenshot(tool_data.get("name", "screenshot"))
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
    Delegates to agents/reviewer.py.
    """
    return execute_verify(state, client, web_tools, terminal, workspace_path, episode_store)


# ============================================================
#  ROUTING LOGIC
# ============================================================

def route_after_plan(state: AgentState) -> Literal["develop", "verify", "end"]:
    intent = state.get("intent", "CODE_CHANGE")
    if intent == "READ_ONLY":
        from rich.console import Console
        c = Console()
        c.print("[bold cyan]Router:[/bold cyan] Read-only task detected. Skipping Developer and finishing.")
        return "end"
    return "develop"

def route_safety(state: AgentState) -> Literal["execute", "blocked"]:
    risk = state.get("risk_assessment", {})
    if risk.get("action") == "BLOCK":
        return "blocked"
    return "execute"


def should_continue(state: AgentState) -> Literal["analyze", "end"]:
    monologue = state.get("internal_monologue", "")
    score = state.get("success_score", 1.0)
    intent = state.get("intent", "CODE_CHANGE")
    
    if intent == "READ_ONLY" and not state.get("verification_result"):
        from rich.console import Console
        c = Console()
        c.print("[bold cyan]Router:[/bold cyan] Read-only task with no execution results. Ending loop.")
        return "end"

    if score < 0.5:
        from rich.console import Console
        c = Console()
        c.print("[yellow]Auto-Correction: Success score low (< 0.5). Routing back to analyze...[/yellow]")
        return "analyze"

    # Did we just run actual tools to get data?
    last_result = state.get("verification_result", "")
    just_ran_tools = bool(last_result and "No tools were needed" not in last_result)

    # For read-only tasks, loop back to analyze to generate the final answer ONLY IF we just got real tool results
    if intent == "READ_ONLY" and just_ran_tools:
        from rich.console import Console
        c = Console()
        c.print("[bold cyan]Router:[/bold cyan] Read-only task with tool results. Routing back to analyze for final answer.")
        return "analyze"

    if state.get("finished") or "FINISHED" in monologue:
        return "end"

    # Default to looping back so the agent can chain actions until it explicitly declares FINISHED
    from rich.console import Console
    Console().print("[bold cyan]Router:[/bold cyan] Task not marked FINISHED. Routing back to analyze for next step.")
    return "analyze"


# ============================================================
#  BUILD THE GRAPH
# ============================================================

workflow = StateGraph(AgentState)

workflow.add_node("analyze", analyze_node)
workflow.add_node("plan", plan_node)
workflow.add_node("develop", develop_node)
workflow.add_node("safety_gate", safety_gate_node)
workflow.add_node("execute", execute_node)
workflow.add_node("verify", verify_node)

workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "plan")
workflow.add_edge("plan", "develop")
workflow.add_edge("develop", "safety_gate")

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