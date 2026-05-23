import sys
import os
import json
import argparse
import subprocess
import re
import time
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, messages_from_dict, messages_to_dict
from core.graph import app, git_tools, terminal, client, _has_tool_calls, _get_display_text

console = Console()

# --- Session Management ---

def load_session(session_path):
    if session_path.exists():
        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return messages_from_dict(data)
        except Exception:
            pass
    return []

def save_session(session_path, messages):
    try:
        session_path.parent.mkdir(parents=True, exist_ok=True)
        with open(session_path, 'w', encoding='utf-8') as f:
            json.dump(messages_to_dict(messages), f)
    except Exception:
        pass

# --- Startup Banner ---

def _get_git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=os.getcwd(), capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""

def _detect_project() -> str:
    indicators = {
        "requirements.txt": "Python", "pyproject.toml": "Python",
        "package.json": "Node.js", "go.mod": "Go",
        "Cargo.toml": "Rust", "pom.xml": "Java",
    }
    detected = [lang for f, lang in indicators.items() if os.path.exists(os.path.join(os.getcwd(), f))]
    return ", ".join(set(detected)) if detected else "Unknown"

def print_startup_banner(mode: str, session_loaded: bool):
    cwd = os.getcwd()
    branch = _get_git_branch()
    project = _detect_project()
    config_exists = os.path.exists(os.path.join(cwd, ".999", "config.md"))

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan", width=16)
    table.add_column()
    table.add_row("Workspace", cwd)
    table.add_row("Project", project)
    table.add_row("Git Branch", branch if branch else "[dim]not a git repo[/dim]")
    table.add_row("Config", "[green]✓ .999/config.md loaded[/green]" if config_exists else "[dim]none[/dim]")
    table.add_row("Mode", f"[yellow]{mode}[/yellow]")
    table.add_row("Session", "[green]resumed[/green]" if session_loaded else "[dim]new[/dim]")

    console.print(Panel(table, title="[bold green]⚡ 999-CLI Software Engineering Suite[/bold green]", border_style="green"))
    console.print("[dim]Type /help for commands. Type exit to quit.[/dim]\n")

# --- Slash Commands ---

def handle_slash_command(cmd: str, inputs: dict, session_file: Path, cumulative_tokens: dict) -> bool:
    """Returns True if the command was handled (skip LLM call)."""

    if cmd == "/help":
        help_table = Table(title="Commands", show_header=True, header_style="bold cyan")
        help_table.add_column("Command", style="green")
        help_table.add_column("Description")
        help_table.add_row("/stop", "Stop all background processes (dev servers)")
        help_table.add_row("/undo", "Revert to last checkpoint (git stash pop)")
        help_table.add_row("/status", "Show git status")
        help_table.add_row("/diff", "Show current git diff")
        help_table.add_row("/clear", "Clear session history")
        help_table.add_row("/compact", "Summarize & compress chat history")
        help_table.add_row("/config", "Create project config (.999/config.md)")
        help_table.add_row("/mode [yolo|safe|default]", "Change approval mode")
        help_table.add_row("/model", "Select LLM model from local server")
        help_table.add_row("/tokens", "Show cumulative token usage")
        help_table.add_row("/speed [fast|deep]", "Toggle between fast and deep analysis modes")
        help_table.add_row("/graph", "Rebuild the Codebase Knowledge Graph")
        help_table.add_row("/trace <symbol>", "Trace dependencies & call path of a class or function")
        help_table.add_row("/impact <file>", "Perform structural change impact analysis")
        help_table.add_row("/help", "Show this help")
        help_table.add_row("exit / quit", "Exit 999-CLI")
        console.print(help_table)
        return True

    elif cmd == "/undo":
        result = git_tools.git_stash_pop()
        if "Skipped" in result:
            console.print("[yellow]Not a git repository. Undo unavailable.[/yellow]")
        elif "Error" in result:
            console.print(f"[red]{result}[/red]")
        else:
            console.print("[green]✓ Changes reverted to last checkpoint.[/green]")
        return True

    elif cmd == "/status":
        result = git_tools.git_status()
        console.print(Panel(result, title="[cyan]Git Status[/cyan]", border_style="cyan"))
        return True

    elif cmd == "/diff":
        from rich.syntax import Syntax
        result = git_tools.git_diff()
        if result and result != "(no output)" and not result.startswith("Skipped"):
            console.print(Syntax(result, "diff", theme="monokai", background_color="default"))
        else:
            console.print(f"[dim]{result}[/dim]")
        return True

    elif cmd == "/clear":
        inputs["messages"] = []
        inputs["verification_result"] = ""
        inputs["internal_monologue"] = ""
        session_file.unlink(missing_ok=True)
        console.print("[green]✓ Session history cleared.[/green]")
        return True

    elif cmd == "/compact":
        msg_count = len(inputs["messages"])
        if msg_count <= 2:
            console.print("[dim]Nothing to compact (conversation too short).[/dim]")
            return True

        # Keep first and last messages, summarize the middle
        summary_parts = []
        for m in inputs["messages"]:
            role = "User" if m.type == "human" else "Agent" if m.type == "ai" else "System"
            content_preview = m.content[:100] + "..." if len(m.content) > 100 else m.content
            summary_parts.append(f"{role}: {content_preview}")

        summary = "Previous conversation summary:\n" + "\n".join(summary_parts)
        inputs["messages"] = [
            SystemMessage(content=summary),
            inputs["messages"][-1]  # Keep the last message for continuity
        ]
        save_session(session_file, inputs["messages"])
        console.print(f"[green]✓ Compacted {msg_count} messages → 2. Context preserved.[/green]")
        return True

    elif cmd == "/config":
        config_dir = Path(os.getcwd()) / ".999"
        config_path = config_dir / "config.md"
        if config_path.exists():
            console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
            console.print(Panel(config_path.read_text(encoding='utf-8')[:500], title="Current Config", border_style="cyan"))
            return True

        project = _detect_project()
        config_content = f"""# 999-CLI Project Configuration

## Project Type
{project}

## Code Style
- Use consistent indentation
- Add docstrings to all functions
- Follow language-specific best practices

## Conventions
- Use descriptive variable names
- Keep functions focused and small
- Add error handling for edge cases

## Notes
- Edit this file to customize 999-CLI behavior for your project
"""
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_content, encoding='utf-8')
        console.print(f"[green]✓ Created project config at .999/config.md[/green]")
        console.print("[dim]Edit this file to customize how 999-CLI works with your project.[/dim]")
        return True

    elif cmd == "/model":
        import urllib.request
        server_url = str(client.base_url)
        try:
            with console.status(f"[bold blue]Fetching models from {server_url}...[/bold blue]"):
                models = client.models.list()

            model_ids = [m.id for m in models.data]
            if not model_ids:
                console.print(f"[yellow]No models found at {server_url}.[/yellow]")
                selected_model = Prompt.ask("Enter model name manually")
            else:
                console.print(f"\n[bold cyan]Available Models at {server_url}:[/bold cyan]")
                for i, m_id in enumerate(model_ids):
                    current = " [yellow](current)[/yellow]" if m_id == inputs.get("model_name") else ""
                    console.print(f" {i+1}. [green]{m_id}[/green]{current}")

                choice = Prompt.ask("\nSelect a model number or enter a custom name", default="1")
                if choice.isdigit() and 1 <= int(choice) <= len(model_ids):
                    selected_model = model_ids[int(choice) - 1]
                else:
                    selected_model = choice

            inputs["model_name"] = selected_model

            # --- Auto-load in LM Studio ---
            with console.status(f"[bold blue]Loading [green]{selected_model}[/green] in LM Studio...[/bold blue]"):
                try:
                    load_url = server_url.replace("/v1", "/api/v1")
                    if not load_url.endswith("/"): load_url += "/"
                    load_url += "models/load"

                    data = json.dumps({"model": selected_model}).encode("utf-8")
                    req = urllib.request.Request(load_url, data=data, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=120) as response:
                        if response.status == 200:
                            console.print(f"[green]✓ Model loaded successfully in LM Studio.[/green]")
                        else:
                            console.print(f"[yellow]⚠ Server returned status {response.status} during load.[/yellow]")
                except Exception as e:
                    err_msg = str(e).lower()
                    if "timed out" in err_msg or "timeout" in err_msg:
                        console.print(f"[yellow]⚠ Auto-load request timed out after 120s. The model may still be loading/active in LM Studio. Please check LM Studio's UI or logs.[/yellow]")
                    else:
                        console.print(f"[dim]Note: Could not auto-load in LM Studio. Error: {str(e)}[/dim]")

            console.print(f"[green]✓ Model set to: [bold]{selected_model}[/bold][/green]")
        except Exception as e:
            console.print(f"[red]Error fetching models: {str(e)}[/red]")
            selected_model = Prompt.ask("Enter model name manually")
            if selected_model:
                inputs["model_name"] = selected_model
                console.print(f"[green]✓ Model set to: [bold]{selected_model}[/bold][/green]")
        return True

    elif cmd.startswith("/mode"):
        parts = cmd.split()
        if len(parts) == 2 and parts[1] in ["yolo", "safe", "default"]:
            inputs["approval_mode"] = parts[1]
            console.print(f"[green]✓ Approval mode set to: [bold]{parts[1]}[/bold][/green]")
        else:
            console.print("[yellow]Usage: /mode [yolo|safe|default][/yellow]")
        return True

    elif cmd.startswith("/speed"):
        parts = cmd.split()
        if len(parts) == 2 and parts[1] in ["fast", "deep"]:
            inputs["speed_mode"] = parts[1]
            console.print(f"[green]✓ Speed mode set to: [bold]{parts[1]}[/bold][/green]")
        else:
            console.print("[yellow]Usage: /speed [fast|deep][/yellow]")
        return True

    elif cmd == "/tokens":
        console.print(Panel(
            f"Prompt tokens: {cumulative_tokens['prompt']}\n"
            f"Completion tokens: {cumulative_tokens['completion']}\n"
            f"Total inference time: {cumulative_tokens['time_ms']}ms",
            title="[cyan]Token Usage (Session)[/cyan]", border_style="cyan"
        ))
        return True

    elif cmd == "/stop":
        result = terminal.stop_background()
        console.print(f"[green]✓ {result}[/green]")
        return True

    elif cmd == "/graph":
        with console.status("[bold blue]Building Codebase Knowledge Graph...[/bold blue]"):
            try:
                from tools.knowledge_graph_builder import KnowledgeGraphBuilder
                builder = KnowledgeGraphBuilder(os.getcwd())
                builder.build_graph()
                console.print("[green]✓ Codebase Knowledge Graph successfully rebuilt! Saved to .999/knowledge_graph.json[/green]")
            except Exception as e:
                console.print(f"[red]Error building Knowledge Graph: {str(e)}[/red]")
        return True

    elif cmd.startswith("/trace"):
        parts = cmd.split()
        if len(parts) < 2:
            console.print("[yellow]Usage: /trace <symbol_name>[/yellow]")
            return True
            
        symbol = parts[1]
        with console.status(f"[bold blue]Tracing symbol '{symbol}' in Codebase Knowledge Graph...[/bold blue]"):
            try:
                from tools.knowledge_graph import CodeKnowledgeGraph
                kg = CodeKnowledgeGraph()
                kg_path = Path(os.getcwd()) / ".999" / "knowledge_graph.json"
                if not kg_path.exists():
                    console.print("[yellow]Knowledge Graph does not exist yet. Please run /graph first to build it.[/yellow]")
                    return True
                if not kg.load_from_disk(str(kg_path)):
                    console.print("[red]Error loading Knowledge Graph.[/red]")
                    return True
                
                # Find matching nodes
                matches = []
                for n_id, n_data in kg.nodes.items():
                    name = n_data.get("properties", {}).get("name", "")
                    if name.lower() == symbol.lower() or symbol.lower() in n_id.lower():
                        matches.append((n_id, n_data))
                
                if not matches:
                    console.print(f"[yellow]Symbol '{symbol}' not found in the Knowledge Graph.[/yellow]")
                    return True
                
                for n_id, n_data in matches[:3]: # Limit to top 3 matching symbols to prevent print bloat
                    n_type = n_data["type"]
                    props = n_data.get("properties", {})
                    name = props.get("name", n_id)
                    path = props.get("path", n_id.split(':')[1] if ':' in n_id else "")
                    start_line = props.get("start_line", "?")
                    
                    console.print(f"\n[bold cyan]🕸️ Traced Symbol: {name} ({n_type.upper()})[/bold cyan]")
                    console.print(f"  [dim]Defined in: {path} (Line {start_line})[/dim]")
                    
                    # Outgoing edges (what this calls or defines)
                    outgoing = []
                    for e_type, targets in n_data.get("edges", {}).items():
                        for t in targets:
                            t_name = kg.nodes.get(t, {}).get("properties", {}).get("name", t)
                            outgoing.append(f"    - {e_type} -> [green]{t_name}[/green]")
                            
                    # Incoming edges (what calls or references this)
                    incoming = []
                    for s_id, s_data in kg.nodes.items():
                        for e_type, targets in s_data.get("edges", {}).items():
                            if n_id in targets:
                                s_name = s_data.get("properties", {}).get("name", s_id)
                                incoming.append(f"    - [green]{s_name}[/green] -> {e_type}")
                                
                    if outgoing:
                        console.print("  [bold magenta]Outward Relations (Invocations/Definitions):[/bold magenta]")
                        for out in outgoing[:8]:
                            console.print(out)
                        if len(outgoing) > 8:
                            console.print(f"    [dim]... and {len(outgoing)-8} more outgoing relations[/dim]")
                    else:
                        console.print("  [dim]No outgoing invocations detected.[/dim]")
                        
                    if incoming:
                        console.print("  [bold yellow]Inward Relations (Dependents/Callers):[/bold yellow]")
                        for inc in incoming[:8]:
                            console.print(inc)
                        if len(incoming) > 8:
                            console.print(f"    [dim]... and {len(incoming)-8} more incoming relations[/dim]")
                    else:
                        console.print("  [dim]No callers/references detected.[/dim]")
            except Exception as e:
                console.print(f"[red]Error tracing symbol: {str(e)}[/red]")
        return True

    elif cmd.startswith("/impact"):
        parts = cmd.split()
        if len(parts) < 2:
            console.print("[yellow]Usage: /impact <file_path_or_class_name>[/yellow]")
            return True
            
        target = parts[1]
        with console.status(f"[bold blue]Calculating change impact footprint for '{target}'...[/bold blue]"):
            try:
                from tools.knowledge_graph import CodeKnowledgeGraph
                kg = CodeKnowledgeGraph()
                kg_path = Path(os.getcwd()) / ".999" / "knowledge_graph.json"
                if not kg_path.exists():
                    console.print("[yellow]Knowledge Graph does not exist yet. Please run /graph first to build it.[/yellow]")
                    return True
                if not kg.load_from_disk(str(kg_path)):
                    console.print("[red]Error loading Knowledge Graph.[/red]")
                    return True
                
                # Find matching node
                target_node_id = None
                for n_id, n_data in kg.nodes.items():
                    name = n_data.get("properties", {}).get("name", "")
                    path = n_data.get("properties", {}).get("path", "")
                    if target.lower() in n_id.lower() or target.lower() == name.lower() or target.lower() in path.lower():
                        target_node_id = n_id
                        break
                        
                if not target_node_id:
                    console.print(f"[yellow]Target '{target}' not found in Knowledge Graph.[/yellow]")
                    return True
                
                # Run reverse BFS to find all incoming dependents
                # Queue stores: (node_id, current_distance)
                queue = [(target_node_id, 0)]
                visited = {target_node_id}
                
                impacted_by_distance = {} # distance -> List[node_data]
                
                while queue:
                    curr_id, dist = queue.pop(0)
                    if dist > 0:
                        if dist not in impacted_by_distance:
                            impacted_by_distance[dist] = []
                        impacted_by_distance[dist].append(kg.nodes[curr_id])
                        
                    # Find all nodes that have an edge pointing TO curr_id
                    for source_id, s_data in kg.nodes.items():
                        if source_id in visited:
                            continue
                        for edge_type, targets in s_data.get("edges", {}).items():
                            if curr_id in targets:
                                visited.add(source_id)
                                queue.append((source_id, dist + 1))
                                break
                                
                t_data = kg.nodes[target_node_id]
                t_name = t_data.get("properties", {}).get("name", target_node_id)
                t_type = t_data["type"]
                
                console.print(f"\n[bold red]💥 Impact Analysis Footprint (Blast Radius) for {t_name} ({t_type.upper()})[/bold red]")
                
                if not impacted_by_distance:
                    console.print("[green]✓ Isolated Component: No downstream dependents found. Safe to modify with zero structural impact![/green]")
                    return True
                    
                console.print(f"[yellow]⚠ Modifying this component has a potential blast radius of {len(visited) - 1} dependents:[/yellow]\n")
                
                for dist in sorted(impacted_by_distance.keys()):
                    console.print(f"  [bold magenta]Distance {dist} (Blast Zone):[/bold magenta]")
                    for dep in impacted_by_distance[dist]:
                        d_name = dep.get("properties", {}).get("name", dep["id"])
                        d_type = dep["type"]
                        d_path = dep.get("properties", {}).get("path", "")
                        console.print(f"    • [yellow]{d_name}[/yellow] ({d_type}) [dim]in {d_path}[/dim]")
            except Exception as e:
                console.print(f"[red]Error calculating change impact: {str(e)}[/red]")
        return True

    return False

# --- Main ---

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="999-CLI Software Engineering Suite")
    parser.add_argument("--yolo", action="store_true", help="Auto-approve all file changes")
    parser.add_argument("--safe", action="store_true", help="Require approval for ALL actions")
    args = parser.parse_args()

    if args.yolo:
        approval_mode = "yolo"
    elif args.safe:
        approval_mode = "safe"
    else:
        approval_mode = "default"

    session_file = Path(os.getcwd()) / ".999" / "session.json"
    loaded_messages = load_session(session_file)

    print_startup_banner(approval_mode, bool(loaded_messages))

    # Cumulative token tracking
    cumulative_tokens = {"prompt": 0, "completion": 0, "time_ms": 0}

    # Initialize state
    inputs = {
        "messages": loaded_messages,
        "current_dir": os.getcwd(),
        "allowed_tools": [
            "read_file", "write_file", "list_files", "patch_file", "search_code", "run_terminal",
            "delete_file", "move_file", "create_dir", "get_file_info", "view_file_lines", "fetch_url", "browse_url",
            "list_dir_tree", "index_workspace", "semantic_search", "get_codebase_summary", "extract_symbols",
            "dependency_graph", "incremental_index", "save_knowledge", "read_knowledge", "list_knowledge",
            "create_artifact", "update_artifact", "read_artifact", "list_artifacts", "delegate_task", "delegate_parallel",
            "git_status", "git_diff", "git_log", "git_commit", "git_checkout", "git_stash", "git_stash_pop",
            "trace_symbol", "impact_analysis", "run_security_scan", "run_unit_tests", "profile_performance"
        ],
        "internal_monologue": "",
        "analysis_result": "",
        "verification_result": "",
        "approval_mode": approval_mode,
        "model_name": "gemma-4-e4b",  # Default model
        "speed_mode": "fast",  # Default to fast mode
    }

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]>[/bold cyan]")

            if user_input.lower() in ['exit', 'quit']:
                save_session(session_file, inputs["messages"])
                console.print("[yellow]Exiting 999-CLI. Session saved.[/yellow]")
                break

            # Handle slash commands
            if user_input.startswith("/"):
                handled = handle_slash_command(user_input.strip(), inputs, session_file, cumulative_tokens)
                if handled:
                    continue

            inputs["messages"].append(HumanMessage(content=user_input))

            # Reset stale turn-specific state
            inputs["internal_monologue"] = ""
            inputs["analysis_result"] = ""
            inputs["verification_result"] = ""
            inputs["finished"] = False
            inputs["plan"] = None
            inputs["_loop_count"] = 0
            inputs["risk_assessment"] = None
            inputs["success_score"] = None

            # Track the agent's final response for history
            agent_response = ""
            turn_start = time.time()

            # ===== PROCESS GRAPH EVENTS =====
            auto_continue = True
            while auto_continue:
                auto_continue = False
                try:
                    # We DON'T start a status/spinner here because graph.py handles its own streaming UI
                    for event in app.stream(inputs):
                        for node_name, node_state in event.items():
                            # SYNC: Correctly MERGE state updates
                            for key, value in node_state.items():
                                if key == "messages" and isinstance(value, list):
                                    for msg in value:
                                        if not any(m.type == msg.type and m.content == msg.content for m in inputs["messages"]):
                                            inputs["messages"].append(msg)
                                else:
                                    inputs[key] = value

                            if node_name == "analyze":
                                console.print("[bold cyan]🔍 Mapping codebase...[/bold cyan]")
                            elif node_name == "plan":
                                monologue = node_state.get('internal_monologue', '')
                                # If the model requested a tool, we should automatically process the result
                                if "<tool_call" in monologue or "Executing Tool" in monologue:
                                    auto_continue = True
                            elif node_name == "safety_gate":
                                risk = node_state.get('risk_assessment', {})
                                if risk.get('action') == 'BLOCK':
                                    console.print(f"\n[bold red]🚫 BLOCKED: {risk.get('reasoning')}[/bold red]")
                                    auto_continue = False
                            elif node_name == "verify":
                                result = node_state.get('verification_result', '')
                                if result and "successfully" not in result:
                                    display_result = result[:1000] + "\n\n... [Output Truncated for UI Display. The Agent received the full result!] ..." if len(result) > 1000 else result
                                    console.print(Panel(display_result, title="[bold blue]📋 Results[/bold blue]", border_style="blue"))

                            # Handle UI updates
                            if node_name == "plan":
                                monologue = node_state.get('internal_monologue', '')
                                if event['plan'].get('internal_monologue'):
                                    display_text = _get_display_text(monologue)
                                    if display_text:
                                        console.print(Panel(Markdown(display_text), title="[bold green]999-CLI[/bold green]", border_style="green"))

                        if auto_continue:
                            console.print("[dim]─[/dim]" * 40)

                except KeyboardInterrupt:
                    console.print("\n[yellow]Turn cancelled. Returning to prompt...[/yellow]")
                    auto_continue = False
                except Exception as e:
                    console.print(f"\n[bold red]System Error: {str(e)}[/bold red]")
                    auto_continue = False

                elapsed = time.time() - turn_start
                if elapsed > 1:
                    console.print(f"[dim]⏱️ Total: {elapsed:.1f}s[/dim]")

            # Save session after each interaction cycle
            save_session(session_file, inputs["messages"])

        except KeyboardInterrupt:
            # Outer interrupt exits the CLI
            console.print("\n[yellow]Exiting 999-CLI. Session saved.[/yellow]")
            save_session(session_file, inputs["messages"])
            break
        except Exception as e:
            import traceback
            console.print("\n[bold red]FATAL CLI ERROR:[/bold red]")
            console.print(str(e))
            traceback.print_exc()
            break

if __name__ == "__main__":
    main()