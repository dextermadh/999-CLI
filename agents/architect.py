import sys
import re
import json
import time
from openai import OpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.prompts import ARCHITECT_SYSTEM_PROMPT

def execute_plan(state, client, workspace_path):
    """
    Calls the LLM to generate a plan / response.
    Streams tokens live to stdout for real-time feedback.
    """
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
        "semantic_search": '{"tool": "semantic_search", "query": "string"}',
        "get_codebase_summary": '{"tool": "get_codebase_summary"}',
        "extract_symbols": '{"tool": "extract_symbols", "path": "string"}',
        "dependency_graph": '{"tool": "dependency_graph"}',
        "incremental_index": '{"tool": "incremental_index"}',
        "save_knowledge": '{"tool": "save_knowledge", "topic": "string", "content": "string"}',
        "read_knowledge": '{"tool": "read_knowledge", "topic": "string"}',
        "list_knowledge": '{"tool": "list_knowledge"}',
        "create_artifact": '{"tool": "create_artifact", "title": "string", "content": "string", "artifact_type": "string"}',
        "update_artifact": '{"tool": "update_artifact", "artifact_id": "string", "content": "string"}',
        "read_artifact": '{"tool": "read_artifact", "artifact_id": "string"}',
        "list_artifacts": '{"tool": "list_artifacts"}',
        "delegate_task": '{"tool": "delegate_task", "task": "string", "context": "string"}'
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
    if len(history) > 50:
        history = history[-50:]

    def _role(m):
        if m.type == "human": return "user"
        if m.type == "ai": return "assistant"
        return "user"

    formatted_history = []
    for m in history:
        content = m.content
        
        if len(content) > 30000:
            lines = content.split('\n')
            truncated_lines = []
            char_count = 0
            for line in lines:
                if char_count + len(line) > 28000:
                    break
                truncated_lines.append(line)
                char_count += len(line) + 1
            content = '\n'.join(truncated_lines) + f"\n\n[Content truncated: {len(''.join(lines))} total chars.]"

        if m.type == "ai":
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
    
    from rich.console import Console
    from rich.text import Text
    stream_console = Console()
    
    try:
        print("\n[bold green]999-CLI[/bold green] [dim]is thinking...[/dim]\n")
        
        from core.llm import call_llm
        stream = call_llm(
            client=client,
            messages=messages,
            model_name=state.get("model_name", "gemma-4-e4b"),
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
            
            if not silence_active:
                if any(p in thought_and_action[-50:] for p in silence_patterns):
                    silence_active = True
                else:
                    clean_delta = delta
                    for p in silence_patterns:
                        clean_delta = clean_delta.replace(p[:5], "")
                    
                    if clean_delta:
                        stream_console.print(Text(clean_delta, style="italic dim"), end="")
                        did_stream = True
            else:
                if any(p in thought_and_action[-20:] for p in ["</tool_call>", "</|tool_call|>", "}"]):
                    if thought_and_action.endswith("\n") or "FINISHED" in thought_and_action[-10:]:
                        silence_active = False
        
        print("\n")
    except Exception as e:
        thought_and_action = f"(Streaming error: {str(e)})"

    if not thought_and_action.strip() or len(thought_and_action.strip()) < 5:
        try:
            messages.append({"role": "user", "content": "Please continue. Use a tool call if necessary."})
            from core.llm import call_llm
            response = call_llm(
                client=client,
                messages=messages,
                model_name=state.get("model_name", "gemma-4-e4b"),
                temperature=0.7,
                max_tokens=2048
            )
            thought_and_action = response.choices[0].message.content or "(Model remains silent.)"
        except Exception as e:
            thought_and_action = f"(Fatal Error: {str(e)})"

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Parse text output with tags
    plan = thought_and_action
    intent = "CODE_CHANGE"
    finished = "FINISHED" in thought_and_action
    
    if "[INTENT: READ_ONLY]" in thought_and_action:
        intent = "READ_ONLY"
    elif "[INTENT: CODE_CHANGE]" in thought_and_action:
        intent = "CODE_CHANGE"

    return {
        "internal_monologue": thought_and_action,
        "did_stream": did_stream,
        "plan": plan,
        "intent": intent,
        "finished": finished,
        "messages": [AIMessage(content=thought_and_action)],
        "token_usage": {
            "prompt": 0,
            "completion": 0,
            "time_ms": elapsed_ms
        }
    }
