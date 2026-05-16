import os
import re
import json
import time
from langchain_core.messages import SystemMessage

def execute_verify(state, client, web_tools, terminal, workspace_path, episode_store):
    """
    Post-execution verification.
    Auto-runs tests if files were modified.
    Auto-checks localhost if a dev server was started.
    Rates success using LLM.
    """
    result_str = state.get('verification_result', '')
    monologue = state.get('internal_monologue', '')

    # 1. Auto-test discovery
    if "Successfully wrote" in result_str or "Successfully patched" in result_str:
        test_result = _auto_run_tests(workspace_path, terminal)
        if test_result:
            result_str += f"\n\n--- Auto Test Results ---\n{test_result}"

    # 2. Auto-Heal: Check localhost if dev server started
    if "npm run dev" in monologue or "next dev" in monologue or "npm start" in monologue:
        from rich.console import Console
        c = Console()
        c.print("[dim]🏥 Auto-Heal: Checking localhost for build errors...[/dim]")
        
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

    # 3. Success Grading
    score = 1.0
    reason = "No issues detected."
    
    try:
        user_query = ""
        for m in reversed(state['messages']):
            if m.type == "human":
                user_query = m.content
                break
                
        if user_query:
            prompt = f"""Analyze the user request and the execution results.
Rate the success of the execution on a scale from 0.0 to 1.0.
If the execution failed, errored, or did not fully address the request, rate it below 0.5.

NOTE: The query is running INSIDE the user's codebase. If the user asks about 'this codebase' or 'the project', they refer to the current workspace.

Read-Only Tasks:
- If the execution results state 'No tools were needed for this step', and the goal is just to answer a question, rate it 1.0. The system will loop back to let the Architect generate the final answer.

User Request: {user_query}
Execution Results: {result_str}

Respond ONLY with a valid JSON object: {{"score": float, "reason": "string"}}
"""
            from core.llm import call_llm
            response = call_llm(
                client=client,
                messages=[{"role": "system", "content": prompt}],
                model_name=state.get("model_name", "gemma-4-e4b"),
                temperature=0.1
            )
            resp_text = response.choices[0].message.content
            match = re.search(r'(\{.*\})', resp_text, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                score = data.get("score", 1.0)
                reason = data.get("reason", "")
                
                from rich.console import Console
                c = Console()
                color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
                c.print(f"[{color}]Auto-Reflection: Score {score} - {reason}[/{color}]")
                
                # Save to memory if successful
                if score >= 0.7 and user_query:
                    episode_store.add_episode(request=user_query, solution=state.get("plan", ""))
    except Exception:
        pass

    return {
        "verification_result": result_str,
        "success_score": score,
        "messages": [SystemMessage(content=f"Tool execution results:\n{result_str}\nAuto-Reflection Score: {score}\nReason: {reason}")] if result_str else []
    }

def _auto_run_tests(workspace_path, terminal) -> str:
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
