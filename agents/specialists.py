import time
from langchain_core.messages import AIMessage
from core.llm import call_llm

TEST_ENGINEER_PROMPT = """You are the specialized Test Engineer agent in a multi-agent swarm.
Your job is to analyze the Architect's plan and write comprehensive unit tests (using pytest) for the functions and classes to be modified.

Rules:
1. Focus ONLY on writing test assertions. Save tests under the 'tests/' directory using write_file.
2. Use the following format for tool calls: <tool_call>write_file{{"path": "tests/test_filename.py", "content": "..."}}</tool_call>
3. If no tests are required or you are just answering a question, do not output any tool calls. Just state that no action is needed.
"""

PERF_OPTIMIZER_PROMPT = """You are the specialized Performance Optimizer agent in a multi-agent swarm.
Your job is to analyze the Architect's plan and ensure the code is optimized for execution speed, space complexity, and RAM usage.

Rules:
1. Audit the plan for performance bottlenecks (like O(N^2) loops or excessive file I/O).
2. If optimization is needed, output a patch or optimized write call to make the code faster.
3. If the code is already optimal or no edits are needed, just state that the performance profile is optimal.
"""

def execute_test_generation(state, client, workspace_path):
    """Calls the Test Engineer LLM to generate unit tests concurrently."""
    plan = state.get("plan", "No plan provided.")
    allowed = state.get('allowed_tools', [])
    tool_help = "\n".join([f"- {t}" for t in allowed])
    
    sys_prompt = TEST_ENGINEER_PROMPT + f"\n\nCURRENT PLAN TO EXECUTE:\n{plan}\n\nAvailable Tools:\n{tool_help}"
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Please write tests for this plan. Workspace: {workspace_path}"}
    ]
    
    start_time = time.time()
    try:
        response = call_llm(
            client=client,
            messages=messages,
            model_name=state.get("model_name", "gemma-4-e4b"),
            temperature=0.2,
            max_tokens=4096
        )
        output = response.choices[0].message.content
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "internal_monologue": output,
            "messages": [AIMessage(content=output)],
            "time_ms": elapsed_ms
        }
    except Exception as e:
        return {"internal_monologue": f"Test Engineer failed: {str(e)}", "messages": []}

def execute_perf_optimization(state, client, workspace_path):
    """Calls the Performance Optimizer LLM to audit performance concurrently."""
    plan = state.get("plan", "No plan provided.")
    allowed = state.get('allowed_tools', [])
    tool_help = "\n".join([f"- {t}" for t in allowed])
    
    sys_prompt = PERF_OPTIMIZER_PROMPT + f"\n\nCURRENT PLAN TO EXECUTE:\n{plan}\n\nAvailable Tools:\n{tool_help}"
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Please audit the performance of this plan. Workspace: {workspace_path}"}
    ]
    
    start_time = time.time()
    try:
        response = call_llm(
            client=client,
            messages=messages,
            model_name=state.get("model_name", "gemma-4-e4b"),
            temperature=0.1,
            max_tokens=4096
        )
        output = response.choices[0].message.content
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "internal_monologue": output,
            "messages": [AIMessage(content=output)],
            "time_ms": elapsed_ms
        }
    except Exception as e:
        return {"internal_monologue": f"Performance Optimizer failed: {str(e)}", "messages": []}
