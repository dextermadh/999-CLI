import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from core.graph import workflow
from langchain_core.messages import HumanMessage

def test_swarm():
    print("--- Testing Swarm Flow ---")
    
    # Initialize state
    state = {
        "messages": [HumanMessage(content="Tell me what is this codebase about")],
        "current_dir": ".",
        "allowed_tools": ["read_file", "write_file", "patch_file", "run_terminal", "extract_symbols", "view_file_lines"],
        "model_name": "gemma-4-e4b"
    }
    
    # Compile graph
    app = workflow.compile()
    
    # Run graph
    print("Running graph...")
    res = app.invoke(state)
    
    print("\nFinal State:")
    print(f"Plan: {res.get('plan')}")
    print(f"Internal Monologue: {res.get('internal_monologue')}")
    print(f"Verification: {res.get('verification_result')}")
    
    print("\n--- Test Completed ---")

if __name__ == "__main__":
    test_swarm()
