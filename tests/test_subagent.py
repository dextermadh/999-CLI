import os
from openai import OpenAI
from tools.subagent import SubagentManager

def main():
    print("--- Testing SubagentManager ---")
    
    # We need to simulate the client or use a real one if available
    # Since we are in a local environment with LM Studio, let's try to use it!
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    sm = SubagentManager(client, model_name="gemma-4-e4b")
    
    print("\n1. Testing delegate()...")
    res = sm.delegate("Summarize the purpose of a subagent.", "Context: Subagents are smaller models or isolated calls used to break down complex tasks.")
    print(res)
    
    print("\n--- All tests completed ---")

if __name__ == "__main__":
    main()
