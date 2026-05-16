import os
import sys
from tools.knowledge_base import KnowledgeBase

def main():
    print("--- Testing KnowledgeBase ---")
    workspace = os.getcwd()
    kb = KnowledgeBase(workspace)
    
    print("\n1. Testing save_knowledge()...")
    res = kb.save_knowledge("Test Topic", "This is a test fact about the project.")
    print("Result:", res)
    
    print("\n2. Testing list_knowledge()...")
    list_res = kb.list_knowledge()
    print(list_res)
    
    print("\n3. Testing read_knowledge()...")
    read_res = kb.read_knowledge("Test Topic")
    print("Content:", read_res)
    
    print("\n--- All tests completed ---")

if __name__ == "__main__":
    main()
