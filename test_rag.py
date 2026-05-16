import os
import sys
# Fix windows print encoding for emojis
sys.stdout.reconfigure(encoding='utf-8')
from tools.rag_engine import RAGEngine

def main():
    print("--- Testing RAGEngine ---")
    workspace = os.getcwd()
    engine = RAGEngine(workspace)
    
    print("\n1. Testing index_workspace()...")
    res = engine.index_workspace()
    print("Result:", res)
    
    print("\n2. Testing get_dependency_graph()...")
    graph = engine.get_dependency_graph()
    print(graph[:500] + "..." if len(graph) > 500 else graph)
    
    print("\n3. Testing semantic_search (Global Query)...")
    summary = engine.semantic_search("What is this codebase about?")
    print(summary[:500] + "..." if len(summary) > 500 else summary)
    
    print("\n4. Testing incremental_index()...")
    inc = engine.incremental_index()
    print("Result:", inc)
    
    print("\n--- All tests completed ---")

if __name__ == "__main__":
    main()
