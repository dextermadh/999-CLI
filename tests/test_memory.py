import os
from tools.episode_store import EpisodeStore

def test_memory():
    print("--- Testing EpisodeStore ---")
    
    workspace_root = os.getcwd()
    store = EpisodeStore(workspace_root=workspace_root)
    
    print("\n1. Adding an episode...")
    store.add_episode(
        request="How do I create a file in Python?",
        solution="Use with open('filename', 'w') as f:\n    f.write('content')"
    )
    
    print("\n2. Searching for similar request...")
    results = store.search_similar("creating a file")
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"Request: {r['request']}")
        print(f"Solution: {r['solution']}")
        print(f"Distance: {r['distance']}")
        
    print("\n--- Test Completed ---")

if __name__ == "__main__":
    test_memory()
