import os
from tools.artifact_manager import ArtifactManager

def main():
    print("--- Testing ArtifactManager ---")
    workspace = os.getcwd()
    am = ArtifactManager(workspace)
    
    print("\n1. Testing create_artifact()...")
    res = am.create_artifact("Test Plan", "This is a test plan content.", "plan")
    print("Result:", res)
    
    print("\n2. Testing list_artifacts()...")
    list_res = am.list_artifacts()
    print(list_res)
    
    print("\n3. Testing read_artifact()...")
    read_res = am.read_artifact("test_plan")
    print("Content:", read_res)
    
    print("\n4. Testing update_artifact()...")
    update_res = am.update_artifact("test_plan", "Updated test plan content.")
    print("Result:", update_res)
    
    print("\n5. Testing read_artifact() after update...")
    read_res = am.read_artifact("test_plan")
    print("Content:", read_res)
    
    print("\n--- All tests completed ---")

if __name__ == "__main__":
    main()
