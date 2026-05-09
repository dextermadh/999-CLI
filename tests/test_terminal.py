from tools.terminal import LocalTerminal
import os

# 1. Setup - point it to your workspace
workspace = "./workspace"
if not os.path.exists(workspace):
    os.makedirs(workspace)

terminal = LocalTerminal(workspace_root=workspace)

# 2. Test A: Basic Command (Verify success)
print("Testing: echo hello")
res_a = terminal.execute("echo hello")
print(f"Result: {res_a}")

# 3. Test B: Directory Verification (Verify sandbox)
# On Windows, 'cd' shows current directory
print("\nTesting: pwd/cd context")
res_b = terminal.execute("cd") 
print(f"Result: {res_b}")

# 4. Test C: Timeout Handling (Verify it doesn't hang your machine)
# 'timeout 5' works on Windows to pause
print("\nTesting: Timeout (should fail gracefully)")
res_c = terminal.execute("timeout /t 60", timeout=2) 
print(f"Result: {res_c}")