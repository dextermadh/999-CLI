import subprocess
import os
import threading
import time


class LocalTerminal:
    def __init__(self, workspace_root: str):
        self.workspace = os.path.abspath(workspace_root)
        self._background_processes = []

    def execute(self, command: str, timeout: int = 60) -> str:
        """Executes a terminal command safely within the workspace."""
        
        # Detect true background servers (that never exit)
        server_patterns = [
            "npm run dev", "npm start", "yarn dev",
            "python -m http.server", "uvicorn", "flask run",
            "ng serve", "vite", "next dev"
        ]
        
        # Detect installers/heavy tasks (that take time but DO exit)
        heavy_tasks = ["npm install", "npm i", "npx create-", "pip install", "yarn install"]
        
        is_server = any(pat in command for pat in server_patterns)
        is_heavy = any(pat in command for pat in heavy_tasks)
        
        if is_server:
            return self._execute_background(command)
            
        # Increase timeout for heavy tasks (5 minutes)
        actual_timeout = 300 if is_heavy else timeout
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=actual_timeout
            )
            
            output = result.stdout if result.stdout else ""
            error = result.stderr if result.stderr else ""
            
            if result.returncode == 0:
                return f"Success:\n{output}"
            else:
                return f"Error (Exit Code {result.returncode}):\n{error}"
                
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 60s."
        except Exception as e:
            return f"Error: {str(e)}"

    def _execute_background(self, command: str) -> str:
        """Starts a long-running command in the background and captures initial output."""
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=self.workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self._background_processes.append(process)
            
            # Wait a few seconds to capture initial output (startup logs)
            initial_output = []
            
            def _read_output():
                try:
                    for line in iter(process.stdout.readline, ''):
                        initial_output.append(line.rstrip())
                        if len(initial_output) >= 20:
                            break
                except:
                    pass
            
            reader = threading.Thread(target=_read_output, daemon=True)
            reader.start()
            reader.join(timeout=8)  # Wait up to 8 seconds for startup
            
            if process.poll() is not None:
                # Process already exited (error or instant command)
                error = process.stderr.read() if process.stderr else ""
                initial_output_text = '\n'.join(initial_output)
                return f"Process exited (code {process.returncode}):\n{initial_output_text}\n{error}"
            
            output_text = "\n".join(initial_output) if initial_output else "(server starting...)"
            return f"Background process started (PID {process.pid}):\n{output_text}\n\nThe process is running in the background."
            
        except Exception as e:
            return f"Error starting background process: {str(e)}"

    def stop_background(self) -> str:
        """Stops all background processes."""
        if not self._background_processes:
            return "No background processes running."
        
        stopped = 0
        for proc in self._background_processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
                stopped += 1
            except:
                try:
                    proc.kill()
                    stopped += 1
                except:
                    pass
        
        self._background_processes.clear()
        return f"Stopped {stopped} background process(es)."