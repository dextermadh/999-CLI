import os
import concurrent.futures
from openai import OpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class SubagentManager:
    def __init__(self, client: OpenAI, model_name: str = "gemma-4-e4b"):
        self.client = client
        self.model_name = model_name

    def delegate(self, task: str, context: str = "") -> str:
        """Delegates a specific task to a subagent with a fresh context."""
        system_prompt = """You are a specialized subagent worker. 
Your goal is to execute the specific task assigned to you by the main Architect agent.
Focus ONLY on the task at hand. Be concise and precise.
Do not assume or hallucinate information not provided in the context.
"""
        user_prompt = f"TASK: {task}\n"
        if context:
            user_prompt += f"\nCONTEXT:\n{context}\n"
            
        try:
            from rich.console import Console
            c = Console()
            c.print(f"[dim]Spawning subagent for task: {task[:50]}...[/dim]")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            return f"Subagent Response:\n{response.choices[0].message.content}"
        except Exception as e:
            return f"Error in subagent execution: {str(e)}"

    def delegate_parallel(self, tasks: list) -> str:
        """
        Executes multiple subagents concurrently, showing a live dashboard with active spinners in the terminal.
        """
        if not tasks:
            return "No tasks specified for delegation."
            
        import concurrent.futures
        from rich.live import Live
        from rich.table import Table
        from rich.spinner import Spinner
        
        # Initialize subagent states
        subagents = {}
        for i, t in enumerate(tasks):
            name = t.get("name") or f"Subagent {i+1}"
            subagents[name] = {
                "status": "thinking",
                "task": t.get("task", ""),
                "context": t.get("context", ""),
                "response": None
            }
            
        def make_renderable():
            table = Table(show_header=False, show_edge=False, box=None, padding=(0, 1))
            # Header matching screenshot style
            table.add_row(f"[bold white]{len(subagents)} subagents running[/bold white]\n")
            for name, info in subagents.items():
                if info["status"] == "thinking":
                    table.add_row(Spinner("dots", style="cyan"), f"[cyan]{name}[/cyan]")
                elif info["status"] == "done":
                    table.add_row("[green]v[/green]", f"[dim white]{name}[/dim white]")
                else:
                    table.add_row("[red]x[/red]", f"[red]{name} (failed)[/red]")
            return table

        def run_worker(name):
            info = subagents[name]
            system_prompt = """You are a specialized subagent worker. 
Your goal is to execute the specific task assigned to you by the main Architect agent.
Focus ONLY on the task at hand. Be concise and precise.
Do not assume or hallucinate information not provided in the context.
"""
            user_prompt = f"TASK: {info['task']}\n"
            if info["context"]:
                user_prompt += f"\nCONTEXT:\n{info['context']}\n"
                
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3
                )
                info["status"] = "done"
                info["response"] = response.choices[0].message.content
            except Exception as e:
                info["status"] = "failed"
                info["response"] = f"Error: {str(e)}"

        # Start concurrent execution with Live rendering
        with Live(make_renderable(), refresh_per_second=10) as live:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(subagents)) as executor:
                futures = {executor.submit(run_worker, name): name for name in subagents}
                for future in concurrent.futures.as_completed(futures):
                    # Update live rendering when any subagent completes
                    live.update(make_renderable())

        # Compile final unified report from all subagents
        report = []
        for name, info in subagents.items():
            report.append(f"=== SUBAGENT REPORT: {name} ===")
            report.append(f"Status: {info['status'].upper()}")
            report.append(f"Response:\n{info['response']}\n")
            
        return "\n".join(report)
