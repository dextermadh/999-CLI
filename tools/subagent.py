import os
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
