from pydantic import BaseModel, Field
from openai import OpenAI
import os

class RiskAssessment(BaseModel): 
    risk_score: float = Field(description='0.0 to 1.0, where 1.0 is high risk')
    category: str = Field(description="e.g., 'File Deletion', 'Sensitive Data Access', 'Safe'")
    reasoning: str = Field(description='brief explanation of the risk')
    action: str = Field(description="'ALLOW', 'MONITOR', or 'BLOCK'")

class RiskClassifier:
    def __init__(self):
        # Pointing to LM Studio instance
        self.client = OpenAI(base_url=os.getenv('LM_STUDIO_URL', 'http://localhost:1234/v1'), api_key='lm-studio')
        self.model = '999-4-e4b-it'

    def analyze_instruction(self, user_input: str, agent_plan: str, model: str = None) -> RiskAssessment:
        '''
        Analyze the agent's intended action before it reaches the tools.
        '''
        target_model = model or self.model
        
        system_prompt = '''
            You are a Security Gatekeeper. Analyze the proposed action for risks.
            
            RULES:
            1. You MUST ONLY respond with a valid JSON object.
            2. NEVER emit tool calls like <tool_call> or {"tool": ...}.
            3. NEVER apologize or talk conversationally.
            4. If the action is safe, use action: "ALLOW".
            5. If the action is risky (modifying core files), use action: "MONITOR".
            6. If the action is destructive (rm, deleting keys), use action: "BLOCK".
            
            OUTPUT FORMAT (JSON ONLY):
            {
                "risk_score": float,
                "category": "string",
                "reasoning": "string",
                "action": "ALLOW | MONITOR | BLOCK"
            }
        '''

        response = self.client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Input: {user_input}\nProposed Action: {agent_plan}"}
            ]
        )

        raw_content = response.choices[0].message.content
        
        try:
            import json
            import re
            
            # 1. Look for ```json ... ``` blocks
            json_blocks = re.findall(r'```(?:json)?\s*\n(.*?)\n```', raw_content, re.DOTALL)
            if json_blocks:
                content_to_parse = json_blocks[0]
            else:
                # 2. Try finding the first '{' and the last '}'
                match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                content_to_parse = match.group(0) if match else raw_content
                
            parsed = json.loads(content_to_parse)
            
            # Handle potential nesting (e.g., {"RiskAssessment": {...}})
            if "risk_score" not in parsed and len(parsed) == 1:
                first_val = list(parsed.values())[0]
                if isinstance(first_val, dict) and "risk_score" in first_val:
                    parsed = first_val
                    
            return RiskAssessment(**parsed)
            
        except Exception as e:
            # Fallback if parsing or validation fails
            print(f"\n[RiskClassifier Warning] Failed to parse JSON: {e}")
            return RiskAssessment(
                risk_score=0.5,
                category="Parsing Error",
                reasoning="Failed to parse model output into valid RiskAssessment schema.",
                action="MONITOR"
            )