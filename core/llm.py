import re
from openai import OpenAI
from langchain_core.messages import BaseMessage

def call_llm(client: OpenAI, messages: list, model_name: str = "gemma-4-e4b", temperature: float = 0.1, max_tokens: int = 4096, stream: bool = False):
    """
    Unified wrapper for LLM calls to ensure correct payload formatting for LM Studio.
    Converts LangChain messages to raw dicts if necessary.
    """
    formatted_messages = []
    
    for m in messages:
        if isinstance(m, dict):
            formatted_messages.append(m)
        elif isinstance(m, BaseMessage):
            # Convert LangChain message to dict
            role = "user"
            if m.type == "ai":
                role = "assistant"
            elif m.type == "system":
                role = "system"
            elif m.type == "human":
                role = "user"
                
            formatted_messages.append({"role": role, "content": m.content})
        else:
            # Fallback for strings or other types
            formatted_messages.append({"role": "user", "content": str(m)})
            
    # Remove any empty messages or messages without content
    formatted_messages = [m for m in formatted_messages if m.get("content")]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )
        return response
    except Exception as e:
        print(f"LLM Call Failed: {str(e)}")
        raise e
