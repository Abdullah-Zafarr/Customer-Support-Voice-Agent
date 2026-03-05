import json
from groq import AsyncGroq
from datetime import datetime
from typing import List, Dict, Any, Optional

from .config import settings
from .logger import logger
from .database import SessionLocal, SupportTicket

# Initialize AsyncGroq client
groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a professional AI Customer Support Agent.
Your job is to greet the caller, understand their issue, and log a support ticket.
Be warm, conversational, and efficient — you're speaking over the phone.
Ask ONE question at a time. Keep responses under 2 sentences.

Flow:
1. Greet the caller and ask for their name.
2. Ask them to describe their issue briefly.
3. Once you have both, use the `create_ticket` tool to log the ticket.
4. Confirm the ticket was created and ask if there's anything else.

Never give long explanations. Be concise and helpful.
"""

def create_ticket_db(name: str, issue: str, urgency: str) -> dict:
    """Create a support ticket in the database."""
    db = SessionLocal()
    try:
        new_ticket = SupportTicket(
            customer_name=name,
            issue_description=issue,
            urgency=urgency,
            created_at=datetime.now()
        )
        db.add(new_ticket)
        db.commit()
        db.refresh(new_ticket)
        
        return {
            "status": "success",
            "ticket_id": new_ticket.id,
            "created_at": new_ticket.created_at.isoformat(),
            "message": f"Support ticket #{new_ticket.id} created for {name}."
        }
    except Exception as e:
        logger.error(f"Failed to create ticket: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        db.close()

# Define the tools (function calling)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a customer support ticket to log the caller's issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The full name of the customer."
                    },
                    "issue": {
                        "type": "string",
                        "description": "A brief description of the customer's issue or request."
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "emergency"],
                        "description": "The estimated urgency level of the issue."
                    }
                },
                "required": ["name", "issue", "urgency"]
            }
        }
    }
]

async def process_llm_turn(messages: List[Dict[str, Any]]) -> dict:
    """Process a turn with the LLM and handle function calling if needed."""
    
    # Ensure system prompt is present
    if not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        
    try:
        response = await groq_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=200,
        )
        
        response_message = response.choices[0].message
        
        # Check if LLM wants to call a function
        tool_calls = response_message.tool_calls
        if tool_calls:
            messages.append(response_message)
            tool_calls_info = []
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                
                if function_name == "create_ticket":
                    function_args = json.loads(tool_call.function.arguments)
                    logger.info(f"LLM called tool {function_name} with args: {function_args}")
                    
                    function_response = create_ticket_db(
                        name=function_args.get("name"),
                        issue=function_args.get("issue"),
                        urgency=function_args.get("urgency", "medium")
                    )
                    
                    tool_calls_info.append({
                        "name": function_name,
                        "result": {
                            "name": function_args.get("name"),
                            "issue": function_args.get("issue"),
                            "urgency": function_args.get("urgency", "medium"),
                            "id": function_response.get("ticket_id")
                        }
                    })
                    
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(function_response)
                    })
                    
            # Call LLM again to generate reply based on tool response
            second_response = await groq_client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )
            return {
                "response": second_response.choices[0].message.content,
                "tool_calls": tool_calls_info
            }
        
        return {
            "response": response_message.content,
            "tool_calls": []
        }

    except Exception as e:
        logger.error(f"Error calling Groq LLM: {e}")
        return {
            "response": "I'm sorry, I'm having trouble processing that right now.",
            "tool_calls": []
        }
