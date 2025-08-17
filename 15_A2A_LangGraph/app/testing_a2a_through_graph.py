"""
Simple LangGraph Agent that communicates with the A2A Agent through the A2A protocol.

This implementation creates a simple agent with LangGraph that can:
1. Take user input
2. Forward requests to the A2A Agent running on localhost:10000
3. Return responses back to the user

Usage:
    python app/testing_a2a_through_graph.py
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Annotated
from uuid import uuid4

import httpx
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAgentState(TypedDict):
    """State schema for our simple test agent."""
    messages: Annotated[List, add_messages]
    last_a2a_response: str


async def _async_query_a2a_agent(query: str) -> str:
    """
    Async helper function to communicate with the A2A Agent.
    """
    base_url = 'http://localhost:10000'
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
            # Initialize A2A card resolver and client
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=base_url,
            )
            
            # Get agent card
            agent_card = await resolver.get_agent_card()
            logger.info(f"Connected to agent: {agent_card.name}")
            
            # Initialize client
            client = A2AClient(
                httpx_client=httpx_client, 
                agent_card=agent_card
            )
            
            # Prepare message
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': query}
                    ],
                    'message_id': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            # Send message and get response
            response = await client.send_message(request)
            
            # Extract response content
            response_text = "No content found"
            
            if hasattr(response, 'root') and hasattr(response.root, 'result'):
                result = response.root.result
                
                # Try to get artifacts first
                if hasattr(result, 'artifacts') and result.artifacts:
                    for artifact in result.artifacts:
                        if hasattr(artifact, 'parts') and artifact.parts:
                            for part in artifact.parts:
                                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                    response_text = part.root.text
                                    break
                            if response_text != "No content found":
                                break
                
                # Try to get from messages if no artifacts found
                if response_text == "No content found" and hasattr(result, 'messages') and result.messages:
                    for message in reversed(result.messages):  # Start from the last message
                        if hasattr(message, 'parts') and message.parts:
                            for part in message.parts:
                                if hasattr(part, 'text') and part.text:
                                    response_text = part.text
                                    break
                            if response_text != "No content found":
                                break
                
                # Try to get from status message if still not found
                if response_text == "No content found" and hasattr(result, 'status') and hasattr(result.status, 'message'):
                    status_message = result.status.message
                    if hasattr(status_message, 'parts') and status_message.parts:
                        for part in status_message.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text = part.text
                                break
                
                # Debug: log the response structure if we still couldn't extract content
                if response_text == "No content found":
                    logger.info(f"Response structure: {response.model_dump() if hasattr(response, 'model_dump') else str(response)}")
                    return "Received response from A2A agent but couldn't extract content. Check logs for response structure."
            
            return response_text
            
    except Exception as e:
        logger.error(f"Error communicating with A2A agent: {e}")
        return f"Error communicating with A2A agent: {str(e)}"


@tool
def query_a2a_agent(query: str) -> str:
    """
    Tool to communicate with the A2A Agent running on localhost:10000.
    
    Args:
        query: The question or request to send to the A2A agent
        
    Returns:
        The response from the A2A agent
    """
    # Run the async function synchronously
    return asyncio.run(_async_query_a2a_agent(query))


def call_model(state: TestAgentState) -> Dict[str, Any]:
    """
    Simple node that processes user messages and decides whether to use the A2A tool.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, HumanMessage):
        # User asked a question, we should query the A2A agent
        # Return a message indicating we need to use the tool
        response = AIMessage(
            content="I'll query the A2A agent for you.",
            tool_calls=[{
                "name": "query_a2a_agent",
                "args": {"query": last_message.content},
                "id": str(uuid4())
            }]
        )
        return {"messages": [response]}
    
    # If it's not a human message, just echo back
    response = AIMessage(content="I can help you query the A2A agent. What would you like to know?")
    return {"messages": [response]}


def should_continue(state: TestAgentState) -> str:
    """
    Decide whether to continue to tool execution or end.
    """
    last_message = state["messages"][-1]
    
    # If the last message has tool calls, go to action
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "action"
    
    # Otherwise, we're done
    return "end"


def create_test_agent_graph():
    """
    Create a simple LangGraph that can communicate with the A2A agent.
    """
    # Create the graph
    graph = StateGraph(TestAgentState)
    
    # Create tool node
    tool_node = ToolNode([query_a2a_agent])
    
    # Add nodes
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    
    # Set entry point
    graph.set_entry_point("agent")
    
    # Add edges
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "action": "action",
            "end": END
        }
    )
    
    # After tool execution, end the conversation
    graph.add_edge("action", END)
    
    return graph.compile()


async def run_interactive_session():
    """
    Run an interactive session with the test agent.
    """
    print("🤖 Simple LangGraph Agent - A2A Protocol Tester")
    print("This agent will forward your questions to the A2A agent running on localhost:10000")
    print("Type 'quit' to exit\n")
    
    # Create the agent graph
    agent_graph = create_test_agent_graph()
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not user_input:
                continue
            
            print("🔄 Processing your request...")
            
            # Run the agent
            result = agent_graph.invoke({
                "messages": [HumanMessage(content=user_input)],
                "last_a2a_response": ""
            })
            
            # Extract and display the response
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    print(f"🤖 A2A Agent: {last_message.content}\n")
                else:
                    print(f"🤖 Response: {last_message}\n")
            else:
                print("🤖 No response received\n")
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


async def run_test_queries():
    """
    Run some test queries to demonstrate the agent functionality.
    """
    print("🧪 Running test queries...\n")
    
    # Create the agent graph
    agent_graph = create_test_agent_graph()
    
    test_queries = [
        "What are the latest developments in artificial intelligence?",
        "Find me recent papers on transformer architectures",
        "What information is available about federal student loan programs?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}: {query}")
        print("🔄 Processing...")
        
        try:
            result = agent_graph.invoke({
                "messages": [HumanMessage(content=query)],
                "last_a2a_response": ""
            })
            
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    print(f"🤖 Response: {last_message.content}")
                else:
                    print(f"🤖 Response: {last_message}")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("-" * 50)


if __name__ == "__main__":
    print("Starting Simple LangGraph Agent for A2A Protocol Testing...")
    
    # Check if we should run in test mode or interactive mode
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        asyncio.run(run_test_queries())
    else:
        asyncio.run(run_interactive_session())