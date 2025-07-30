from fastapi import FastAPI
from fastapi.responses import StreamingResponse  # For streaming responses
import uuid
from typing import Any
import os
import uvicorn
import asyncio
from ag_ui.core import (
    RunAgentInput,
    StateSnapshotEvent,
    EventType,
    RunStartedEvent,
    RunFinishedEvent,
    TextMessageStartEvent,
    TextMessageEndEvent,
    TextMessageContentEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolCallArgsEvent,
    StateDeltaEvent
)
from ag_ui.encoder import EventEncoder
from copilotkit import CopilotKitState
from autogen.agentchat import a_initiate_group_chat, initiate_group_chat
from autogen.agentchat.group.patterns import DefaultPattern
from stock_analysis import chat_bot
from autogen.agentchat.group import ContextVariables
from openai import OpenAI
from dotenv import load_dotenv
import json
load_dotenv()

app = FastAPI()


class AgentState(CopilotKitState):
    """
    This is the state of the agent.
    It is a subclass of the MessagesState class from langgraph.
    """

    tools: list
    messages: list
    be_stock_data: Any
    be_arguments: dict
    available_cash: int
    investment_summary : dict
    tool_logs : list

@app.post("/ag2-agent")
async def ag2_agent(input_data: RunAgentInput):
    try:

        async def event_generator():
            encoder = EventEncoder()
            event_queue = asyncio.Queue()

            async def emit_event(event):
                event_queue.put_nowait(event)
            def sample_function(numb : int):
                print(numb)
            message_id = str(uuid.uuid4())

            yield encoder.encode(
                RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                )
            )

            yield encoder.encode(
                StateSnapshotEvent(
                    type=EventType.STATE_SNAPSHOT, 
                    snapshot={
                        "available_cash": input_data.state["available_cash"],
                        "investment_summary" : input_data.state["investment_summary"],
                        "investment_portfolio" : input_data.state["investment_portfolio"],
                        "investment_date" : input_data.state["investment_date"],
                        "tool_logs" : []
                    }
                )
            )       
            # chat_bot.context_variables.clear()
            shared_context = ContextVariables(
                data={"tool_logs": [], "messages": input_data.messages, "emitEvent": emit_event, "available_cash": input_data.state["available_cash"], "investment_portfolio" : input_data.state["investment_portfolio"], "be_arguments" : {}, "sample_function": sample_function}
            )
            pattern = DefaultPattern(
                initial_agent=chat_bot,
                agents=[chat_bot],
                # agents=[chat_bot,stock_data_bot,cash_allocation_bot, insights_bot],
                # group_manager_args={"emit_event": emit_event},
                context_variables=shared_context
            )
            
            if input_data.messages[-1].role == "user":
                try:
                    tool_log_id = str(uuid.uuid4())
                    yield encoder.encode(
                        StateDeltaEvent(
                            type=EventType.STATE_DELTA,
                            delta=[
                                {
                                    "op": "add",
                                    "path": "/tool_logs/-",
                                    "value": {
                                        "message": "Analyzing user query",
                                        "status": "processing",
                                        "id": tool_log_id,
                                    },
                                }
                            ],
                        )
                    )
                    agent_task = asyncio.create_task(
                        a_initiate_group_chat(
                            pattern=pattern,
                            messages=input_data.messages[-1].content + " PORTFOLIO DETAILS : " + json.dumps(input_data.state['investment_portfolio']) + ". INVESTMENT DATE : " + input_data.state['investment_date'],
                        )
                    )
                    while True:
                        try:
                            event = await asyncio.wait_for(event_queue.get(), timeout=1)
                            yield encoder.encode(event)
                        except asyncio.TimeoutError as e:
                            print(e)
                            # Check if the agent is done
                            if agent_task.done():
                                break
                except Exception as e:
                    print(e) 
                # agent_task = initiate_group_chat(
                #     pattern=pattern,
                #     messages=input_data.messages[-1].content + " PORTFOLIO DETAILS : " + json.dumps(input_data.state['investment_portfolio']) + ". INVESTMENT DATE : " + input_data.state['investment_date'],
                # )
            else:
                tool_log_id = str(uuid.uuid4())
                yield encoder.encode(
                    StateDeltaEvent(
                        type=EventType.STATE_DELTA,
                        delta=[
                            {
                                "op": "add",
                                "path": "/tool_logs/-",
                                "value": {
                                    "message": "Analyzing user query",
                                    "status": "processing",
                                    "id": tool_log_id,
                                },
                            }
                        ],
                    )
                )
                model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = model.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=input_data.messages
                )
                yield encoder.encode(
                    StateDeltaEvent(
                        type=EventType.STATE_DELTA,
                        delta=[
                            {
                                "op": "replace",
                                "path": "/tool_logs",
                                "value": []
                            }
                        ]
                    )
                )
                yield encoder.encode(
                    TextMessageStartEvent(
                        type=EventType.TEXT_MESSAGE_START,
                        message_id=message_id,
                        role="assistant",
                    )
                )
                # yield encoder.encode(
                #     StateDeltaEvent(
                #         type=EventType.STATE_DELTA,
                #         delta=[
                #             {
                #                 "op": "replace",
                #                 "path": "/tool_logs/0/status",
                #                 "value": "completed",
                #             }
                #         ],
                #     )
                # )
                # Only send content event if content is not empty
                if response.choices[0].message.content:
                    content = response.choices[0].message.content
                    # Split content into 100 parts
                    n_parts = 100
                    part_length = max(1, len(content) // n_parts)
                    parts = [content[i:i+part_length] for i in range(0, len(content), part_length)]
                    # If splitting results in more than 5 due to rounding, merge last parts
                    if len(parts) > n_parts:
                        parts = parts[:n_parts-1] + [''.join(parts[n_parts-1:])]
                    for part in parts:
                        yield encoder.encode(
                            TextMessageContentEvent(
                                type=EventType.TEXT_MESSAGE_CONTENT,
                                message_id=message_id,
                                delta=part,
                            )
                        )
                        await asyncio.sleep(0.05)
                else:
                    yield encoder.encode(
                        TextMessageContentEvent(
                            type=EventType.TEXT_MESSAGE_CONTENT,
                            message_id=message_id,
                            delta="Something went wrong! Please try again.",
                        )
                    )
                
                yield encoder.encode(
                    TextMessageEndEvent(
                        type=EventType.TEXT_MESSAGE_END,
                        message_id=message_id,
                    )
                )
               
            
            yield encoder.encode(
            StateDeltaEvent(
                type=EventType.STATE_DELTA,
                delta=[
                    {
                        "op": "replace",
                        "path": "/tool_logs",
                        "value": []
                    }
                ]
            )
            )
            if input_data.messages[-1].role == "user":
                if agent_task.result()[1].data['messages'][-1].role != 'user':
                    if agent_task.result()[1].data['messages'][-1].tool_calls:
                        # for tool_call in state['messages'][-1].tool_calls:
                        yield encoder.encode(
                            ToolCallStartEvent(
                                type=EventType.TOOL_CALL_START,
                                tool_call_id=agent_task.result()[1].data['messages'][-1].tool_calls[0].id,
                                toolCallName=agent_task.result()[1].data['messages'][-1]
                                .tool_calls[0]
                                .function.name,
                            )
                        )

                        yield encoder.encode(
                            ToolCallArgsEvent(
                                type=EventType.TOOL_CALL_ARGS,
                                tool_call_id=agent_task.result()[1].data['messages'][-1].tool_calls[0].id,
                                delta=agent_task.result()[1].data['messages'][-1]
                                .tool_calls[0]
                                .function.arguments,
                            )
                        )

                        yield encoder.encode(
                            ToolCallEndEvent(
                                type=EventType.TOOL_CALL_END,
                                tool_call_id=agent_task.result()[1].data['messages'][-1].tool_calls[0].id,
                            )
                        )
                else:
                    yield encoder.encode(
                        TextMessageStartEvent(
                            type=EventType.TEXT_MESSAGE_START,
                            message_id=message_id,
                            role="assistant",
                        )
                    )

                    # Only send content event if content is not empty
                    if agent_task.result()[0].chat_history[-1]['content']:
                        content = agent_task.result()[0].chat_history[-1]['content']
                        # Split content into 100 parts
                        n_parts = 100
                        part_length = max(1, len(content) // n_parts)
                        parts = [content[i:i+part_length] for i in range(0, len(content), part_length)]
                        # If splitting results in more than 5 due to rounding, merge last parts
                        if len(parts) > n_parts:
                            parts = parts[:n_parts-1] + [''.join(parts[n_parts-1:])]
                        for part in parts:
                            yield encoder.encode(
                                TextMessageContentEvent(
                                    type=EventType.TEXT_MESSAGE_CONTENT,
                                    message_id=message_id,
                                    delta=part,
                                )
                            )
                            await asyncio.sleep(0.05)
                    else:
                        yield encoder.encode(
                            TextMessageContentEvent(
                                type=EventType.TEXT_MESSAGE_CONTENT,
                                message_id=message_id,
                                delta="Something went wrong! Please try again.",
                            )
                        )
                    
                    yield encoder.encode(
                        TextMessageEndEvent(
                            type=EventType.TEXT_MESSAGE_END,
                            message_id=message_id,
                        )
                    )
            yield encoder.encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                )
            )

    except Exception as e:
        print(e)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )


if __name__ == "__main__":
    main()
