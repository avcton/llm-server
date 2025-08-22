import json
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from app.services.arabic_llm import ALLaMService
from app.validators.api_request import ChatCompletionRequest


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    # Startup tasks
    allam_service.load_model()

    yield

    # Shutdown tasks
    allam_service.unload_model()
    
    
app = FastAPI(lifespan=lifespan)
allam_service = ALLaMService()


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    OpenAI-compatible Chat Completions endpoint.
    Supports both streaming and non-streaming modes.
    """
    # STREAMING MODE
    if request.stream:
        async def event_stream():
            try:
                async for token in allam_service.converse(request.messages):
                    # OpenAI-style chunk
                    chunk = {
                        "id": "chatcmpl-12345",   # you can generate real UUIDs if you like
                        "object": "chat.completion.chunk",
                        "created": 1690000000,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                # End of stream
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                print(f"Error during streaming: {e}")
                err = {"error": {"message": str(e)}}
                yield f"data: {json.dumps(err)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # NON-STREAMING MODE
    else:
        response_text = ""
        async for token in allam_service.converse(request.messages):
            response_text += token

        completion = {
            "id": "chatcmpl-67890",
            "object": "chat.completion",
            "created": 1690000000,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text.strip(),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": None,   # you can compute if needed
                "completion_tokens": None,
                "total_tokens": None,
            },
        }
        return completion
