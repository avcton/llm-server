from fastapi import FastAPI, Body
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from app.services.arabic_llm import ALLaMService


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    # Startup tasks
    allam_service.load_model()

    yield

    # Shutdown tasks
    allam_service.unload_model()
    
    
app = FastAPI(lifespan=lifespan)
allam_service = ALLaMService()


@app.post("/chat")
async def chat(user_message: str = Body(...)):
    """Handle chat requests by streaming responses from the ALLaMService."""
    async def token_stream():
        async for token in allam_service.converse(user_message):
            print("Yielding token:", token)
            yield token  # Stream tokens to client

    return StreamingResponse(token_stream(), media_type="text/plain")
