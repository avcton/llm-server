from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from app.services.arabic_llm import ALLaMService


app = FastAPI()
allam_service = ALLaMService()


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    # Startup tasks
    allam_service.load_model()

    yield

    # Shutdown tasks
    allam_service.unload_model()


@app.post("/chat")
def chat(user_message: str):
    async def token_stream():
        async for token in allam_service.converse(user_message):
            yield token  # Stream tokens to client

    return StreamingResponse(token_stream(), media_type="text/plain")
