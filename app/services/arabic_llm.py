import torch
import asyncio
from typing import List, Dict, AsyncGenerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)


class ALLaMService:
    MODEL = "ALLaM-AI/ALLaM-7B-Instruct-preview"
    DEVICE = "cuda"

    def __init__(self):
        self._model: AutoModelForCausalLM = None
        self._tokenizer: AutoTokenizer = None
        self._lock = asyncio.Lock()

    # -------------------
    # Lifecycle
    # -------------------
    def load_model(self):
        """Load model and tokenizer into memory."""
        if self._model is None:
            print("Loading ALLaM model...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL,
                torch_dtype="auto",
                device_map=self.DEVICE
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL, use_fast=False
            )
            print("ALLaM model loaded successfully.")

    def unload_model(self):
        """Free VRAM + CPU RAM"""
        if self._model is not None:
            print("Unloading ALLaM model...")
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            torch.cuda.empty_cache()
            print("ALLaM model unloaded.")

    # -------------------
    # Conversation
    # -------------------
    async def converse(self, history: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Converse with the model. Async generator that yields tokens one by one."""
        self.load_model()  # Ensure model/tokenizer are loaded

        # Format conversation into a chat template
        inputs_text = self._tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self._tokenizer(
            inputs_text,
            return_tensors="pt",
            return_token_type_ids=False
        ).to(self.DEVICE)

        # Prepare streamer for live generation
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.6,
            streamer=streamer
        )

        # Ensure only one generation at a time
        async with self._lock:
            # Run generate() in a thread (non-blocking)
            asyncio.create_task(self._run_generate(generation_kwargs))

            # Async iterate tokens
            response_text = ""
            async for token in self._consume_stream(streamer):
                response_text += token
                yield token

    # -------------------
    # Utilities
    # -------------------
    async def _run_generate(self, generation_kwargs: dict):
        """Run blocking generate() in background thread."""
        await asyncio.to_thread(self._model.generate, **generation_kwargs)

    async def _consume_stream(self, streamer: TextIteratorStreamer):
        """Convert the sync streamer into async generator by polling until streamer finishes."""
        loop = asyncio.get_running_loop()

        def get_next():
            try:
                return next(streamer)
            except StopIteration:
                return None

        while True:
            token = await loop.run_in_executor(None, get_next)
            if token is None:
                break
            yield token
