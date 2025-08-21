import torch
import asyncio
import threading
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
        # store conversation history (can also key by user/session ID)
        self._history: List[Dict[str, str]] = []
        self._lock = asyncio.Lock()

    # -------------------
    # Lifecycle
    # -------------------
    def load_model(self):
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
    async def converse(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        Converse with the model. Maintains context across turns.
        Async generator that yields tokens one by one.
        """
        self.load_model()  # Ensure model/tokenizer are loaded

        # Add user input to history
        self._history.append({"role": "user", "content": user_message})

        # Format conversation into a chat template
        inputs_text = self._tokenizer.apply_chat_template(
            self._history,
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
            # Launch generation in background
            asyncio.create_task(self._run_generate(generation_kwargs))

            # Async iterate tokens
            response_text = ""
            async for token in self._consume_stream(streamer):
                response_text += token
                yield token

            self._history.append({"role": "assistant", "content": response_text.strip()})

    # -------------------
    # Utilities
    # -------------------
    async def _run_generate(self, generation_kwargs: dict):
        """Run blocking generate() in background thread"""
        await asyncio.to_thread(self._model.generate, **generation_kwargs)
        
    async def _consume_stream(self, streamer: TextIteratorStreamer):
        """Async wrapper over the sync streamer iterator"""
        for token in streamer:
            yield token
            await asyncio.sleep(0)  # Let event loop breathe
    
    def reset_history(self):
        """Clear conversation memory"""
        self._history = []

    def get_history(self) -> List[Dict[str, str]]:
        return self._history
