"""claudo litellm wrapper — patches for DO Gradient AI compatibility + PERFORMANCE."""
import os, sys

# ── PERFORMANCE: Disable telemetry & verbose logging before any imports ──
os.environ["LITELLM_TELEMETRY"] = "False"
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["LITELLM_DONT_SHOW_FEEDBACK_BOX"] = "1"
os.environ["LITELLM_DROP_PARAMS"] = "True"

# Patch uvicorn to use asyncio instead of uvloop (uvloop broken on Python 3.14+)
try:
    import uvicorn.config
    uvicorn.config.LOOP_SETUPS["uvloop"] = "uvicorn.loops.asyncio:asyncio_setup"
except Exception:
    pass

# ── PERFORMANCE: Disable LiteLLM callbacks and spend tracking ──
try:
    import litellm
    litellm.telemetry = False
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm.service_callback = []
    litellm.callbacks = []
    litellm.input_callback = []
    litellm.set_verbose = False
    litellm.json_logs = False
    litellm.suppress_debug_info = True
    litellm.drop_params = True
    litellm.num_retries = 0
    # CRITICAL: Disable OpenAI Responses API — DO/DeepSeek doesn't support it
    # and it causes ResponseCompletedEvent validation errors + truncated output
    litellm.enable_responses_api = False
    try:
        litellm.use_responses_api = False
    except Exception:
        pass
except Exception:
    pass

# ── CRITICAL: Patch ALL Responses API routing paths ──
# LiteLLM routes to Responses API when it sees openai/ prefix + thinking params.
# This must be disabled at multiple levels since different code paths check it.
try:
    from litellm.utils import supports_responses_api as _orig_supports
    litellm.utils.supports_responses_api = lambda *a, **kw: False
except Exception:
    pass
try:
    from litellm import utils as _lu
    _lu.supports_responses_api = lambda *a, **kw: False
except Exception:
    pass

# Patch the Anthropic pass-through adapter to exclude params that DO/Anthropic
# rejects when forwarded via OpenAI format (e.g. context_management in interactive mode).
try:
    from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
        LiteLLMMessagesToCompletionTransformationHandler as _Handler,
    )

    _orig_prepare = _Handler._prepare_completion_kwargs

    def _strip_empty_text_blocks(messages):
        if not isinstance(messages, list):
            return messages
        cleaned = []
        for msg in messages:
            if not isinstance(msg, dict):
                cleaned.append(msg)
                continue
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text")
                        if text is None or (isinstance(text, str) and text.strip() == ""):
                            continue
                    new_content.append(block)
                if len(new_content) == 0:
                    continue
                msg = dict(msg)
                msg["content"] = new_content
            elif isinstance(content, str) and content.strip() == "":
                continue
            cleaned.append(msg)
        return cleaned

    def _patched_prepare(**kwargs):
        extra_kwargs = kwargs.get("extra_kwargs") or {}
        for drop_key in ("context_management",):
            extra_kwargs.pop(drop_key, None)
        kwargs["extra_kwargs"] = extra_kwargs
        if "messages" in kwargs:
            kwargs["messages"] = _strip_empty_text_blocks(kwargs.get("messages"))
        system = kwargs.get("system")
        if isinstance(system, list):
            new_system = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if text is None or (isinstance(text, str) and text.strip() == ""):
                        continue
                new_system.append(block)
            kwargs["system"] = new_system if new_system else None
        elif isinstance(system, str) and system.strip() == "":
            kwargs["system"] = None
        return _orig_prepare(**kwargs)

    _Handler._prepare_completion_kwargs = staticmethod(_patched_prepare)

    # Prevent routing to OpenAI Responses API for DO models
    _orig_route = _Handler._route_openai_thinking_to_responses_api_if_needed

    def _patched_route(completion_kwargs, *, thinking=None):
        model = completion_kwargs.get("model", "")
        if isinstance(model, str) and ("anthropic" in model or "deepseek" in model):
            return  # skip responses API routing for DO models
        return _orig_route(completion_kwargs, thinking=thinking)

    _Handler._route_openai_thinking_to_responses_api_if_needed = staticmethod(_patched_route)
except Exception as e:
    print(f"[claudo] WARNING: failed to patch adapter: {e}", file=sys.stderr)

# ── CRITICAL: Patch litellm_logging to handle ResponseCompletedEvent ──
try:
    from litellm.litellm_core_utils.litellm_logging import Logging as _Logging
    _orig_handle = _Logging._handle_anthropic_messages_response_logging
    
    def _patched_handle(self, result):
        # If Litellm streaming iterator passed ResponseCompletedEvent, extract the response
        if hasattr(result, "response") and type(result).__name__ == "ResponseCompletedEvent":
            result = getattr(result, "response")
        try:
            return _orig_handle(self, result)
        except Exception as e:
            # Drop errors so they don't break the async loop
            return result
            
    _Logging._handle_anthropic_messages_response_logging = _patched_handle
except Exception:
    pass

# ── CRITICAL: Force max_tokens override ──
# Claude Code sends max_tokens: 8192 which overrides config and limits output.
# We intercept acompletion and force it to 65536 (DeepSeek supports up to 384k).
try:
    import litellm.main
    _orig_acompletion = litellm.main.acompletion
    
    async def _patched_acompletion(*args, **kwargs):
        kwargs["max_tokens"] = 65536
        return await _orig_acompletion(*args, **kwargs)
        
    litellm.main.acompletion = _patched_acompletion
    litellm.acompletion = _patched_acompletion
except Exception as e:
    print(f"[claudo] WARNING: failed to patch max_tokens: {e}", file=sys.stderr)

# Silence non-blocking pydantic logging error in LiteLLM's logging_worker.
try:
    import logging
    logging.getLogger("litellm.proxy.proxy_logging").setLevel(logging.CRITICAL)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
except Exception:
    pass

from litellm.proxy.proxy_cli import run_server
run_server()
