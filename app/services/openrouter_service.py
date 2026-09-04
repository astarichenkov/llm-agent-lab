"""OpenRouter provider client for Day 5 (separate from DeepSeek).

Uses the OpenAI SDK pointed at OpenRouter's base URL with its own key.
Small and explicit; only used by Day 5.
"""
import logging
import time

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from app.config import Settings
from app.or_models import get_model
from app.schemas.day5 import OpenRouterRunRequest, OpenRouterRunResponse

logger = logging.getLogger("app.services.openrouter")


class OpenRouterError(Exception):
    """User-safe base error with an HTTP status hint."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class OpenRouterService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.openrouter_api_key or "missing-openrouter-key",
            base_url=settings.openrouter_base_url,
            timeout=settings.openrouter_timeout_seconds,
            default_headers={
                "X-Title": "DeepSeek Study Project",
                "HTTP-Referer": "https://openrouter.ai",
            },
        )

    async def run(self, request: OpenRouterRunRequest) -> OpenRouterRunResponse:
        if not self._settings.openrouter_api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY не задан. Добавьте ключ OpenRouter в .env / окружение.",
                503,
            )
        model_meta = get_model(request.model)
        if model_meta is None:
            raise OpenRouterError("Unknown model id.", 400)

        logger.info(
            "openrouter request started requested_model=%s message_length=%s "
            "max_tokens=%s temperature=%s stop_configured=%s",
            request.model,
            len(request.message),
            request.max_tokens,
            request.temperature,
            bool(request.stop_sequence),
        )
        started = time.perf_counter()
        params: dict = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.message}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.stop_sequence:
            params["stop"] = [request.stop_sequence]
        try:
            response = await self._client.chat.completions.create(**params)
        except AuthenticationError:
            logger.warning("OpenRouter auth error")
            raise OpenRouterError(
                "OpenRouter аутентификация не удалась. Проверьте OPENROUTER_API_KEY.", 401
            )
        except RateLimitError:
            logger.warning("OpenRouter rate limit")
            raise OpenRouterError("OpenRouter rate limit. Повторите позже.", 429)
        except APITimeoutError:
            logger.warning("OpenRouter timeout")
            raise OpenRouterError("OpenRouter превысил время ожидания.", 504)
        except APIConnectionError:
            logger.warning("OpenRouter connection error")
            raise OpenRouterError("Не удалось подключиться к OpenRouter.", 502)
        except APIError as exc:
            status = getattr(exc, "status_code", 502)
            logger.warning("OpenRouter API error status=%s", status)
            if status == 404:
                raise OpenRouterError(
                    "Выбранная модель недоступна через OpenRouter (модель не найдена).",
                    404,
                )
            raise OpenRouterError(
                f"OpenRouter вернул ошибку (HTTP {status}).", status
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("OpenRouter unexpected error")
            raise OpenRouterError("Непредвиденная ошибка OpenRouter.", 500) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        try:
            choice = response.choices[0]
            answer = choice.message.content
            finish_reason = choice.finish_reason
        except (AttributeError, IndexError, TypeError):
            raise OpenRouterError("OpenRouter вернул некорректный ответ.", 502)
        if not answer or not answer.strip():
            raise OpenRouterError("OpenRouter вернул пустой ответ.", 502)

        actual_model = getattr(response, "model", None) or request.model

        usage = None
        u = getattr(response, "usage", None)
        if u is not None:
            usage = {}
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                v = getattr(u, k, None)
                if v is not None:
                    usage[k] = v

        # cost: prefer provider-reported value if present in usage, else estimate
        cost = None
        cost_estimated = False
        if usage is not None and isinstance(u, object):
            provider_cost = getattr(u, "cost", None) if u is not None else None
            if provider_cost is not None:
                cost = float(provider_cost)
            else:
                pt = usage.get("prompt_tokens") or 0
                ct = usage.get("completion_tokens") or 0
                ip = model_meta.get("input_per_million")
                op = model_meta.get("output_per_million")
                if ip is not None and op is not None and (pt or ct):
                    cost = (pt * float(ip) + ct * float(op)) / 1_000_000.0
                    cost_estimated = True

        logger.info(
            "openrouter request completed requested_model=%s actual_model=%s "
            "finish_reason=%s elapsed_ms=%.1f usage=%s cost=%s",
            request.model,
            actual_model,
            finish_reason,
            elapsed_ms,
            usage,
            cost,
        )
        return OpenRouterRunResponse(
            answer=answer.strip(),
            requested_model=request.model,
            actual_model=actual_model,
            finish_reason=finish_reason,
            elapsed_ms=elapsed_ms,
            usage=usage,
            cost=cost,
            cost_estimated=cost_estimated,
            pricing={
                "input_per_million": model_meta.get("input_per_million"),
                "output_per_million": model_meta.get("output_per_million"),
                "context_length": model_meta.get("context_length"),
            },
            model_url=model_meta.get("url"),
        )
