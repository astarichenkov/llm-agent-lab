"""DeepSeek LLM client wrapper built on the official OpenAI Python SDK.

The class is deliberately isolated from FastAPI so it can be unit-tested
with a fully mocked HTTP layer and swapped for a fake via dependency
injection in the API tests.

Two public operations:

* ``chat(message)``          — single unrestricted request (POST /api/chat);
* ``compare(request)``       — the SAME prompt twice: once unrestricted and
  once with REAL API response controls (``response_format`` = JSON mode,
  editable ``json_structure`` instruction in messages, ``max_tokens`` and
  ``stop``). Exactly two provider calls per comparison.
"""
import json
import logging

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from app.config import Settings
from app.schemas.chat import ChatResponse
from app.schemas.compare import (
    FIXED_RESPONSE_FORMAT,
    CompareRequest,
    CompareResponse,
    CompareResult,
    ControlledSettings,
)

logger = logging.getLogger("app.services.deepseek")

from app.schemas.reasoning import ReasoningRequest, ReasoningResponse  # noqa: E402
from app.services.grading import grade_task  # noqa: E402
from app.schemas.temperature import TemperatureRequest, TemperatureResponse  # noqa: E402

# Application-level default output-token cap used by /api/chat and by the
# UNRESTRICTED side of a comparison. It keeps normal responses bounded and
# is independent of the student-configurable max_tokens of the controlled
# request (so the comparison isolates the effect of response controls).
DEFAULT_MAX_TOKENS = 1024


class DeepSeekError(Exception):
    """Base class for all DeepSeek service failures.

    * ``message`` -- user-friendly text, safe to expose to the browser
      (no secrets, no stack traces, no internal details).
    * ``status_code`` -- the HTTP status the API layer should return.
    """

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DeepSeekAuthenticationError(DeepSeekError):
    """The API key is missing or invalid."""

    def __init__(self) -> None:
        super().__init__(
            "DeepSeek authentication failed. Please check the API key.", 401
        )


class DeepSeekRateLimitError(DeepSeekError):
    """DeepSeek throttled the request (HTTP 429)."""

    def __init__(self) -> None:
        super().__init__(
            "DeepSeek rate limit exceeded. Please wait a moment and try again.", 429
        )


class DeepSeekTimeoutError(DeepSeekError):
    """The upstream request exceeded the configured timeout."""

    def __init__(self) -> None:
        super().__init__(
            "DeepSeek took too long to respond. Please try again.", 504
        )


class DeepSeekNetworkError(DeepSeekError):
    """The upstream host could not be reached (DNS/TCP/TLS failure)."""

    def __init__(self) -> None:
        super().__init__(
            "Could not reach the DeepSeek service. Please try again later.", 502
        )


class DeepSeekMalformedResponseError(DeepSeekError):
    """The provider returned something we could not parse."""

    def __init__(self) -> None:
        super().__init__(
            "DeepSeek returned an unexpected response. Please try again.", 502
        )


class DeepSeekOutputLimitError(DeepSeekError):
    """DeepSeek JSON Output returned EMPTY content because the generation
    token budget (max_tokens) was too small — NOT a malformed response.

    Verified against the real provider: with response_format=json_object a
    too-small max_tokens yields HTTP 200, choices_len=1 and EMPTY content
    with finish_reason="length".
    """

    def __init__(self) -> None:
        super().__init__(
            "Ответ не поместился в заданный лимит max_tokens. "
            "Увеличьте максимальную длину ответа.",
            502,
        )
        self.finish_reason = "length"


class DeepSeekService:
    """Thin wrapper around ``AsyncOpenAI`` configured for DeepSeek."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # A placeholder key keeps the client constructible even without
        # configuration; a real (missing/invalid) key surfaces as a
        # friendly 401 from the API layer instead of a client crash.
        self._client = AsyncOpenAI(
            api_key=settings.deepseek_api_key or "missing-api-key",
            base_url=settings.deepseek_base_url,
            timeout=settings.deepseek_timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def chat(self, message: str) -> ChatResponse:
        """Single unrestricted request (used by ``POST /api/chat``)."""
        result = await self._complete(
            [
                {"role": "system", "content": self._settings.system_prompt},
                {"role": "user", "content": message},
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return ChatResponse(answer=result.answer)

    async def compare(self, request: CompareRequest) -> CompareResponse:
        """Send the SAME prompt twice and compare the answers.

        Exactly two provider calls:

        1. ``unrestricted`` — system + original user prompt only, app default
           cap. NO response_format, NO custom max_tokens, NO stop, NO JSON
           structure / termination instruction.
        2. ``controlled``   — the SAME original user prompt plus a JSON
           control instruction, and REAL API parameters:

           * ``response_format={"type": "json_object"}`` (fixed);
           * ``max_tokens=<from UI>``;
           * ``stop=[<from UI>]`` only when the user provided a sequence
             (empty is omitted — no artificial stop marker by default).

           The control instruction is appended to the SAME final user message
           as the original prompt (kept verbatim). It explicitly mentions
           JSON, embeds the user's editable JSON structure and asks the model
           to fill it with a bounded list. No artificial END marker is
           injected, so the JSON keeps its natural ending.

        Failure policy (partial-failure handling):

        * if the UNRESTRICTED call fails -> raise (HTTP error; nothing to
          compare);
        * if the CONTROLLED call fails -> the unrestricted result is still
          returned with ``controlled=None``, a friendly ``controlled_error``
          and the requested ``settings`` so the UI can show what was
          requested.

        No retries; one comparison never spends more than two provider calls.
        """
        json_struct_size = len(json.dumps(request.json_structure, ensure_ascii=False))
        logger.info(
            "comparison started message_length=%s json_structure_present=%s "
            "json_structure_size=%s response_format=json_object max_tokens=%s "
            "stop_configured=%s",
            len(request.message),
            True,
            json_struct_size,
            request.max_tokens,
            bool(request.stop_sequence),
        )

        system = {"role": "system", "content": self._settings.system_prompt}
        user = {"role": "user", "content": request.message}

        # 1) Unrestricted request.
        logger.info("unrestricted call started")
        unrestricted = await self._complete(
            [system, user],
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        logger.info(
            "unrestricted call completed finish_reason=%s content_length=%s",
            unrestricted.finish_reason,
            len(unrestricted.answer),
        )

        # 2) Controlled request.
        # Empirically DeepSeek json mode is most reliable when the final user
        # message itself carries the JSON instruction (a bare prompt following
        # a separate instruction message is more likely to yield empty output).
        # The original user prompt is kept VERBATIM as the prefix of the final
        # message, then the control instruction is appended.
        instruction = self._build_control_instruction(request)
        final_controlled_message = request.message + "\n\n" + instruction
        controlled_messages = [system, {"role": "user", "content": final_controlled_message}]
        stop_list = [request.stop_sequence] if request.stop_sequence else None
        settings_echo = ControlledSettings(
            response_format=dict(FIXED_RESPONSE_FORMAT),
            max_tokens=request.max_tokens,
            stop=stop_list,
            json_structure=request.json_structure,
        )
        logger.info(
            "controlled call started response_format=json_object max_tokens=%s "
            "stop_configured=%s",
            request.max_tokens,
            bool(request.stop_sequence),
        )
        try:
            controlled = await self._complete(
                controlled_messages,
                response_format=FIXED_RESPONSE_FORMAT,
                max_tokens=request.max_tokens,
                stop=request.stop_sequence,
            )
        except DeepSeekOutputLimitError as exc:
            logger.warning(
                "controlled call failed output_limit exception_class=%s "
                "finish_reason=%s max_tokens=%s",
                type(exc).__name__,
                exc.finish_reason,
                request.max_tokens,
            )
            return CompareResponse(
                unrestricted=self._result(unrestricted),
                settings=settings_echo,
                controlled=None,
                controlled_error=exc.message,
                controlled_finish_reason=exc.finish_reason,
            )
        except DeepSeekError as exc:
            logger.warning(
                "controlled call failed exception_class=%s", type(exc).__name__
            )
            return CompareResponse(
                unrestricted=self._result(unrestricted),
                settings=settings_echo,
                controlled=None,
                controlled_error=exc.message,
            )

        logger.info(
            "controlled call completed finish_reason=%s content_length=%s",
            controlled.finish_reason,
            len(controlled.answer),
        )
        logger.info("comparison completed")
        return CompareResponse(
            unrestricted=self._result(unrestricted),
            settings=settings_echo,
            controlled=self._result(controlled, settings=settings_echo),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _result(
        self,
        result: CompareResult,
        settings: ControlledSettings | None = None,
    ) -> CompareResult:
        return CompareResult(
            answer=result.answer,
            finish_reason=result.finish_reason,
            settings=settings,
        )

    def _build_control_instruction(self, request: CompareRequest) -> str:
        """The controlled-request instruction (a real ``messages`` message).

        * explicitly mentions JSON (DeepSeek JSON Output requires the word
          "json" in the messages when response_format=json_object is used);
        * embeds the user's editable JSON structure verbatim;
        * asks the model to fill the structure with a BOUNDED list (max ~7
          items) so the JSON stays compact and json mode completes it;
        * does NOT inject an artificial END marker — the JSON keeps its
          natural ending (stop is optional and separate).

        The original user message is NOT altered.
        """
        structure_text = json.dumps(
            request.json_structure, ensure_ascii=False, indent=2
        )
        parts = [
            "Верни ответ только в формате JSON.",
            "Используй следующую структуру JSON:\n" + structure_text,
            (
                "Заполни структуру реальными данными по запросу пользователя. "
                "Укажи только основные ингредиенты — максимум 7 позиций. "
                "Ответ должен содержать только корректный JSON."
            ),
        ]
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Day 3 — reasoning strategies (one provider call per strategy request;
    # the generate/use two-stage workflow is driven as two separate requests)
    # ------------------------------------------------------------------
    async def reasoning(self, request: ReasoningRequest) -> ReasoningResponse:
        method = request.method
        logger.info(
            "reasoning %s started task_length=%s max_tokens=%s stop_configured=%s",
            method,
            len(request.task),
            request.max_tokens,
            bool(request.stop_sequence),
        )
        try:
            if method == "direct":
                result = await self._reason_direct(request)
            elif method == "step_by_step":
                result = await self._reason_step_by_step(request)
            elif method == "generate_prompt":
                result = await self._reason_generate_prompt(request)
            elif method == "use_prompt":
                result = await self._reason_use_prompt(request)
            elif method == "experts":
                result = await self._reason_experts(request)
            else:  # pragma: no cover - guarded by Literal schema
                raise ValueError(f"unknown method: {method}")
        except DeepSeekError:
            logger.warning("reasoning %s failed", method)
            raise
        logger.info(
            "reasoning %s completed finish_reason=%s content_length=%s",
            method,
            result.finish_reason,
            len(result.solution or result.generated_prompt or ""),
        )
        return result

    async def _call_plain(self, content: str, max_tokens: int, stop: str | None):
        """One plain-text (no json mode) provider call returning provider fields."""
        return await self._call(
            [{"role": "user", "content": content}],
            max_tokens=max_tokens,
            stop=stop,
        )

    async def _reason_direct(self, request: ReasoningRequest) -> ReasoningResponse:
        prompt_sent = request.task.strip()
        answer, finish, usage = await self._call_plain(
            prompt_sent, request.max_tokens, request.stop_sequence
        )
        return ReasoningResponse(
            method="direct",
            kind="solution",
            prompt_sent=prompt_sent,
            solution=answer,
            finish_reason=finish,
            usage=usage,
            status=grade_task(request.task, answer),
        )

    async def _reason_step_by_step(self, request: ReasoningRequest) -> ReasoningResponse:
        task = request.task.strip()
        prompt_sent = (
            "Решай задачу пошагово.\n"
            "Последовательно проверь условия задачи.\n"
            "В конце отдельно укажи итоговый ответ.\n\n"
            f"Задача:\n{task}"
        )
        answer, finish, usage = await self._call_plain(
            prompt_sent, request.max_tokens, request.stop_sequence
        )
        return ReasoningResponse(
            method="step_by_step",
            kind="solution",
            prompt_sent=prompt_sent,
            solution=answer,
            finish_reason=finish,
            usage=usage,
            status=grade_task(request.task, answer),
        )

    async def _reason_generate_prompt(self, request: ReasoningRequest) -> ReasoningResponse:
        task = request.task.strip()
        prompt_sent = (
            "Составь эффективный промпт для решения следующей логической задачи.\n\n"
            "Промпт должен помочь другой модели:\n"
            "- внимательно проанализировать все условия;\n"
            "- получить максимально точное решение;\n"
            "- проверить решение на соответствие каждому условию;\n"
            "- в конце дать однозначный итоговый ответ.\n\n"
            "Не решай саму задачу.\n"
            "Верни только готовый промпт, который следует передать другой модели.\n\n"
            "Важно: готовый промпт должен быть самодостаточным и содержать текст "
            "исходной задачи полностью.\n\n"
            f"Исходная задача:\n\n{task}"
        )
        answer, finish, usage = await self._call_plain(
            prompt_sent, request.max_tokens, request.stop_sequence
        )
        return ReasoningResponse(
            method="generate_prompt",
            kind="generated_prompt",
            prompt_sent=prompt_sent,
            generated_prompt=answer.strip(),
            finish_reason=finish,
            usage=usage,
        )

    async def _reason_use_prompt(self, request: ReasoningRequest) -> ReasoningResponse:
        task = request.task.strip()
        generated = (request.generated_prompt or "").strip()
        final = self._ensure_task_included(generated, task)
        answer, finish, usage = await self._call_plain(
            final, request.max_tokens, request.stop_sequence
        )
        return ReasoningResponse(
            method="use_prompt",
            kind="solution",
            prompt_sent=final,
            solution=answer,
            finish_reason=finish,
            usage=usage,
            status=grade_task(request.task, answer),
        )

    @staticmethod
    def _ensure_task_included(prompt: str, task: str) -> str:
        """The final prompt must contain the original task. If it does not,
        append the task so the model is never asked to solve an unspecified
        problem."""
        normalized = " ".join(prompt.split())
        normalized_task = " ".join(task.split())
        if normalized_task and normalized_task in normalized:
            return prompt
        return prompt + "\n\nЗадача:\n" + task

    async def _reason_experts(self, request: ReasoningRequest) -> ReasoningResponse:
        task = request.task.strip()
        prompt_sent = (
            "Реши следующую задачу с помощью группы экспертов.\n\n"
            "Эксперт 1 — Аналитик.\n"
            "Независимо проанализируй условия задачи и предложи решение.\n\n"
            "Эксперт 2 — Инженер.\n"
            "Реши задачу систематически. Проверь допустимые варианты и получи "
            "решение независимо от аналитика.\n\n"
            "Эксперт 3 — Критик.\n"
            "Проверь рассуждения и решения. Найди возможные противоречия с "
            "исходными условиями и укажи ошибки, если они есть.\n\n"
            "После мнений всех трёх экспертов сформируй общий итоговый ответ.\n\n"
            "Структура ответа:\n"
            "АНАЛИТИК:\n"
            "ИНЖЕНЕР:\n"
            "КРИТИК:\n"
            "ИТОГОВОЕ РЕШЕНИЕ:\n\n"
            f"Задача:\n\n{task}"
        )
        answer, finish, usage = await self._call_plain(
            prompt_sent, request.max_tokens, request.stop_sequence
        )
        return ReasoningResponse(
            method="experts",
            kind="solution",
            prompt_sent=prompt_sent,
            solution=answer,
            finish_reason=finish,
            usage=usage,
            status=grade_task(request.task, answer),
        )

    async def complete_with_temperature(
        self, request: TemperatureRequest
    ) -> TemperatureResponse:
        """Day 4: send the message with the given REAL temperature value.
        Thinking mode is disabled so temperature actually affects generation.
        """
        logger.info(
            "temperature request started temperature=%s thinking_mode=disabled "
            "message_length=%s max_tokens=%s stop_configured=%s",
            request.temperature,
            len(request.message),
            request.max_tokens,
            bool(request.stop_sequence),
        )
        content, finish, usage = await self._call(
            [
                {"role": "system", "content": self._settings.system_prompt},
                {"role": "user", "content": request.message},
            ],
            max_tokens=request.max_tokens,
            stop=request.stop_sequence,
            temperature=request.temperature,
            extra_body={"thinking": {"type": "disabled"}},
        )
        logger.info(
            "temperature request completed finish_reason=%s content_length=%s usage=%s",
            finish,
            len(content),
            usage,
        )
        return TemperatureResponse(
            answer=content,
            finish_reason=finish,
            usage=usage,
            applied_parameters={
                "model": self._settings.deepseek_model,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stop": [request.stop_sequence] if request.stop_sequence else None,
                "mode": "non-thinking",
            },
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stop: str | None = None,
        thinking: bool | None = None,
    ) -> tuple[str, str | None, dict | None]:
        """Generic chat completion used by the persistent ``Agent`` layer.

        Unlike :meth:`chat` this method does NOT inject the application system
        prompt and does NOT build the message list: the caller (an ``Agent``)
        owns the full conversation. It simply forwards ``messages`` to the
        provider and returns ``(content, finish_reason, usage)``.

        ``thinking`` is the provider-neutral intent from ``AgentConfig``:

        * ``False`` -> reasoning is explicitly DISABLED through DeepSeek's
          official Non-Thinking mechanism (``extra_body={"thinking":
          {"type": "disabled"}}``), so it does not spend the output budget;
        * ``True``  -> leave the provider default untouched;
        * ``None``  -> send no override at all (legacy/neutral behaviour, so
          callers outside the Agent layer keep their previous requests).
        """
        extra_body = None
        if thinking is False:
            extra_body = {"thinking": {"type": "disabled"}}
        content, finish_reason, usage = await self._call(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            extra_body=extra_body,
        )
        # Safe diagnostic trace for the actual provider call (no secrets, no
        # user content). Correlates the Agent config with the RAW provider
        # result so a mis-mapped finish_reason can be spotted immediately.
        logger.info(
            "agent generate completed model=%s max_tokens=%s thinking=%s "
            "extra_body=%s messages=%s finish_reason=%s content_length=%s usage=%s",
            model or self._settings.deepseek_model,
            max_tokens,
            thinking,
            extra_body,
            len(messages),
            finish_reason,
            len(content),
            usage,
        )
        return content, finish_reason, usage

    async def _call(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict | None = None,
        max_tokens: int | None = None,
        stop: str | None = None,
        temperature: float = 0.7,
        extra_body: dict | None = None,
        model: str | None = None,
    ):
        """Run one Chat Completions call and return raw provider fields.

        Returns ``(content, finish_reason, usage)``. ``usage`` is a small
        dict (prompt/completion/total tokens) when the provider returned it,
        otherwise ``None``. Raises the same classified ``DeepSeekError``
        subclasses as ``_complete`` on provider failures / malformed /
        empty-output responses.

        ``extra_body`` (optional) is forwarded verbatim to the provider. It is
        used by Day 4 and by the Agent layer (Day 6 / Day 7) to turn DeepSeek
        Non-Thinking mode on. Day 2 / Day 3 pass ``None``.
        """
        params: dict = {
            "model": model or self._settings.deepseek_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            params["response_format"] = response_format
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if stop:
            params["stop"] = [stop]
        if extra_body:
            params["extra_body"] = extra_body

        try:
            response = await self._client.chat.completions.create(**params)
        except AuthenticationError as exc:
            logger.warning("DeepSeek auth error (type=%s)", type(exc).__name__)
            raise DeepSeekAuthenticationError() from exc
        except RateLimitError as exc:
            logger.warning("DeepSeek rate limit (type=%s)", type(exc).__name__)
            raise DeepSeekRateLimitError() from exc
        except APITimeoutError as exc:
            logger.warning("DeepSeek timeout (type=%s)", type(exc).__name__)
            raise DeepSeekTimeoutError() from exc
        except APIConnectionError as exc:
            logger.warning("DeepSeek connection error (type=%s)", type(exc).__name__)
            raise DeepSeekNetworkError() from exc
        except APIError as exc:
            logger.warning(
                "DeepSeek API error (type=%s, status=%s)",
                type(exc).__name__,
                getattr(exc, "status_code", "?"),
            )
            raise DeepSeekError(
                "The DeepSeek API reported an error. Please try again.", 502
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive catch-all
            logger.exception(
                "Unexpected DeepSeek client error (type=%s)", type(exc).__name__
            )
            raise DeepSeekError(
                "An unexpected error occurred while contacting DeepSeek.", 500
            ) from exc

        # Validate the shape of the provider response before using it.
        try:
            choice = response.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason
        except (AttributeError, IndexError, TypeError) as exc:
            try:
                choices_len = len(response.choices)
            except Exception:
                choices_len = "?"
            try:
                first_message = response.choices[0].message
                message_present = first_message is not None
            except Exception:
                first_message = None
                message_present = "?"
            try:
                first_finish = response.choices[0].finish_reason
            except Exception:
                first_finish = "?"
            logger.warning(
                "Malformed DeepSeek response structure (type=%s, choices_len=%s, "
                "first_choice_has_message=%s, first_finish_reason=%s) - "
                "choices missing/empty or message absent",
                type(response).__name__,
                choices_len,
                message_present,
                first_finish,
            )
            raise DeepSeekMalformedResponseError() from exc

        if content is None or not content.strip():
            stop_cfg = params.get("stop")
            stop_len = len(stop_cfg[0]) if stop_cfg else 0
            logger.warning(
                "DeepSeek returned empty content (content_is_none=%s, "
                "content_length=0, finish_reason=%r, response_format=%s, "
                "max_tokens=%s, stop_configured=%s, stop_length=%s, choices_len=%s)",
                content is None,
                finish_reason,
                params.get("response_format"),
                params.get("max_tokens"),
                bool(stop_cfg),
                stop_len,
                len(response.choices),
            )
            if finish_reason == "length":
                logger.warning("Classified as output-token-limit (finish_reason=length)")
                raise DeepSeekOutputLimitError() from None
            raise DeepSeekMalformedResponseError() from None

        return content.strip(), finish_reason, self._extract_usage(response)

    @staticmethod
    def _extract_usage(response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        out = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            val = getattr(usage, key, None)
            if val is not None:
                out[key] = val
        return out or None

    async def _complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict | None = None,
        max_tokens: int | None = None,
        stop: str | None = None,
    ) -> CompareResult:
        """Run one Chat Completions call and return a ``CompareResult``."""
        content, finish_reason, _usage = await self._call(
            messages,
            response_format=response_format,
            max_tokens=max_tokens,
            stop=stop,
        )
        return CompareResult(answer=content, finish_reason=finish_reason)
