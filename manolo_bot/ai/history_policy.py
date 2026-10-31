"""History persistence policy for agent loop traces.

Decides which of the new messages produced by an agent ``ainvoke`` call
(the *tail* beyond what was sent) should be kept in conversation history.

Library-first use (no bot config needed)::

    from manolo_bot.ai.history_policy import FullTracePolicy
    from manolo_bot.ai.llmagent import LLMAgent

    agent = LLMAgent(..., history_policy=FullTracePolicy())

Use ``FinalOnlyPolicy`` (the default) to preserve the historical
text-only-final-message behavior, ``FullTracePolicy`` to keep the full
tool-call trace, or ``CompactTracePolicy`` for a bounded middle ground: the
verbatim trace while it is small, and a deterministic one-line-per-tool-call
ledger plus the final answer once it grows past a threshold.
"""

import abc
import json

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

#: Default tail-size threshold (in content chars) above which
#: :class:`CompactTracePolicy` switches from verbatim passthrough to a ledger.
DEFAULT_MAX_TAIL_CHARS = 3000
#: Default cap (in chars) for one rendered tool output in the ledger.
DEFAULT_PER_TOOL_CHARS = 500
#: Fixed cap (in chars) for rendered tool-call arguments in a ledger line.
#: Arguments are rarely the reason a tail is heavy, so this is not a knob
#: (the constructor keeps the two knobs the issue specifies).
LEDGER_ARGS_CHARS = 120


def flatten_content(content) -> str:
    """Flatten provider content blocks to plain text (``FinalOnly`` semantics).

    Kept module-level so ``FinalOnlyPolicy`` and ``CompactTracePolicy`` produce
    byte-identical final messages.
    """
    if isinstance(content, list):
        text = ""
        for i, content_item in enumerate(content):
            if isinstance(content_item, str):
                text += content_item
            elif isinstance(content_item, dict):
                text += str(content_item.get("text", ""))
            else:
                getter = getattr(content_item, "get", None)
                text += str(getter("text", "") if callable(getter) else "")
            if i + 1 != len(content):
                text += "\n\n"
        return text
    if isinstance(content, str):
        return content
    return str(content) if content is not None else ""


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to at most ``limit`` chars, marking the cut with ``...``."""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _render_output_block(content_item) -> str:
    """Render one tool-output content block, replacing binaries with placeholders."""
    if isinstance(content_item, str):
        return content_item
    if isinstance(content_item, dict):
        text = content_item.get("text")
        if isinstance(text, str):
            return text
        block_type = str(content_item.get("type", ""))
        if "image" in block_type:
            return "[image output omitted]"
        if "audio" in block_type or block_type in {"media", "input_audio"}:
            return "[audio output omitted]"
        return f"[{block_type or 'non-text'} output omitted]"
    getter = getattr(content_item, "get", None)
    if callable(getter):
        return str(getter("text", ""))
    return ""


def _render_tool_output(content) -> str:
    """Render a ``ToolMessage`` content value as plain text for the ledger."""
    if isinstance(content, list):
        return "\n".join(_render_output_block(item) for item in content)
    return flatten_content(content)


class BaseHistoryPolicy(abc.ABC):
    """Select which tail messages to persist in conversation history."""

    @abc.abstractmethod
    def select(self, tail: list[BaseMessage]) -> list[BaseMessage]:
        """Return the subset of ``tail`` to keep, in order.

        :param tail: New messages from the agent loop (``full[sent_len:]``).
        :return: Messages to persist/return (last one is returned to caller).
        """
        raise NotImplementedError


class FinalOnlyPolicy(BaseHistoryPolicy):
    """Keep only the final message as a text-only ``AIMessage``.

    Preserves the historical behavior where the tool-call trace is
    discarded and only the final text is persisted (via
    ``LLMBot.postprocess_response``).

    Library-first use (no bot config needed)::

        from manolo_bot.ai.history_policy import FinalOnlyPolicy

        agent = LLMAgent(..., history_policy=FinalOnlyPolicy())
    """

    def select(self, tail: list[BaseMessage]) -> list[BaseMessage]:
        if not tail:
            return []
        # Fresh AIMessage => no tool_calls, no extra metadata.
        return [AIMessage(content=flatten_content(tail[-1].content))]


class FullTracePolicy(BaseHistoryPolicy):
    """Keep the full agent loop tail verbatim (tool calls + tool results).

    Library-first use (no bot config needed)::

        from manolo_bot.ai.history_policy import FullTracePolicy

        agent = LLMAgent(..., history_policy=FullTracePolicy())
    """

    def select(self, tail: list[BaseMessage]) -> list[BaseMessage]:
        return list(tail)


class CompactTracePolicy(BaseHistoryPolicy):
    """Keep a small tail verbatim, otherwise a deterministic tool ledger + final.

    The tail size proxy is ``sum(len(str(content)))`` over the tail messages
    (provider-independent, no tokenizer). At or below ``max_tail_chars`` the
    policy behaves like :class:`FullTracePolicy`. Above it, the result is
    ``[ledger, final]``:

    - ``ledger``: one plain ``AIMessage`` with one line per tool call, in call
      order -- ``name(truncated args) -> outcome, N chars``. Outputs are capped
      at ``per_tool_chars`` with an ellipsis, errors/empties/missing results
      are kept as signals, and multimodal outputs become placeholders. The
      ledger itself is capped at ``max_tail_chars``; overflow is summarised as
      ``... (N more tool calls omitted)``.
    - ``final``: the same text-only ``AIMessage`` :class:`FinalOnlyPolicy`
      would produce (byte-identical flattening).

    Ledger lines never carry ``tool_calls``, so persisted history stays
    pair-atomic by construction: :func:`~manolo_bot.storage.messages.base.expand_tool_block`
    sees singletons. ``select()`` stays synchronous, so staging is unchanged.

    Library-first use (no bot config needed)::

        from manolo_bot.ai.history_policy import CompactTracePolicy

        agent = LLMAgent(..., history_policy=CompactTracePolicy(max_tail_chars=3000))
    """

    def __init__(
        self,
        max_tail_chars: int = DEFAULT_MAX_TAIL_CHARS,
        per_tool_chars: int = DEFAULT_PER_TOOL_CHARS,
    ) -> None:
        if max_tail_chars < 0:
            raise ValueError("max_tail_chars must be >= 0")
        if per_tool_chars < 4:
            raise ValueError("per_tool_chars must be >= 4")
        self.max_tail_chars = max_tail_chars
        self.per_tool_chars = per_tool_chars

    def select(self, tail: list[BaseMessage]) -> list[BaseMessage]:
        if not tail:
            return []
        if self._tail_chars(tail) <= self.max_tail_chars:
            return list(tail)
        final = AIMessage(content=flatten_content(tail[-1].content))
        ledger = self._build_ledger(tail)
        if not ledger:
            # Nothing to ledger (e.g. one huge answer): keeping an empty
            # AIMessage would only add provider noise.
            return [final]
        return [AIMessage(content=ledger), final]

    @staticmethod
    def _tail_chars(tail: list[BaseMessage]) -> int:
        return sum(len(str(message.content or "")) for message in tail)

    def _build_ledger(self, tail: list[BaseMessage]) -> str:
        outputs: dict[str, list[ToolMessage]] = {}
        for message in tail:
            if isinstance(message, ToolMessage):
                outputs.setdefault(message.tool_call_id, []).append(message)

        lines = []
        for message in tail:
            if not isinstance(message, AIMessage):
                continue
            for call in message.tool_calls or []:
                results = outputs.get(call.get("id"), [])
                lines.append(self._format_call(call, results))
        return self._cap_ledger(lines)

    def _format_call(self, call: dict, results: list[ToolMessage]) -> str:
        name = call.get("name") or "?"
        args = _truncate(
            json.dumps(call.get("args") or {}, ensure_ascii=False, sort_keys=True, default=str),
            LEDGER_ARGS_CHARS,
        )
        if not results:
            return f"{name}({args}) -> (no result)"

        output = "\n".join(_render_tool_output(result.content) for result in results)
        chars = len(output)
        # One line per call: collapse newlines/runs of spaces from real outputs.
        flat = " ".join(output.split())
        if not flat:
            outcome = "(empty)"
        else:
            outcome = _truncate(flat, self.per_tool_chars)
        if any(getattr(result, "status", None) == "error" for result in results):
            outcome = f"[error] {outcome}"
        return f"{name}({args}) -> {outcome}, {chars} chars"

    def _cap_ledger(self, lines: list[str]) -> str:
        """Join ledger lines, bounding the total to ``max_tail_chars``."""
        if not lines:
            return ""
        text = "\n".join(lines)
        if len(text) <= self.max_tail_chars:
            return text

        kept: list[str] = []
        used = 0
        for line in lines:
            cost = len(line) + (1 if kept else 0)
            if used + cost > self.max_tail_chars:
                break
            kept.append(line)
            used += cost

        omitted = len(lines) - len(kept)
        plural = "s" if omitted != 1 else ""
        marker = f"... ({omitted} more tool call{plural} omitted)"
        while kept and used + 1 + len(marker) > self.max_tail_chars:
            used -= len(kept.pop()) + (1 if kept else 0)
        return "\n".join([*kept, marker]) if kept else marker
