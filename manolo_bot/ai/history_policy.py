"""History persistence policy for agent loop traces.

Decides which of the new messages produced by an agent ``ainvoke`` call
(the *tail* beyond what was sent) should be kept in conversation history.

Library-first use (no bot config needed)::

    from manolo_bot.ai.history_policy import FullTracePolicy
    from manolo_bot.ai.llmagent import LLMAgent

    agent = LLMAgent(..., history_policy=FullTracePolicy())

Use ``FinalOnlyPolicy`` (the default) to preserve the historical
text-only-final-message behavior, or ``FullTracePolicy`` to keep the full
tool-call trace.
"""

import abc

from langchain_core.messages import AIMessage, BaseMessage


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
        last = tail[-1]
        content = last.content
        if isinstance(content, list):
            # Flatten like LLMBot.postprocess_response does.
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
        elif isinstance(content, str):
            text = content
        else:
            text = str(content) if content is not None else ""
        # Fresh AIMessage => no tool_calls, no extra metadata.
        return [AIMessage(content=text)]


class FullTracePolicy(BaseHistoryPolicy):
    """Keep the full agent loop tail verbatim (tool calls + tool results).

    Library-first use (no bot config needed)::

        from manolo_bot.ai.history_policy import FullTracePolicy

        agent = LLMAgent(..., history_policy=FullTracePolicy())
    """

    def select(self, tail: list[BaseMessage]) -> list[BaseMessage]:
        return list(tail)
