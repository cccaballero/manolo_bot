---
name: Character Protection Feature
about: Add feature to prevent users from altering the bot's character
title: 'Add feature to prevent users from altering the bot''s character'
labels: enhancement
assignees: ''

---

## Problem

Currently, users in the Telegram chat can potentially alter the bot's character by giving it instructions that override the original character definition. This is a common issue with LLM-based bots where users can try to "jailbreak" or manipulate the bot's behavior through prompt engineering.

## Proposed Solution

Implement a feature that prevents users from altering the bot's character by:

1. Adding a prompt filter that detects and blocks attempts to change the bot's character or behavior
2. Implementing a system message that reinforces the bot's character and explicitly instructs it to ignore user attempts to change its behavior
3. Adding a configuration option to enable/disable this protection

## Implementation Details

1. Create a new configuration option `prevent_character_alteration` (default: true)
2. Add a function to detect potential character alteration attempts in user messages:
   * Look for phrases like "ignore previous instructions", "act as", "pretend to be", etc.
   * Check for attempts to redefine the bot's identity or behavior
   * Support detection in multiple languages (at minimum Spanish and English)
3. Add a system message that reinforces the bot's character and explicitly instructs it to ignore user attempts to change its behavior
4. When a potential alteration attempt is detected:
   * Either ignore the message
   * Or respond with a predefined message explaining that the bot cannot change its character
   * Log the attempt for monitoring

## Technical Considerations

* The detection system should be configurable to allow for different levels of strictness
* False positives should be minimized to avoid disrupting normal conversation
* The feature should work with all supported LLM backends (Google, OpenAI, Ollama)
* The detection system must support multiple languages, especially Spanish (the default preferred language) and English
* Consider using language-agnostic patterns where possible to detect character alteration attempts
* Provide configurable response templates in multiple languages

## Acceptance Criteria

- [ ] Bot consistently maintains its defined character even when users attempt to alter it
- [ ] Configuration option works as expected
- [ ] Detection system has minimal false positives
- [ ] Feature works correctly with messages in different languages
- [ ] Feature is well-documented in the README