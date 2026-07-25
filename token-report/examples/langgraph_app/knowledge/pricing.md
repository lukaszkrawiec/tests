---
title: Token pricing and cost model
summary: How input, output, and cached tokens are billed, and where cost actually accrues.
---

# Token pricing and cost model

Input tokens are billed for everything sent to the model on a given request: the system
prompt, every tool definition, the full conversation history, and the current user turn.
Output tokens are billed separately and typically at a higher rate.

The consequence that surprises people is that conversation history is re-billed on every
turn. A ten-turn conversation does not cost ten times the first turn; it costs closer to
the sum of a growing prefix, which is quadratic in the number of turns. This is why
long-running agent loops dominate a bill even when each individual message is small.

Cached tokens are billed at a reduced rate on a cache hit and a slightly increased rate
on the write that populates the cache. Caching is prefix-based: it works only for an
exact match from the start of the request up to a breakpoint. Any change near the
beginning of a prompt invalidates everything after it, which is why stable content
belongs first and volatile content last.
