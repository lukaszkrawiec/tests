---
title: Context window management
summary: Budgeting a finite window across prompt, history, retrieval, and output.
---

# Context window management

The context window is shared by everything: system prompt, tool definitions, retrieved
documents, conversation history, and the space reserved for the response. A budget that
accounts for the first four and forgets the fifth fails at the worst moment, when a long
run finally needs a long answer.

Reserve output space explicitly and treat it as spent. What remains is the working
budget, and it should be allocated deliberately — a fixed ceiling for retrieval, a fixed
ceiling for history, and compaction that triggers on the history ceiling rather than on
total window pressure. Triggering on total pressure makes behaviour depend on how much
was retrieved, which makes failures irreproducible.

Resident content is the floor under all of this. Every token of system prompt and tool
schema is a token unavailable to retrieval and history for the entire run, which is why
resident growth deserves a tighter budget than anything else.
