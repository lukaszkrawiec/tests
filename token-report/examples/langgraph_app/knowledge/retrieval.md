---
title: Retrieval and progressive disclosure
summary: When to inline reference material and when to fetch it on demand.
---

# Retrieval and progressive disclosure

Reference material can either sit in the system prompt, where every request pays for it,
or live behind a retrieval tool, where only the requests that need it pay. The decision
is an expected-value calculation: inline it when the probability a given request needs it
is high enough that the retrieval round trip costs more than the tokens saved.

Progressive disclosure is the middle path. Keep a compact index in the prompt — one line
per document, enough for the model to know what exists and decide whether to fetch it —
and keep the documents themselves behind a tool. The index is resident and grows linearly
with the number of documents, so the growth to watch is the index, not the corpus.

The failure mode is an index that grows richer over time. A one-line summary becomes
three lines, then a bulleted list of the document's sections, until the index costs more
than inlining a few of the documents would have. Measure the index separately from the
corpus so this is visible.
