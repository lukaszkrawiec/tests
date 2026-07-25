---
title: Tool schema design
summary: Why tool definitions are expensive and how to keep them small.
---

# Tool schema design

Every tool definition is resident: it is sent on every request for the whole run,
whether or not the tool is called. A tool with a long description and fifteen optional
parameters costs that on every turn of every conversation, forever.

Descriptions carry most of the weight. A parameter description that restates the
parameter name adds cost and no information. Enumerations are usually cheaper and
clearer than prose explaining which values are legal.

The strongest lever is the number of tools. Two tools that differ only in a mode flag
should usually be one tool with a mode parameter. Tools that are only relevant in one
phase of a graph should be bound to that node rather than to the whole graph, so other
nodes do not pay for them.
