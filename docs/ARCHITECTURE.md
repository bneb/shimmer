# Architecture Overview

Shimmer is an execution engine for tool-using LLM agents on Apple Silicon. It wraps `llama.cpp` with Metal GPU acceleration, adding streaming tool interception, edit validation, and KV cache compaction inside the inference loop.

The architecture consists of five core sub-systems.

---

## 1. Matrix-Matrix Prompt Lookup Decoding (Tree-PLD)

> **Note (2026-06-21): Disabled for agentic workloads.** n-gram drafts sourced from the system prompt produce token corruption in chat-template responses. Observed: missing `{` and `"` in JSON tool calls, token fusion (`"toolrg"` instead of `{"name": "rg"...}`). Re-enable after a draft-source filter is added that excludes prompt-sourced n-gram matches.

Traditional auto-regressive generation evaluates one token at a time. Shimmer implements a speculative decoding algorithm that uses local N-Gram matching against the KV cache history to construct branching draft trees. These draft trees are evaluated simultaneously on the Metal GPU in a single matrix-matrix operation.

```mermaid
graph TD
    classDef step fill:#1e293b,stroke:#64748b,stroke-width:2px,color:#f8fafc,rx:8px,ry:8px
    classDef eval fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc,rx:8px,ry:8px
    classDef decision fill:#334155,stroke:#c084fc,stroke-width:2px,color:#f8fafc,rx:12px,ry:12px
    classDef success fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#ecfdf5,rx:8px,ry:8px
    classDef fail fill:#7f1d1d,stroke:#f87171,stroke-width:2px,color:#fef2f2,rx:8px,ry:8px

    subgraph "CPU: N-Gram Speculation"
        A["N-Gram Cache Search"]:::step -->|Extract Matches| B("Construct Draft Tree"):::step
    end

    subgraph "GPU: Apple Metal Acceleration"
        B --> C["Metal Matrix-Matrix Evaluation"]:::eval
        C -->|Simultaneous Logit Verification| D{"Rollback & Correct"}:::decision
    end

    subgraph "Cache Commit Phase"
        D -->|Tokens Accepted| E["Commit Valid Sequences"]:::success
        D -->|Tokens Rejected| F["Prune Failed Futures"]:::fail
    end
```

---

## 2. Batched Multi-Agent Swarm Concurrency (MASC)

To bypass the memory bandwidth bottlenecks of single-batch inference, MASC groups $K$ distinct agents into a single forward pass. The context manager shares the underlying System Prompt KV cache across all $K$ sequences, while allocating discrete history tracking for individual agents.

```mermaid
graph LR
    classDef sys fill:#1e293b,stroke:#a855f7,stroke-width:2px,color:#f8fafc,rx:8px,ry:8px
    classDef ctx fill:#334155,stroke:#64748b,stroke-width:2px,color:#f8fafc,rx:8px,ry:8px
    classDef eval fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc,rx:8px,ry:8px
    classDef out fill:#065f46,stroke:#10b981,stroke-width:2px,color:#ecfdf5,rx:8px,ry:8px

    A["Shared System Prompt KV"]:::sys

    subgraph "Parallel Agent Contexts"
        A --> B["Agent 1 Context"]:::ctx
        A --> C["Agent 2 Context"]:::ctx
        A --> D["Agent 3 Context"]:::ctx
    end
    
    subgraph "Metal GPU"
        B --> E["Batched Inference Pass"]:::eval
        C --> E
        D --> E
    end

    subgraph "Logit Emission"
        E --> F["Sequence 1 Logits"]:::out
        E --> G["Sequence 2 Logits"]:::out
        E --> H["Sequence 3 Logits"]:::out
    end
```

---

## 3. Speculative Tool Execution

Standard agent loops block generation while tools are executed. Shimmer introduces an optimization that eagerly spawns OS subprocesses immediately upon detecting the opening tokens of a tool syntax.

```mermaid
sequenceDiagram
    participant E as Inference Engine
    participant I as Interceptor Thread
    participant S as OS Subprocess

    E->>I: Emit: <tool_call>
    Note over I,S: Speculative Dispatch Triggered
    I->>S: Eagerly Spawn Subprocess
    Note over S: Execution occurs during generation
    E->>I: Emit: {"name": "rg", ...}</tool_call>
    I->>E: Validate Syntax -> Halt Engine
    S-->>I: Yield stdout results
    I->>E: Inject Output
```

---

## 4. RoPE-Safe KV Compaction & ANE Compression

Shimmer uses page-based KV cache eviction with priority ordering. Tool output pages are evicted first, system prompt and recent history are preserved. RoPE positions are shifted to maintain coherence after compaction. 

If ANE Compression is enabled, ejected blocks are dense-vectorized on the Apple Neural Engine to maintain context constraints.

```mermaid
sequenceDiagram
    participant C as Context Manager
    participant M as KV Cache (Metal)
    participant A as Apple Neural Engine

    C->>C: Capacity Threshold Breached
    C->>C: Identify Oldest History Block
    C->>A: Dispatch Block for Compression
    A-->>C: 64-Token Vector Summary
    C->>M: Clear Old KV Range
    C->>M: Shift Remaining Blocks Left (RoPE update)
    C->>M: Prepend Vector Summary
```

---

## 5. Edit Validator Pipeline

Shimmer's 6 safety validators operate at turn boundaries (after `</edit>` or EOG), never mid-generation. All use append semantics: feedback is injected as a clean user turn rather than clearing KV cache and inserting tokens mid-stream. This prevents infinite loops caused by interrupting the model's thought chain.

```mermaid
graph TD
    classDef gen fill:#1e293b,stroke:#64748b,stroke-width:2px,color:#f8fafc,rx:8px,ry:8px
    classDef detect fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc,rx:8px,ry:8px
    classDef validate fill:#334155,stroke:#c084fc,stroke-width:2px,color:#f8fafc,rx:8px,ry:8px
    classDef inject fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#ecfdf5,rx:8px,ry:8px
    classDef reject fill:#7f1d1d,stroke:#f87171,stroke-width:2px,color:#fef2f2,rx:8px,ry:8px
    classDef commit fill:#065f46,stroke:#10b981,stroke-width:2px,color:#ecfdf5,rx:8px,ry:8px

    subgraph "Generation"
        A["Model generates tokens"]:::gen --> B{"</edit> detected?"}:::detect
        B -->|No| A
    end

    subgraph "Validation Chain (at turn boundary)"
        B -->|Yes| C["1. Blind Edit Blocker<br/>tool_calls == 0?"]:::validate
        C -->|Block| R1["Inject: 'use tools first'"]:::reject
        C -->|Pass| D["2. TDD Enforcement<br/>tests run?"]:::validate
        D -->|Block| R2["Inject: 'run tests first'"]:::reject
        D -->|Pass| E["3. Path Blocker<br/>file exists?"]:::validate
        E -->|Block| R3["Inject: 'file not found'"]:::reject
        E -->|Pass| F["4. Search Verifier<br/>exact match once?"]:::validate
        F -->|Block| R4["Inject: 'use cat to read file'"]:::reject
        F -->|Pass| G["5. Syntax Checker<br/>Python AST valid?"]:::validate
        G -->|Block| R5["Inject: syntax error details"]:::reject
        G -->|Pass| H["Commit edit ✓"]:::commit
    end

    R1 --> A
    R2 --> A
    R3 --> A
    R4 --> A
    R5 --> A
```

**Key design decision — append vs rollback:** Earlier versions used KV cache rollback (clear + inject mid-generation) which broke speculative decoding compatibility and caused infinite loops when the model was interrupted mid-thought. The current append-only approach means feedback appears as a clean user turn after the model finishes speaking. The cost is increased context usage (failed attempts stay in history), but this is managed by the compaction system.

## 6. Asynchronous Tool Interception

Token output is routed to a detached `std::sync::mpsc` channel. Regex compilation and pattern matching occur on a separate CPU thread, isolating parsing overhead from the primary Metal graph.

The interceptor detects three patterns in the token stream:
- **JSON tool calls:** ````json\n{"name": "...", "arguments": [...]}\n```` → dispatched to tool execution
- **XML edit tags:** `<edit file="...">`, `<search>...</search>`, `<replace>...</replace>`, `</edit>` → routed through the validator pipeline
- **Edit boundary:** `</edit>` closures set `edit_tag_closed` flag for the blind edit blocker

```mermaid
graph TD
    classDef thread fill:#1e293b,stroke:#8b5cf6,stroke-width:2px,color:#f8fafc,rx:8px,ry:8px
    classDef queue fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc,rx:8px,ry:8px
    classDef action fill:#991b1b,stroke:#ef4444,stroke-width:2px,color:#fef2f2,rx:8px,ry:8px

    subgraph "Core Threads"
        A["Main Thread (GPU Submission)"]:::thread -->|Token String| B("MPSC Channel"):::queue
        B --> C["Background Regex Thread"]:::thread
    end

    C -->|Pattern Match Found| D["Interrupt Signal Triggered"]:::action
    D -.->|Halt Generation| A
```

---

## 7. API Server

Shimmer is embedded with an `axum` driven high-performance web server that exposes an OpenAI-compatible `v1/chat/completions` API structure. 

```mermaid
graph LR
    classDef sys fill:#1e293b,stroke:#f59e0b,stroke-width:2px,color:#fffbeb,rx:8px,ry:8px
    classDef eval fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc,rx:8px,ry:8px
    classDef core fill:#334155,stroke:#10b981,stroke-width:2px,color:#ecfdf5,rx:8px,ry:8px

    A["API Client (e.g. Antigravity)"]:::sys -->|HTTP POST| B["Axum Web Server"]:::eval
    B --> C["JSON Payload Parser"]:::eval
    
    subgraph "Shimmer Daemon"
        C -->|Spawn Agent Thread| D["Shimmer Inference Graph"]:::core
    end
    
    D -.->|SSE Token Stream| A
```

## 8. Agentless Pipeline

New in June 2026 — an alternative evaluation mode that replaces model-driven investigation with a deterministic 3-phase pipeline. Activated via `--agentless` in the Python harness and `--no-tools` in the Rust binary.

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENTLESS PIPELINE                           │
├─────────────┬──────────────────┬───────────────────────────────┤
│ Phase 1     │ Phase 2          │ Phase 3                       │
│ Localization│ Repair           │ Validation                    │
├─────────────┼──────────────────┼───────────────────────────────┤
│ Python:     │ Python:          │ Python:                       │
│ extract     │ build agentless  │ extract_patch()               │
│ keywords    │ prompt with      │ verify_patch()                │
│ from issue  │ inlined files    │                               │
│     │       │      │           │                               │
│     ▼       │      ▼           │                               │
│ rg grep     │ shimmer          │                               │
│ ranking     │ --no-tools       │                               │
│     │       │ (single-shot)    │                               │
│     ▼       │      │           │                               │
│ ranked      │      ▼           │                               │
│ file list   │ XML edit block   │                               │
└─────────────┴──────────────────┴───────────────────────────────┘
```

**Key architectural difference:** In `--no-tools` mode, `AgentConfig.disable_tool_interceptor = true`. The `ToolInterceptor` gates all JSON tool-call detection behind `detect_json_tools`. Edit tag parsing (`<edit>`, `<search>`, `<replace>`, `</edit>`) continues to run — the model still outputs XML edit blocks. All 6 validators are bypassed (their enable flags are gated on `!args.no_tools` in `create_agent_config()`). The model generates straight through until EOG without interruption.

**File localization** uses regex keyword extraction (file paths, CamelCase, ALL_CAPS, snake_case) followed by `rg` grep to rank files by match count. Files are inlined into the prompt with smart truncation: ≤100 lines shown in full, 100-300 lines show first 30 + last 20, >300 lines show first 30 only. A 48K character budget (~12K tokens) prevents context overflow.
