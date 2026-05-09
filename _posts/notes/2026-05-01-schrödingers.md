---
layout: post
title: "Schrödinger's Compiler"
summary: "All programs are equal, but some run faster"
date:  2026-05-01
keywords:
  [
    "compilers",
    "QBE",
    "equality saturation",
    "e-graphs",
    "reinforcement learning",
    "GNN",
    "neurosymbolic AI",
    "program optimization",
    "egg"
  ]
categories: notes
---

**Author:** Ibrahim El Kaddouri

**Repository:** <span class="tooltip">
<a href="https://github.com/IbrahimElk/X">
    <img src="/assets/images/2026-05-01/darksouls.jpg" style="width: 10%">
</a>
<span class="tooltip-text"> Still under construction :( </span>
</span>
<br>
<br>
<br>
## STILL IN PROGRESS
<br>
<br>
<br>
<br>

## Abstract

Modern compilers apply optimization passes in a fixed, hand-crafted order which is
brittle. A pass applied too early can permanently destroy structure that a later pass
could have exploited, the classic *phase ordering problem*. This project proposes a
neurosymbolic optimizer for [QBE](https://c9x.me/compile/), a lightweight compiler
backend, that sidesteps this problem entirely. The symbolic half uses *equality
saturation* (via e-graphs) to explore all valid program rewrites simultaneously and
non-destructively, the neural half uses a *graph neural network* (GNN) trained with
reinforcement learning to guide which rewrites are worth pursuing. Because every
transformation the neural component selects is applied by the symbolic engine,
correctness is guaranteed by construction, the RL agent never needs to check whether
its output is valid, only whether it is fast.

<br>

## Background

### QBE

QBE is a compiler backend written in roughly 14000 lines of C. It accepts an SSA-form
intermediate representation (IR) and emits optimized x86-64 or ARM64 machine code.

<span class="code-label">QBE IR</span>

```llvm
function w $add(w %a, w %b) {
@start
    %c =w add %a, %b
    ret %c
}
```

Each function is a set of basic blocks in SSA form, every variable is defined exactly
once and the structure forms a directed acyclic graph (DAG) of operations. This
structure is precisely what equality saturation is designed to work with.

### E-graphs and equality saturation

An *e-graph* is a data structure that compactly represents a potentially enormous set
of equivalent programs at once. The key insight is that rewrites are **additive**.
Instead of replacing `x * 2` with `x << 1`, an e-graph records that these two
expressions are equivalent, both remain available. This means the order in which you
apply rules does not matter, because no information is ever destroyed.

*Equality saturation* is the process of repeatedly applying rewrite rules to an e-graph
until no new equalities can be added (saturation). Once saturated, you extract the
cheapest program according to a cost model (e.g., instruction count, estimated
latency). The open-source library [`egg`](https://egraphs-good.github.io/) provides
a fast Rust implementation of this process, with Python bindings available.

A small taste of what rewrite rules might look like in `egg`'s rule syntax:

| Rewrite rule | Meaning |
|---|---|
| `(mul ?x 2)` &rarr; `(shl ?x 1)` | Strength reduction |
| `(add ?x 0)` &rarr; `?x` | Identity elimination |
| `(load (store ?ptr ?val) ?ptr)` &rarr; `?val` | Load-after-store forwarding |
| `(sub ?x ?x)` &rarr; `0` | Self-cancellation |
| `(neg (neg ?x))` &rarr; `?x` | Double negation |

These rules are *sound*, they preserve program semantics,
so any combination of them applied in any order yields a correct program.

### The phase ordering problem

Traditional compilers run passes in sequence. First, dead-code elimination, then
constant folding, then register allocation and so on. The problem is that applying
pass A might block pass B from firing. Equality saturation solves this by never
committing to a single form. All equivalent programs coexist in the e-graph
simultaneously and the best one is extracted at the end.

However, equality saturation has its own problem. The e-graph can grow
**exponentially** if all rules are applied indiscriminately. This is
where the neural component enters.

### Neurosymbolic guidance

Rather than applying all rewrite rules eagerly (which causes blowup) or in a fixed
order (which reintroduces the phase ordering problem), a GNN policy learns to *score*
which rules are most promising given the current e-graph state. The RL training loop
rewards the agent for producing programs that run faster than the baseline QBE output.

This idea has been validated in adjacent domains. *Omelette* (Singh & Hall, 2022)
showed that an RL system learning rewrite rule orderings for equality saturation can
improve solution quality by up to 100% over standard `egg`. *X-RLflow* (He et al.,
2023) applied a GNN-based RL agent to tensor computation graph optimization,
outperforming cost-based search by exploiting long-horizon reward structure.
*Aurora* (Barbulescu et al., 2024) applied the same principle to SQL query rewriting.
None of these targeted a general-purpose low-level IR like QBE.

<br>

## References

1. M Willsey, C Nandi, YR Wang, O Flatt, Z Tatlock, P Panchekha
   *egg: Fast and extensible equality saturation.*
   Proceedings of the ACM on Programming Languages 5 (POPL), 1-29

2. Singh, Z. (2022).
   *Deep reinforcement learning for equality saturation.*
   MPhil Thesis, University of Cambridge.

3. He, G., Singh, Z., & Yoneki, E. (2023).
   *MCTS-GEB: Monte Carlo Tree Search is a Good E-graph Builder.*
   Proceedings of EuroMLSys 2023.
   DOI: 10.1145/3578356.3592577
   arXiv:2303.04651

4. He, G., Parker, S., & Yoneki, E. (2023).
   *X-RLflow: Graph Reinforcement Learning for Neural Network Subgraphs Transformation.*
   arXiv:2304.14698

5. Barbulescu, G.-O., Wang, T., Singh, Z., & Yoneki, E. (2024).
   *Learned Graph Rewriting with Equality Saturation:
   A New Paradigm in Relational Query Rewrite and Beyond*
   arXiv:2407.12794.

6. Mankowitz, D. J., et al. (2023).
   *Faster sorting algorithms discovered using deep reinforcement learning.*
   Nature, 618, 257–263.

7. Jia, Z., et al. (2019).
   *TASO: Optimizing deep learning computation with automatic generation of graph substitutions.*
   Proceedings of SOSP 2019.

8. Zayed A., & Dubach, C. (2025).
   *DialEgg: Dialect-agnostic MLIR optimizer using equality saturation with Egglog.*
   Proceedings of CGO 2025.

9. Merckx, J., Besard, T., & De Sutter, B. (2026).
   *Equality Saturation for Optimizing High-Level Julia IR.*
   ACM Transactions on Architecture and Code Optimization,
   23(1), Article 24.
   DOI: 10.1145/3795883
