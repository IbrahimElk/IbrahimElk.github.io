---
layout: post
title: "Graph Surgeon"
date:  2026-01-10
summary: "What if we made transformers faster by doing less stupid stuff..."
keywords:
  [
    "tvm",
    "onnx",
    "compiler",
    "relax",
    "cuda",
  ]
categories: notes
---

**Author:** Ibrahim El Kaddouri

**Repository:** <span class="tooltip">
<a href="https://github.com/IbrahimElk/GraphSurgeon">
    <img src="/assets/images/2026-01-01/rabbit.jpg" style="width: 10%">
</a>
<span class="tooltip-text"> The repository is still private, under construction </span>
</span>

<script src="/assets/ts/interactive.js"></script>

## Abstract

This project presents **Graph Surgeon**, a custom graph-level compiler optimisation
pipeline for transformer-based language models, built on top of Apache TVM's Relax
<span class="term">intermediate representation. <span class="term-def">
An Intermediate Representation (IR) is like a translation layer. When you compile
C++ to machine code, the compiler first converts your code into an intermediate form
that is easier to analyse and simplify, but not yet CPU instructions.
TVM has two IRs: Relax (graph-level) and TensorIR (loop-level). More below.</span></span>

The goal of this project is to see whether targeted graph-level transformations,
specifically QKV attention-head fusion applied to
<span class="term">ONNX<span class="term-def">Open Neural Network Exchange
is a standard file format for neural networks. A universal format that different
frameworks (PyTorch, TensorFlow, etc.) can all export to and import from.</span></span>
computation graphs, can produce measurable and explainable performance improvements.

This project demonstrates that QKV fusion delivers an **11.8% speedup on CPU**
for DistilBERT. On GPU, the same pass has no effect or a slight regression.
<span class="term">DLight<span class="term-def">DLight is TVM's
rule-based TIR scheduling system, a set of hand-crafted rules for how to lay out
loops, tiles and threads on the GPU. "Rule-based" means it applies fixed strategies
rather than searching for the best one.</span></span>
rule-based TIR scheduling proved either neutral or harmful across all tested models.

<div class="callout tip-box">
  <span class="callout-label">Takeaway</span>
  A compiler pass is an optimization in the context of a specific hardware target,
  a specific numerical precision and a specific IR structure. The same pass can be
  a meaningful win on one target and a regression on another. The hardware matters
  as much as the transformation.
</div>

## 1. Introduction

### 1.1 The Performance Problem in Modern LLMs

Transformer-based language models dominate NLP. Models like GPT-2, BERT and their
distilled variants are deployed across text generation, question answering, code
completion and more. Their computational cost is substantial, even DistilGPT-2,
with 82 million parameters, requires hundreds of millions of floating-point
operations per inference pass. At scale, this translates into energy, hardware
cost and latency.

The conventional response is hardware, more and bigger GPUs, specialised accelerators,
aggressive batching. But there is a complementary and often underexploited lever,
**the compiler**. A sufficiently intelligent compiler, given the
<span class="term">computation graph<span class="term-def">
A computation graph is a diagram of every operation a neural network performs,
where each node is one operation (a matrix multiply, an addition) and each edge
is the tensor flowing between them.
</span></span>
of a model, can eliminate redundant operations, fuse adjacent kernels, rewrite memory
access patterns and restructure the graph in ways invisible to the model author,
but which can dramatically reduce execution time.

<div class="callout explainer">
  <span class="callout-label">Analogy: Why Compilers Matter</span>
  Think of following a recipe for a meal. You might see "chop 3 onions", "chop 3
  garlic cloves" and "chop 3 peppers" as three separate steps somewhere in the recipe.
  A <span class="joke">smart cook<span class="joke-def">obviously you...</span></span>
  might notice that all three happen at the cutting board. You then set up
  the cutting board once and chop everything in one go, instead of setting up
  a new cutting board for each step in the recipe. The results are identical,
  just with less wasted time. That is what a compiler pass does to a neural network's
  computation graph.
</div>

### 1.2 Why Compiler Optimisations, Not Model Compression?

There is a rich literature on making models smaller and faster through
model-level techniques:
<span class="term">pruning<span class="term-def">
Removing weights (or entire neurons) from a neural network that contribute little
to the output, like removing irrelevant words from a sentence. A pruned model has
fewer parameters and runs faster, but may lose some accuracy.</span></span>,
<span class="term">quantisation<span class="term-def">Replacing 32-bit floating-point
weights (float32) with lower-precision representations (float16, int8). This reduces
memory and increases throughput, at the cost of some numerical precision.</span></span>,
<span class="term">knowledge distillation<span class="term-def">Training a smaller
"student" model to mimic the behaviour of a larger "teacher" model. DistilBERT and
DistilGPT-2 (used in this project) were created this way.</span></span>,
and low-rank factorisation. These are valid, but they **change the model**.

Compiler optimisation is **semantics-preserving**. The output of the optimised model
is bit-for-bit identical to the baseline (for structural passes) or numerically
equivalent within a tolerance (for approximation passes). A compiler pass can be
applied to **any** model without retraining, and it is orthogonal to compression,
you can compress **and** compiler-optimise the same model.

<div class="callout finding">
  <span class="callout-label">Key distinction</span>
  Compression changes what the model <strong>is</strong>.
  Compilation changes how it <strong>runs</strong>.
</div>

## 2. Background

### 2.1 The Transformer Architecture

The transformer was introduced by Vaswani et al. (2017)<sup><a href="#ref-1">[1]</a></sup>
and the key innovation was the
<span class="term">self-attention<span class="term-def">
A mechanism that lets each token look at every other token in the input and
decide how much to weight each one when building its own representation.
See Jurafsky &amp; Martin (2026), §8.1.
</span></span>
mechanism. We'll go step by step, following the
<a href="https://netron.app/?url=https://ibrahimelk.github.io/assets/data/2026-04-10/distilbert.onnx">ONNX graph</a>
of a DistilBERT transformer model, in order to explain how it works in detail.
 
**Inputs**<br>
The model takes two integer tensors of shape `[1, seq_len]`. The first tensor,
`input_ids`, is a sequence of vocabulary indices. Each integer maps to one token
in the model's vocabulary. The second tensor, `attention_mask`, is a binary tensor
where 1 marks a real token and 0 marks padding.

<div style="text-align: center;" id="fig-1">
  <img src="/assets/images/2026-04-10/figure1.svg"
       style="width: 100%; height: auto;"
       alt="">
</div>
<div style="text-align: left;">
<figcaption>
<strong>Figure 1</strong>
</figcaption>
</div>
<br>

**Token embedding lookup**<br>
The model holds a learned weight matrix `wte` of shape
`[30522, 768]`, one 768-dimensional float vector per vocabulary token.
A `Gather` op uses `input_ids` as row indices and retrieves the
corresponding vectors, producing a float tensor of shape `[1, seq_len, 768]`.
These vectors encode semantic meaning. They were trained to place
conceptually similar tokens near each other in that 768-dimensional space.<sup><a href="#ref-12">[12]</a></sup>

<div style="text-align: center;" id="fig-2">
  <img src="/assets/images/2026-04-10/figure2.svg"
       style="width: 100%; height: auto;"
       alt="">
</div>
<div style="text-align: left;">
<figcaption>
<strong>Figure 2</strong>
</figcaption>
</div>
<br>

**Position embedding lookup**<br>
Because transformers process all tokens simultaneously and have no built-in
notion of order, position must be injected explicitly.<sup><a href="#ref-12">[12]</a></sup>
A second learned matrix, `wpe`, of shape
<span class="term">`[512, 768]`<span class="term-def">
DistilBERT's context window is 512 tokens. This is a hard architectural
limit: inputs longer than 512 tokens must be truncated or chunked before
they reach the model, and that's why seq_len &le; 512.
</span></span>,
stores one vector per possible position (0 through 511).
A `Gather` op with an `indices` attribute specifying the index sequence
`[0, 1, 2, ..., seq_len−1]` retrieves the corresponding
rows, giving another `[1, seq_len, 768]` tensor.<sup><a href="#ref-13">[13]</a></sup>

<div style="text-align: center;" id="fig-3">
  <img src="/assets/images/2026-04-10/figure3.svg"
       style="width: 100%; height: auto;"
       alt="">
</div>
<div style="text-align: left;">
<figcaption>
<strong>Figure 3</strong>
</figcaption>
</div>
<br>

**Embedding addition**<br>
The token and position tensors are added element-wise to
form the initial hidden state `h₀` of shape `[1, seq_len, 768]`. Each position in
this tensor blends what the token means with where it sits in the sequence.<sup><a href="#ref-12">[12]</a></sup>
This is the input to the first transformer block.

<div style="text-align: center;" id="fig-4">
  <img src="/assets/images/2026-04-10/figure4.svg"
       style="width: 100%; height: auto;"
       alt="">
</div>
<div style="text-align: left;">
<figcaption>
<strong>Figure 4</strong>
</figcaption>
</div>
<br>

**Six transformer blocks**<br>
The following structure runs six times with separate
learned weights at each layer.

<div style="text-align: center;" id="fig-5">
  <img src="/assets/images/2026-04-10/figure5.svg"
       style="width: 100%; height: auto;"
       alt="">
</div>
<div style="text-align: left;">
<figcaption>
<strong>Figure 5</strong>
</figcaption>
</div>
<br>

*Self Attention Mechanism*<br>
We will now describe the workings of one of those 6 blocks.
As we will see, DistilBERT uses <span class="term">post-norm ordering
<span class="term-def"> The Jurafsky &amp; Martin textbook (2026, §8.2, Fig. 8.7)
describes the more common modern pre-norm architecture (LayerNorm before
attention/FFN). DistilBERT instead follows the original Vaswani et al. (2017)
<b>post-norm</b> design (LayerNorm after the residual addition), which was also
used in BERT.
</span></span>.

*Q, K, V projections* multiply the normalised hidden state against three separate
weight matrices, each `[768, 768]`, to produce
<span class="term">Query, Key and Value tensors<span class="term-def">
These correspond to the three roles described in Jurafsky &amp; Martin
(2026, §8.1.1): the <b>query</b> is the current element being compared,
the <b>key</b> is a token being compared against, and the
<b>value</b> is the content that gets weighted and summed.
Each is computed
as q = xW<sup>Q</sup>, k = xW<sup>K</sup>, v = xW<sup>V</sup> (Eq. 8.9).
</span></span>.

Each result is reshaped from `[1, seq_len, 768]` to `[1, seq_len, 12, 64]`, splitting
the 768 dimensions across 12 attention heads of 64 dimensions each, then transposed
to `[1, 12, seq_len, 64]` so each head's data is contiguous in memory.

<div style="text-align: center;" id="fig-6">
  <img src="/assets/images/2026-04-10/qkv_projections_with_weights.svg"
       style="width: 100%; height: auto;"
       alt="">
</div>
<div style="text-align: left;">
<figcaption>
<strong>Figure 6</strong>
</figcaption>
</div>
<br>

*Scaled dot-product attention* runs independently per head. `Q · Kᵀ` produces a
`[seq_len, seq_len]` <span class="term">score matrix<span class="term-def">
This implements Eq. 8.11 from Jurafsky &amp; Martin (2026, §8.1.1):
score(x<sub>i</sub>, x<sub>j</sub>) = (q<sub>i</sub> · k<sub>j</sub>) / √d<sub>k</sub>.
The full parallel form is QK<sup>T</sup>/√d<sub>k</sub> (Eq. 8.33).
</span></span>
measuring how much each token attends to every
other. Dividing by &radic;64 = 8 prevents the scores from growing too large.

Afterwards, the `attention_mask` is applied. Positions where the mask is 0 (i.e. padding
tokens) receive a large negative value so that softmax drives those weights to
zero. Because DistilBERT is a
<span class="term">bidirectional encoder<span class="term-def">
Unlike the causal (left-to-right) decoder described in Jurafsky &amp; Martin
(2026, §8.1), DistilBERT is an encoder model in the BERT family. Encoders
let every token attend to every other token in the sequence, not just to
preceding tokens. Causal masking of future positions (§8.3, Fig. 8.10) is
therefore <b>not</b> used here. Jurafsky &amp; Martin introduce bidirectional
encoder models in Chapter 9.
</span></span>,
every non-padding token can attend to every other non-padding token,
there is no causal mask blocking future positions.
Softmax is applied row-wise to normalise the scores into weights. Finally,
multiplying by V yields a weighted sum of value vectors per position with output
shape `[1, 12, seq_len, 64]`.

<div style="text-align: center;" id="fig-7">
  <img src="/assets/images/2026-04-10/figure8.svg"
       style="width: 100%; height: auto;"
       alt="">
</div>
<div style="text-align: left;">
<figcaption>
<strong>Figure 7</strong>
</figcaption>
</div>
<br>

*Output projection, residual, and Layer Norm 1.* The 12 head outputs are
concatenated back to `[1, seq_len, 768]` via reshape and transpose, then
passed through a linear layer of shape `[768, 768]` that mixes information
across heads.<sup><a href="#ref-12">[12]</a></sup> A
<span class="term">residual connection<span class="term-def">
The residual connection provides a skip path for gradients and means the
attention sub-layer only needs to learn a correction to the input, not a full
new representation. In the residual stream view (Elhage et al., 2021;
Jurafsky &amp; Martin, 2026, §8.2), information flows up through the stream
and each component adds to it.
</span></span>
adds the hidden state from before the Q/K/V projections back to the
attention output. A normalisation step is then applied to the result. The formula is
$$ \text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \varepsilon} + \beta $$
where &gamma; and &beta; are learned parameters of size 768.<sup><a href="#ref-12">[12]</a></sup>

<div style="text-align: center;" id="fig-8">
  <img src="/assets/images/2026-04-10/figure9.svg"
       style="width: 100%; height: auto;"
       alt="">
</div>
<div style="text-align: left;">
<figcaption>
<strong>Figure 8</strong>
</figcaption>
</div>
<br>

*FFN, residual, and Layer Norm 2.*
A two-layer <span class="term">feed-forward network<span class="term-def">
This is the position-wise FFN described in Jurafsky &amp; Martin (2026, §8.2,
Eq. 8.21). It applies the same weights independently to each token position.
The textbook notes that it is common for the FFN hidden dimension to be
larger than the model dimension (e.g. d=512, d<sub>ff</sub>=2048 in the original
transformer; here d=768, d<sub>ff</sub>=3072, the same 4× factor).
</span></span>
takes the post-LN1 state as input. The first linear layer expands from 768 to
3072 dimensions (a 4× factor), a <span class="term">GELU<span class="term-def">
Gaussian Error Linear Unit: a smooth, non-monotonic activation function
defined as GELU(x) = x · Φ(x), where Φ(x) is the standard normal CDF.
Unlike ReLU, which hard-gates at zero, GELU weights each input by the
probability that it is positive under a Gaussian, producing a soft gating
effect. It is the default activation in BERT, GPT-2, and most modern
transformers (Hendrycks &amp; Gimpel, 2016).
</span></span>
non-linearity is applied element-wise,
and the second layer contracts back to 768. A second residual connection adds
the post-LN1 state to the FFN output, and a second layer normalisation
normalises the sum. This produces the final output of the block.

<div style="text-align: center;" id="fig-9">
  <img src="/assets/images/2026-04-10/figure10.svg"
       style="width: 100%; height: auto;"
       alt="">
</div>
<div style="text-align: left;">
<figcaption>
<strong>Figure 9</strong>
</figcaption>
</div>
<br>

**Output**<br>
The ONNX graph ends at the last hidden state of shape `[1, seq_len, 768]`. At
inference time, the hidden state goes through a <span class="term">masked LM head
<span class="term-def">
Unlike the autoregressive (left-to-right) language modelling described in
Jurafsky &amp; Martin (2026, §8.5), DistilBERT uses <b>masked language
modelling</b> (MLM): some input tokens are replaced with a special [MASK]
token, and the model predicts what word belongs in each masked position.
The textbook introduces this approach in Chapter 9 on BERT and
bidirectional encoders.
</span></span>
and softmax to obtain a probability
distribution over all 30&thinsp;522 tokens. For masked language modelling, the
predictions at [MASK] positions are read off to recover the hidden words. For
downstream tasks such as classification, only the representation at a special
[CLS] token position is used.

<div style="text-align: center;" id="fig-11">
  <img src="/assets/images/2026-04-10/figure13.svg"
       style="width: 100%; height: auto;"
       alt="">
</div>
<div style="text-align: left;">
<figcaption>
<strong>Figure 11</strong>
</figcaption>
</div>
<br>

<div class="callout finding">
  <span class="callout-label">What the compiler sees</span>
  The full graph for one forward pass contains hundreds of nodes. The dominant cost
  is the 6 &times; 3 = 18 QKV projection MatMuls and the 6 &times; 2 = 12 FFN
  MatMuls. Everything else, the Gathers, LayerNorms, Softmaxes, reshapes, is
  comparatively cheap. The compiler's job is to reduce the cost of those expensive
  nodes without changing the output.
</div>

### 2.2 What is Apache TVM?

Apache TVM is a machine learning compiler framework. It takes a neural network
(from any training framework) and produces fast, deployable code for any hardware
target.

#### What is an IRModule?

The official docs say an IRModule "encompasses the entirety of the ML models,
incorporating the computational graph, tensor programs, and potential calls to
external libraries." That's... not very helpful.

<div class="collapsible">
  <button class="collapsible-trigger" onclick="toggleCollapse(this)">
    <span>What even is an "IR"?</span>
    <span class="caret">▼</span>
  </button>
  <div class="collapsible-body">
    <p>When a compiler (any compiler, C++, Java, ...) transforms code,
    it can't jump directly from "human-readable source" to "machine code" in
    one step. There's too much to do. Instead, it converts through one or more
    <strong>Intermediate Representations (IRs)</strong>: simplified, structured
    forms of the program that are easier to analyse and transform.</p>
    <p>A classic example: GCC compiles C code to GIMPLE (a simplified form of C),
    then to RTL (Register Transfer Language, closer to assembly), then to
    machine code. Each IR is progressively lower-level.</p>
    <p><strong>TVM has two IRs:</strong></p>
    <ul style="padding-left:1.4rem; margin-bottom:0.8rem;">
      <li style="margin-bottom:0.5rem;"><strong>Relax</strong>, high-level, describes
      the computation graph (which operations, in what order, on what-shaped tensors)</li>
      <li><strong>TensorIR (TIR)</strong>, low-level, describes exactly how to implement
      each operation (loops, memory buffers, thread assignment)</li>
    </ul>
  </div>
</div>

An <strong>IRModule</strong> is simply a <em>container</em>, like a Python module
(a <code>.py</code> file), that holds all the functions needed to run a model,
written in TVM's IRs. That's it. The IRModule is a <em>data structure</em> that
describes the model. Passes transform this data structure. When you call 
<code>relax.build()</code>, TVM reads the IRModule and generates actual executable
code (LLVM bitcode or CUDA PTX).

<div class="callout explainer">
  <span class="callout-label">Grand Scheme Of Things</span>
  <p>
    <strong>ONNX import:</strong> <code>Gemm</code> node &rarr; <code>R.matmul</code><br>
    <small>A Relax op, high-level, no implementation yet</small><br><br>
    <strong>After LegalizeOps:</strong> <code>R.matmul</code> &rarr; <code>T.prim_func</code><br>
    <small>A TIR loop nest, now you have an actual implementation</small><br><br>
    <strong>After codegen:</strong> TIR &rarr; LLVM IR &rarr; native x86/CUDA PTX
  </p>
  <p>Graph Surgeon's passes happen at the first stage: they restructure the Relax
  graph <em>before</em> any TIR exists. This is why they're called "graph-level"
  passes.</p>
</div>

### 2.3 Relax: The Graph Level

Relax is TVM's high-level IR for representing the <em>computation graph</em> of a
model. It is a functional, dataflow-oriented language. Here's what a simple two-layer
MLP looks like in Relax:

<span class="code-label">Relax</span>

```python
@R.function
def main(
    x:      R.Tensor((1, 784),   dtype="float32"),
    weight: R.Tensor((784, 256), dtype="float32"),
    bias:   R.Tensor((256,),     dtype="float32"),
) -> R.Tensor((1, 256), dtype="float32"):
    with R.dataflow():              # "safe zone" for optimisation
        lv0 = R.matmul(x, weight)
        lv1 = R.add(lv0, bias)
        gv  = R.nn.relu(lv1)
        R.output(gv)
    return gv
```

Notice that Relax does <em>not</em> say how to implement <code>R.matmul</code>.
No loops, no thread assignments, no tiling. The <strong>dataflow block</strong>
(<code>with R.dataflow()</code>) is a hint to the compiler that this region is
purely functional, no side effects, so it can safely reorder, fuse or eliminate
operations.

<div class="collapsible">
  <button class="collapsible-trigger" onclick="toggleCollapse(this)">
    <span>What is a "binding" and why does it matter for passes?</span>
    <span class="caret">▼</span>
  </button>
  <div class="collapsible-body">
    <p>In Relax, each line in a dataflow block is a <strong>binding</strong>:
    a name assigned to the result of one operation.
    <code>lv0 = R.matmul(x, weight)</code> means: "compute the matmul, name the
    result <code>lv0</code>."</p>
    <p>Compiler passes traverse these bindings. The QKV fusion pass, for example,
    uses <code>ExprMutator</code>, a TVM API that walks through every binding in the
    dataflow block. When it finds three consecutive matmul bindings that all take the
    same input variable, it knows it has found a Q, K, V triple and can merge them.</p>
    <p>This is fundamentally similar to how static analysis tools work in software
    engineering, they traverse an AST (abstract syntax tree) looking for patterns
    and transform them. Relax's dataflow graph is TVM's equivalent of an AST for ML
    models.</p>
  </div>
</div>

The `@R.function` decorator does the same trick as `@T.prim_func` below.
It never executes the function as Python. Instead, TVM captures it into
a tree of IR objects. The line `lv0 = R.matmul(x, weight)` becomes a
`relax.VarBinding` whose value is a `relax.Call` node:

```python
# What Python "sees" at runtime after the decorator runs:
VarBinding(
    var   = DataflowVar("lv0"),
    value = Call(op="relax.matmul", args=[Var("x"), Var("weight")])
)
```

Pattern-matching in a pass is literally just checking this tree. In
`qkv_fusion.py`, detecting a matmul binding is:

```python
def _match_matmul(self, expr):
    # check if node is Call to "relax.matmul" op
    if not (isinstance(expr, relax.Call) and
            expr.op == tvm.ir.Op.get("relax.matmul")):
        return None
    x, w = expr.args[0], expr.args[1]
    ...
```

And rewriting the graph is rebuilding the binding list with new `Call` nodes in
place of the old ones.
<!-- when it replaces three `VarBinding`s with one fused matmul plus three -->
<!-- `TupleGetItem` slices. -->

### 2.4 TensorIR: The Kernel Level

Below Relax sits TensorIR (TIR), TVM's low-level IR for individual tensor programs.
Where Relax says *what* to compute, TIR specifies *how*: the loop nests, memory
layout, thread bindings and tiling decisions.

<span class="code-label">TensorIR</span>

```python
@T.prim_func
def matmul(A: T.Buffer((128, 768), "float32"),
           B: T.Buffer((768, 768), "float32"),
           C: T.Buffer((128, 768), "float32")):
    for i, j, k in T.grid(128, 768, 768):
        with T.block("matmul"):
            # SSR = Spatial, Spatial, Reduction
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

Like the `@R.function` decorator, `@T.prim_func` tells TVM not to run this as
Python, but to parse it into IR objects. Each buffer is declared with an explicit
shape and dtype, not as a hint, but as a hard constraint that lets later passes
compute memory strides and choose vectorisation widths without guessing. `C` is
the output buffer, written in place, there is no `return` statement, which is the
convention for all `prim_func` kernels.<sup><a href="#ref-16">[16]</a></sup>

`T.grid(128, 768, 768)` is syntactic sugar for three nested loops, defining
128 × 768 × 768 = 75,497,472 iterations. The `T.block` wraps the actual
computation into a self-contained unit, and the `"SSR"` annotation on the axes
is the crucial part. It tells the compiler that `vi` and `vj` are **spatial**
(independent, safe to parallelise and tile freely) while `vk` is a **reduction**
(all `k` iterations write to the same accumulator, so it cannot be freely
reordered without an explicit parallel reduction pattern). The `T.init()` block
cleanly separates the accumulator reset from the accumulation step, letting the
compiler hoist or fuse the initialisation correctly when the schedule is
transformed.<sup><a href="#ref-16">[16]</a></sup>

<div class="collapsible" >
<button class="collapsible-trigger" onclick="toggleCollapse(this)">
<span>Deep dive into the TIR program</span>
<span class="caret">▼</span>
</button>
<div class="collapsible-body" markdown="1">
<p><strong><code>@T.prim_func</code> declaring a primitive tensor function.</strong>
A <em>primitive tensor function</em> is the smallest schedulable grain of work in TVM:
one matmul, one convolution, one fused conv+relu. Marking a function with this decorator
tells TVM to parse it into IR, type-check it, and make it available for transformation
and code generation.<sup><a href="#ref-14">[14]</a></sup></p>
<p><strong><code>T.axis.remap("SSR", [i, j, k])</code></strong>
This shorthand declares three block axes in one line, equivalent to writing:
<sup><a href="#ref-16">[16]</a></sup></p>
<pre><code>vi = T.axis.spatial(128, i)   # S independent, safe to parallelise
vj = T.axis.spatial(768, j)   # S independent, safe to parallelise
vk = T.axis.reduce (768, k)   # R accumulates into C[vi,vj]
</code></pre>
<p>Each declaration conveys three things: the binding of the block variable to
its outer loop variable, the range of valid indices and the axis type. The
spatial axes can be freely tiled, vectorised and mapped to GPU threads without
data races. The reduce axis <code>vk</code> contributes to the same accumulator
for all <code>k</code> values and can only be parallelised with an explicit
parallel reduction pattern such as a tree-reduce.<sup><a href="#ref-16">[16]</a></sup></p>
<p><strong><code>T.init()</code></strong>
The <code>with T.init()</code> sub-block formally declares the initialisation
action for the reduction.<sup><a href="#ref-16">[16]</a></sup> It makes reduction
semantics unambiguous and allows the compiler to hoist this initialisation
correctly when the loop is later split or vectorised, it only needs to fire
once per <code>(vi, vj)</code> pair, not on every <code>k</code> iteration.</p>
  </div>
</div>

### 2.5 The Compilation Pipeline

TVM's compilation is a sequence of transformation passes applied to the IRModule.
Click a stage below to learn what happens at each step:

<div class="interactive-panel">
  <div class="panel-header"><span class="panel-icon">🏭
  </span> The TVM Compilation Pipeline</div>
  <div class="panel-body">
    <div class="pipeline-interactive" id="pipe-stages">
      <div class="pipe-stage" onclick="showPipeStage(this, 'stage-import')">
        <div class="pipe-stage-icon">📥</div>
        <div><div class="pipe-stage-name">1. ONNX Import</div><div class="pipe-stage-sub">from_onnx() → Relax IRModule</div></div>
      </div>
      <div class="pipe-arrow">↓</div>
      <div class="pipe-stage" onclick="showPipeStage(this, 'stage-custom')">
        <div class="pipe-stage-icon">⚙️</div>
        <div><div class="pipe-stage-name">2. Graph Surgeon Passes</div><div class="pipe-stage-sub">QKV fusion, FuseTransposeMatmul</div></div>
      </div>
      <div class="pipe-arrow">↓</div>
      <div class="pipe-stage" onclick="showPipeStage(this, 'stage-normalize')">
        <div class="pipe-stage-icon">🧹</div>
        <div><div class="pipe-stage-name">3. Normalize + FoldConstant + DCE</div><div class="pipe-stage-sub">Cleanup passes: fold constants, remove dead code</div></div>
      </div>
      <div class="pipe-arrow">↓</div>
      <div class="pipe-stage" onclick="showPipeStage(this, 'stage-legalize')">
        <div class="pipe-stage-icon">🔽</div>
        <div><div class="pipe-stage-name">4. LegalizeOps</div><div class="pipe-stage-sub">Relax ops → TIR implementations</div></div>
      </div>
      <div class="pipe-arrow">↓</div>
      <div class="pipe-stage" onclick="showPipeStage(this, 'stage-fusekernels')">
        <div class="pipe-stage-icon">🔗</div>
        <div><div class="pipe-stage-name">5. FuseOps + FuseTIR</div><div class="pipe-stage-sub">Adjacent TIR functions fused into single kernels</div></div>
      </div>
      <div class="pipe-arrow">↓</div>
      <div class="pipe-stage" onclick="showPipeStage(this, 'stage-build')">
        <div class="pipe-stage-icon">🚀</div>
        <div><div class="pipe-stage-name">6. relax.build()</div><div class="pipe-stage-sub">TIR → LLVM / CUDA PTX → executable</div></div>
      </div>
    </div>
    <div id="pipeline-detail">
      <em>Click a stage above to see what happens there.</em>
    </div>
  </div>
</div>

### 2.6 The ONNX Format

ONNX (Open Neural Network Exchange)<sup><a href="#ref-6">[6]</a></sup> is a standard,
framework-agnostic file format for neural networks.

<div class="callout explainer">
  <span class="callout-label">💡 Why go through ONNX at all?</span>
  <p>TVM can import from PyTorch directly via <code>torch.export</code>,
  but ONNX provides a more stable, explicit graph. ONNX explicitly lays
  out every operation as a typed node in a directed acyclic graph,
  making it easier to pattern-match and restructure.</p>
</div>

The project uses ONNX opset 18 with static shapes (batch=1, seq=128).
Static shapes are important for Graph Surgeon: the QKV fusion pass needs
to know the exact dimensions of weight matrices at compile time to perform
the concatenation. With dynamic shapes, you'd need to defer this to runtime.


## 3. Technical Approach

### 3.1 Target Models

<table class="results-table">
<tr><th>Model</th><th>Type</th><th>Params</th><th>Layers</th><th>Why it's interesting</th></tr>
<tr>
  <td>DistilGPT-2</td><td>LLM (decoder)</td><td class="mono">82M</td><td>6</td>
  <td>Causal language model; QKV is not implemented as separate MatMuls in ONNX</td>
</tr>
<tr>
  <td>DistilBERT</td><td>LLM (encoder)</td><td class="mono">66M</td><td>6</td>
  <td>Bidirectional encoder; QKV is implemented as separate MatMuls in ONNX</td>
</tr>
</table>

<div class="collapsible">
  <button class="collapsible-trigger" onclick="toggleCollapse(this)">
    <span>What's the difference between an encoder and a decoder transformer?</span>
    <span class="caret">▼</span>
  </button>
  <div class="collapsible-body">
    <p><strong>Encoder (BERT-style):</strong> Every token can attend to every other
    token, "bidirectional" attention. Great for tasks where you process a whole
    sentence and need to understand it (classification, NER, question answering).
    Processes the whole input at once.</p>
    <p><strong>Decoder (GPT-style):</strong> Each token can only attend to previous
    tokens, "causal" (left-to-right) attention, enforced by masking future positions
    in the attention score matrix. Designed for generation: you predict one token
    at a time, left to right. GPT-2 and ChatGPT use this architecture.</p>
  </div>
</div>

### 3.2 The QKV Fusion Pass

The most important pass in this project. Here's what happens step by step:

<div class="steps">
  <div class="step">
    <div class="step-num">1</div>
    <div class="step-content">
      <strong>Graph traversal:</strong> The pass uses TVM's <code>ExprMutator</code> API to walk through each <code>DataflowBlock</code> in the main Relax function, visiting bindings one at a time.
    </div>
  </div>
  <div class="step">
    <div class="step-num">2</div>
    <div class="step-content">
      <strong>Pattern matching:</strong> For each group of three matmul+bias-add subgraphs that share a common input variable, the pass identifies a Q, K, V triple.
    </div>
  </div>
  <div class="step">
    <div class="step-num">3</div>
    <div class="step-content">
      <strong>Weight concatenation:</strong> The three weight matrices [768×768], [768×768], [768×768] are concatenated via <code>numpy.concatenate</code> into a single [768×2304] matrix. Biases too. This happens <em>once</em>, at compile time, not at every inference.
    </div>
  </div>
  <div class="step">
    <div class="step-num">4</div>
    <div class="step-content">
      <strong>Graph rewrite:</strong> The three original bindings are replaced with: one fused matmul (768×2304), followed by three <code>relax.op.strided_slice</code> calls to extract Q, K, V from the result.
    </div>
  </div>
  <div class="step">
    <div class="step-num">5</div>
    <div class="step-content">
      <strong>Constant folding:</strong> A follow-up <code>FoldConstant()</code> pass evaluates the concatenated weights, materialising them into the graph. No runtime overhead from the concatenation.
    </div>
  </div>
</div>

<div class="interactive-panel" id="qkv-demo">
  <div class="panel-header"><span class="panel-icon">✂️
  </span> QKV Fusion </div>
  <div class="panel-body2">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
      <div style="font-family:'DM Sans',sans-serif; font-size:0.85rem; color:var(--text-secondary);" id="qkv-status">Showing: <strong>Before fusion</strong></div>
      <button class="btn" onclick="toggleQKV()">Toggle Fusion</button>
    </div>
    <div id="qkv-before" class="qkv-state active">
      <div style="display:grid; grid-template-columns:1fr auto 1fr; gap:1rem; align-items:start;">
        <div>
          <div style="font-family:'DM Sans',sans-serif; font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:var(--text-secondary); margin-bottom:0.5rem;">Relax Graph Nodes</div>
          <div class="kern-box highlight">lv_q = R.matmul(x, W_Q)<br>lv_q2 = R.add(lv_q, b_Q)</div>
          <div class="kern-box highlight" style="margin-top:0.3rem;">lv_k = R.matmul(x, W_K)<br>lv_k2 = R.add(lv_k, b_K)</div>
          <div class="kern-box highlight" style="margin-top:0.3rem;">lv_v = R.matmul(x, W_V)<br>lv_v2 = R.add(lv_v, b_V)</div>
          <div style="font-family:'DM Sans',sans-serif; font-size:0.78rem; margin-top:0.7rem; color:var(--red);">↑ 6 ops, 3 reads of X from memory</div>
        </div>
        <div style="text-align:center; padding-top:3rem; color:var(--text-secondary); font-size:1.5rem;">→</div>
        <div>
          <div style="font-family:'DM Sans',sans-serif; font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:var(--text-secondary); margin-bottom:0.5rem;">Kernel Launches (GPU)</div>
          <svg width="100%" height="140" viewBox="0 0 200 140" style="display:block;">
            <rect x="5" y="5" width="190" height="35" rx="4" fill="#fdedec" stroke="#f5b7b1"/>
            <text x="100" y="22" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="9" fill="#922b21">kernel_matmul_Q(X, W_Q)</text>
            <text x="100" y="35" text-anchor="middle" font-family="DM Sans,sans-serif" font-size="8" fill="#555">reads X: ~393 KB</text>
            <rect x="5" y="48" width="190" height="35" rx="4" fill="#fdedec" stroke="#f5b7b1"/>
            <text x="100" y="65" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="9" fill="#922b21">kernel_matmul_K(X, W_K)</text>
            <text x="100" y="78" text-anchor="middle" font-family="DM Sans,sans-serif" font-size="8" fill="#555">reads X: ~393 KB again</text>
            <rect x="5" y="91" width="190" height="35" rx="4" fill="#fdedec" stroke="#f5b7b1"/>
            <text x="100" y="108" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="9" fill="#922b21">kernel_matmul_V(X, W_V)</text>
            <text x="100" y="121" text-anchor="middle" font-family="DM Sans,sans-serif" font-size="8" fill="#555">reads X: ~393 KB again</text>
            <text x="100" y="136" text-anchor="middle" font-family="DM Sans,sans-serif" font-size="8" fill="#c0392b" font-weight="700">Total X reads: ~1.2 MB</text>
          </svg>
        </div>
      </div>
    </div>
    <div id="qkv-after" class="qkv-state">
      <div style="display:grid; grid-template-columns:1fr auto 1fr; gap:1rem; align-items:start;">
        <div>
          <div style="font-family:'DM Sans',sans-serif; font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:var(--green); margin-bottom:0.5rem;">Relax Graph Nodes (after fusion)</div>
          <div class="kern-box fused-kern">lv_qkv = R.matmul(x, W_QKV)<br>lv_qkv2 = R.add(lv_qkv, b_QKV)</div>
          <div class="kern-box" style="margin-top:0.3rem;">Q = R.strided_slice(lv_qkv2, [0,0,0], [b,s,d])</div>
          <div class="kern-box" style="margin-top:0.3rem;">K = R.strided_slice(lv_qkv2, [0,0,d], [b,s,2d])</div>
          <div class="kern-box" style="margin-top:0.3rem;">V = R.strided_slice(lv_qkv2, [0,0,2d], [b,s,3d])</div>
          <div style="font-family:'DM Sans',sans-serif; font-size:0.78rem; margin-top:0.7rem; color:var(--green);">↑ 5 ops, 1 read of X (slices are near-free)</div>
        </div>
        <div style="text-align:center; padding-top:3rem; color:var(--text-secondary); font-size:1.5rem;">→</div>
        <div>
          <div style="font-family:'DM Sans',sans-serif; font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:var(--green); margin-bottom:0.5rem;">Kernel Launches (GPU)</div>
          <svg width="100%" height="140" viewBox="0 0 200 140" style="display:block;">
            <rect x="5" y="15" width="190" height="50" rx="4" fill="#eafaf1" stroke="#abebc6" stroke-width="2"/>
            <text x="100" y="36" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="9" fill="#1e8449" font-weight="700">kernel_matmul_QKV(X, W_QKV)</text>
            <text x="100" y="50" text-anchor="middle" font-family="DM Sans,sans-serif" font-size="8" fill="#555">reads X once: ~393 KB</text>
            <text x="100" y="62" text-anchor="middle" font-family="DM Sans,sans-serif" font-size="8" fill="#555">W_QKV is 768×2304 (3× wider)</text>
            <rect x="5" y="75" width="58" height="25" rx="4" fill="#f5eef8" stroke="#d7bde2"/>
            <text x="34" y="92" text-anchor="middle" font-family="DM Sans,sans-serif" font-size="8" fill="#6c3483">slice_Q</text>
            <rect x="71" y="75" width="58" height="25" rx="4" fill="#f5eef8" stroke="#d7bde2"/>
            <text x="100" y="92" text-anchor="middle" font-family="DM Sans,sans-serif" font-size="8" fill="#6c3483">slice_K</text>
            <rect x="137" y="75" width="58" height="25" rx="4" fill="#f5eef8" stroke="#d7bde2"/>
            <text x="166" y="92" text-anchor="middle" font-family="DM Sans,sans-serif" font-size="8" fill="#6c3483">slice_V</text>
            <text x="100" y="115" text-anchor="middle" font-family="DM Sans,sans-serif" font-size="8" fill="#27ae60" font-weight="700">Total X reads: ~393 KB ✓</text>
            <text x="100" y="130" text-anchor="middle" font-family="DM Sans,sans-serif" font-size="8" fill="#27ae60">Saved ~820 KB per attention layer</text>
          </svg>
        </div>
      </div>
    </div>
  </div>
</div>

### 3.3 FuseTransposeMatmul Pass

FuseTransposeMatmul is a built-in TVM pass. In ONNX-exported
transformers, some operations produce explicit <code>Transpose</code> nodes before
a <code>Matmul</code>. This pass folds those transpose nodes into the matmul's
<code>transA</code>/<code>transB</code> flags, eliminating a memcpy on CPU, where
explicit transpositions have real cost. On GPU, cuBLAS handles
<code>CUBLAS_OP_T</code> natively as a flag, so this pass eliminates
a Relax node that was never a real cost to begin with.

## 4. Results

* GPU: NVIDIA Turing (sm_75), CUDA. <br>
* CPU: x86-64. <br>
* Benchmarks: 200 warmup + 200 timed iterations.<br>
* Statistical significance: Welch t-test at p&nbsp;&lt;&nbsp;0.01.<br>

### 4.1 GPU Results

<table class="results-table">
<tr><th>Model</th><th>Pipeline</th><th>Mean (ms)</th><th>Std (ms)</th><th>Speedup</th><th>p&lt;0.01?</th></tr>
<tr><td rowspan="4">DistilGPT-2</td>
  <td class="mono">o2_baseline</td><td>27.03</td><td>1.550</td><td>-</td><td>-</td></tr>
<tr><td class="mono">llm_fuse_transpose</td><td>27.54</td><td>0.005</td><td class="negative">0.981×</td><td>✗</td></tr>
<tr><td class="mono">llm_qkv</td><td>27.54</td><td>0.001</td><td class="negative">0.982×</td><td>✗</td></tr>
<tr><td class="mono">llm_dlight</td><td>27.71</td><td>0.053</td><td class="negative">0.976×</td><td>✓</td></tr>
<tr><td rowspan="4">DistilBERT</td>
  <td class="mono">o2_baseline</td><td>14.78</td><td>0.004</td><td>-</td><td>-</td></tr>
<tr><td class="mono">llm_fuse_transpose</td><td>14.79</td><td>0.005</td><td class="negative">0.999×</td><td>✓</td></tr>
<tr><td class="mono">llm_qkv</td><td>14.92</td><td>0.004</td><td class="negative">0.990×</td><td>✓</td></tr>
<tr><td class="mono">llm_dlight</td><td>14.77</td><td>0.041</td><td class="neutral">1.001×</td><td>✗</td></tr>
</table>

<div class="callout finding">
  <span class="callout-label">Findings: GPU LLMs</span>
  <p><strong>No graph-level pass improved transformer inference on GPU.</strong> QKV fusion, transpose elimination, and DLight TIR scheduling all resulted in no change or slight regressions.
  </p>
</div>

### 4.2 CPU Results

<table class="results-table">
<tr><th>Model</th><th>Pipeline</th><th>Mean (ms)</th><th>Speedup</th><th>p&lt;0.01?</th></tr>
<tr><td rowspan="3">DistilGPT-2</td>
  <td class="mono">o2_baseline</td><td>20,282</td><td>-</td><td>-</td></tr>
<tr><td class="mono">llm_qkv</td><td>19,657</td><td class="positive">1.032×</td><td>✓</td></tr>
<tr><td class="mono">llm_all</td><td>19,732</td><td class="positive">1.028×</td><td>✓</td></tr>
<tr><td rowspan="3">DistilBERT</td>
  <td class="mono">o2_baseline</td><td>9,293</td><td>-</td><td>-</td></tr>
<tr><td class="mono">llm_fuse_transpose</td><td>8,998</td><td class="positive">1.033×</td><td>✓</td></tr>
<tr><td class="mono">llm_qkv</td><td>8,309</td><td class="positive" style="font-size:1.05em;">1.118×</td><td>✓</td></tr>
</table>

<div class="callout finding">
  <span class="callout-label">⭐ Findings: CPU LLMs</span>
  <p><strong>QKV fusion delivers +11.8% on CPU for DistilBERT.</strong> The single largest improvement in the entire project. The pass was designed for CPU cache behaviour, and it works exactly where it was designed to work. FuseTransposeMatmul also helps (+3.3%), confirming that explicit transpose nodes carry real cost on x86, unlike GPU, where cuBLAS handles transposed inputs natively.</p>
</div>

<div class="callout warning">
  <span class="callout-label">Important Caveat</span>
  <p>The CPU runs fell back to <code>-mcpu=generic</code> due to an LLVM 22.1.2 compatibility issue with <code>-mcpu=native</code>. This means <strong>no AVX2, no FMA</strong>, the CPU results are without SIMD vectorisation. The 11.8% QKV speedup would likely be even larger with AVX2 enabled, as the fused matmul's wider inner loop can better exploit SIMD vector units (which process 8 floats in parallel).</p>
</div>

## 5. Conclusion and Future Work

By concatenating $$W_{Q}, W_{K}, W_{V}$$ into $$W_{QKV}$$ and doing one wide matmul,
we save the cost of loading the input tensor X from memory three times instead
of once. On CPU, where memory bandwidth is tight and cache is small, that's a real
win, hence the 11.8%. On GPU, cuBLAS is already so good at scheduling individual
matmuls (and the L2 cache is large enough to keep X hot across launches) that the
saving evaporates. The slight regression we see on GPU is probably because the
768×2304 matmul is a less "canonical" shape than three 768×768 ones, and cuBLAS's
internal heuristics are tuned for the latter.

The real bottleneck is the attention score matrix. The QKV projections are
compute-bound matmuls, hardware is doing useful arithmetic almost every cycle.
The expensive part that graph-level fusion can't touch is the attention computation
itself, `Q · Kᵀ` produces a `[seq_len, seq_len]` score matrix.
The correct solution to this is FlashAttention (Dao et al., 2022),<sup><a href="#ref-17">[17]</a></sup>
which fuses `Q · Kᵀ`, scaling, masking, softmax, `softmax · V`, into a single tiled CUDA kernel.

## References

<ol class="references-list">

<li id="ref-1">
  Vaswani, A. et al. (2017). <em>Attention Is All You Need.</em>
  Advances in Neural Information Processing Systems, 30.
  <a href="https://arxiv.org/abs/1706.03762">arxiv.org/abs/1706.03762</a>
</li>

<li id="ref-2">
  Chen, T. et al. (2018). <em>TVM: An Automated End-to-End Optimizing Compiler for Deep Learning.</em>
  <a href="https://arxiv.org/abs/1802.04799">arxiv.org/abs/1802.04799</a>
</li>

<li id="ref-3">
  Apache TVM Documentation. <em>Overview.</em>
  <a href="https://tvm.apache.org/docs/get_started/overview.html">tvm.apache.org/docs/get_started/overview.html</a>
</li>

<li id="ref-4">
  Apache TVM Documentation. <em>IRModule.</em>
  <a href="https://tvm.apache.org/docs/get_started/tutorials/ir_module.html">tvm.apache.org/docs/get_started/tutorials/ir_module.html</a>
</li>

<li id="ref-5">
  Apache TVM Documentation. <em>Relax: Graph Abstraction for ML Models.</em>
  <a href="https://tvm.apache.org/docs/deep_dive/relax/abstraction.html">tvm.apache.org/docs/deep_dive/relax/abstraction.html</a>
</li>

<li id="ref-6">
  ONNX Project. <em>ONNX: Open Neural Network Exchange.</em>
  <a href="https://onnx.ai/onnx/intro/concepts.html">onnx.ai/onnx/intro/concepts.html</a>
</li>

<li id="ref-7">
  Shao, J. et al. (2022).
  <em>Tensor Program Optimization with Probabilistic Programs.</em> NeurIPS 2022.
  <a href="https://proceedings.neurips.cc/paper_files/paper/2022/file/e894eafae43e68b4c8dfdacf742bcbf3-Paper-Conference.pdf">proceedings.neurips.cc</a>
</li>

<li id="ref-8">
  Apache TVM Documentation. <em>MetaSchedule: Search-Based Auto-Tuning.</em>
  <a href="https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html">tvm.apache.org/docs/.../meta_schedule.html</a>
</li>

<li id="ref-9">
  Apache TVM Documentation. <em>Optimize Large Language Model.</em>
  <a href="https://tvm.apache.org/docs/how_to/tutorials/optimize_llm.html">tvm.apache.org/docs/.../optimize_llm.html</a>
</li>

<li id="ref-10">
  Li, Y. et al. (2021). <em>A Short Study on Compressing Decoder-Based Language Models.</em>
  <a href="https://arxiv.org/abs/2110.08460">arxiv.org/abs/2110.08460</a>
</li>

<li id="ref-11">
  Apache TVM Documentation. <em>Customize Optimization.</em>
  <a href="https://tvm.apache.org/docs/how_to/tutorials/customize_opt.html">tvm.apache.org/docs/.../customize_opt.html</a>
</li>

<li id="ref-12">
  Jurafsky, D. &amp; Martin, J. H. <em>Speech and Language Processing.</em>
  Draft of January 6, 2026.
  <a href="https://web.stanford.edu/~jurafsky/slp3/">web.stanford.edu/~jurafsky/slp3/</a>
  <a href="https://web.stanford.edu/~jurafsky/slp3/thanks.html">It's a very good book!</a>
</li>

<li id="ref-13">
  ONNX Operator Reference. <em>Gather.</em>
  <a href="https://onnx.ai/onnx/operators/onnx__Gather.html#inputs">onnx.ai/onnx/operators/onnx__Gather.html</a>
</li>

<li id="ref-14">
  Apache TVM Documentation. <em>Tensor Program Abstraction.</em>
  <a href="https://tvm.apache.org/docs/deep_dive/tensor_ir/abstraction.html">tvm.apache.org/docs/deep_dive/tensor_ir/abstraction.html</a>
</li>

<li id="ref-15">
  Apache TVM Documentation. <em>TensorIR Overview.</em>
  <a href="https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html">tvm.apache.org/docs/deep_dive/tensor_ir/index.html</a>
</li>

<li id="ref-16">
  Apache TVM Documentation. <em>Understand TensorIR Abstraction.</em>
  <a href="https://tvm.apache.org/docs/deep_dive/tensor_ir/learning.html">tvm.apache.org/docs/deep_dive/tensor_ir/learning.html</a>
</li>

<li id="ref-17">
  Dao, T., Fu, D. Y., Ermon, S., Rudra, A., &amp; Ré, C. (2022). <em>FlashAttention: Fast and
  Memory-Efficient Exact Attention with IO-Awareness.</em>
  Advances in Neural Information Processing Systems, 35.
  <a href="https://arxiv.org/abs/2205.14135">arxiv.org/abs/2205.14135</a>
</li>

<li id="ref-18">
  Dao, T. (2023). <em>FlashAttention-2: Faster Attention with Better
  Parallelism and Work Partitioning.</em>
  International Conference on Learning Representations (ICLR 2024).
  <a href="https://arxiv.org/abs/2307.08691">arxiv.org/abs/2307.08691</a>
</li>

</ol>
