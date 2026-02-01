---
layout: post
title: "Making Language Models Go BRRR... "
date:  2026-01-10
keywords:
  [
    "tvm",
    "onnx",
  ]
categories: notes
---

**Author:** Ibrahim El Kaddouri  

**Repository:** <span class="tooltip">
<a href="https://github.com/IbrahimElk/GraphSurgeon">
    <img src="/assets/images/2026-01-01/rabbit.jpg" style="width: 10%">
</a>
<span class="tooltip-text"> The repository is private </span>
</span>
<br>
<br>

## Coming Soon

<br>
*Graph Surgeon* is a small compiler project built on Apache TVM where I take a model
exported to ONNX, import it into TVM and compile it to get a clean baseline with proper
benchmarks. From there, I implement my own graph-level optimization pass, the kind of
transformation a real compiler would do, then rebuild the model through the new
pipeline and measure the impact. The final result is a before/after comparison backed
by a benchmark that shows whether the pass actually makes the model faster
(or if it makes it worse, and why).


## References

1. [Apache TVM](https://tvm.apache.org/docs/get_started/tutorials/quick_start.html)
2. [arXiv TVM](https://arxiv.org/abs/1802.04799)
3. [interesting](https://bit.ly/4fPyhMM)
