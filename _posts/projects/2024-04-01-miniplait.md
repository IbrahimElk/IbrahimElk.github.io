---
layout: post
title: "Minimal Language, Maximal Suffering "
date: 2024-04-01
summary: "A small functional language with desugaring, lexical scoping,
static typing and a lazy evaluator with recursion. "
keywords:
  - interpreter
  - abstract-syntax-tree
  - lexical-scoping
  - closures
  - desugaring
  - syntactic-sugar
  - type-systems
  - lazy-evaluation
categories: projects
---

**Author:** Ibrahim El Kaddouri

**Repository:**
<a href="https://github.com/IbrahimElk/miniplait">
  <img src="/assets/images/2024-04-01/icon.png" style="width:5.0%" />
</a>


*Have you ever wondered what happens when you type `x = 5` in Python,
or when you write `const add = (a, b) => a + b` in JavaScript ?*

This project implements *miniPlait*, a small functional language designed
to make the core ideas of programming languages concrete. It based on a
project course by [Brown Univeristy](https://cs.brown.edu/courses/csci1730/2020/)
by Professor Shiram. The functional language implements lexical scoping, closures,
syntactic desugaring and typing. The implementation progresses from an eager
interpreter to a statically typed variant and finally to a lazy evaluator extended
with typed recursion.


## Introduction

Implementing a language is about choosing **semantics** (what programs *mean*),
not just **syntax** (what programs *look like*). miniPlait is intentionally small,
but still expressive enough to require real design decisions:

- Lexical scoping vs. dynamic scoping
- Surface syntax vs. core syntax
- Runtime checks vs. compile-time checks
- Eager evaluation vs. lazy evaluation


The [code](https://github.com/IbrahimElk/miniplait) is
structured as 'phases' (each one builds on the previous one):

- `code/1-interp/`: eager interpreter + desugaring
- `code/3-tcheck/`: standalone static type checker
- `code/4-tynterp/`: combined typed interpreter
- `code/5-lazy/`: lazy typed interpreter + `rec`


## Implementation

### Abstract Syntax Representation

The language uses algebraic data types to represent abstract syntax trees.
The core expression type `Expr` includes:

<!-- invisinsible character needed for colour rendering -->
```racket
(define-type Expr
  (e-num [value : Number])
  (e-str [value : String])
  (e-bool [value : Boolean])
  (e-op [op : Operator]
        [left : Expr]
        [right : Expr])
  (e-if [​cond : Expr]
        [consq : Expr]
        [altern : Expr])
  (e-lam [param : Symbol]
         [body : Expr])
  (e-app [func : Expr]
         [arg : Expr])
  (e-var [name : Symbol]))
```

Values produced by evaluation are represented as:

```racket
(define-type Value
  (v-num [value : Number])
  (v-str [value : String])
  (v-bool [value : Boolean])
  (v-fun [param : Symbol]
         [body : Expr]
         [env : Env]))
```

Notably, function values (`v-fun`) capture their defining environment, implementing
lexical scoping through closure creation.


## Part I: Core Interpreter

The foundational interpreter implements eager, call-by-value evaluation with
environment-based variable resolution.

### Environment Model

The interpreter maintains an immutable hash table mapping variable names to values:

```racket
(define-type-alias Env (Hashof Symbol Value))
```

Immutable hash tables ensure that environment extensions create new bindings without
mutating existing ones, supporting proper lexical scoping and variable shadowing.

### Evaluation Semantics

The `interp` function implements a post-order traversal of the abstract syntax tree,
evaluating children left-to-right before processing parent nodes. This evaluation order
provides deterministic error reporting when multiple errors exist in a program.

*Binary Operations*: The language supports four binary operators:
- Arithmetic addition (`+`) on numbers
- String concatenation (`++`) on strings  
- Numeric equality (`num=`) returning booleans
- String equality (`str=`) returning booleans

Type checking occurs at runtime, raising errors when operands have incorrect types.

*Conditional Expressions*: The `if` construct evaluates its condition first. If the
condition evaluates to `true`, the consequent branch executes; if `false`, the
alternative branch executes. The implementation short-circuits evaluation, ensuring
only the selected branch is evaluated. This property becomes important when branches
contain errors or non-terminating computations.

*Function Application*: When applying a function to an argument, the interpreter:
1. Evaluates the function expression to a `v-fun` value
2. Evaluates the argument expression to a value
3. Extends the function's captured environment with a binding from the parameter
   to the argument value
4. Evaluates the function body in this extended environment

This mechanism implements proper lexical scoping and closure semantics.

### Error Handling

The interpreter raises exceptions for several error conditions:
- Unbound variables
- Type mismatches in operations (e.g., adding a string to a number)
- Non-boolean conditions in `if` expressions
- Attempting to apply non-function values

Errors follow Racket's convention: `(error <symbol> <message string>)`.

## Part II: Syntactic Desugaring

Desugaring translates extended syntax into core language constructs,
enabling language extensibility without modifying the interpreter.

### Extended Abstract Syntax

The extended syntax representation `Expr+` includes three sugar constructs:

```racket
(define-type Expr+
  ...
  (sugar-and [left : Expr+]
             [right : Expr+])
  (sugar-or [left : Expr+]
            [right : Expr+])
  (sugar-let [var : Symbol]
             [value : Expr+]
             [body : Expr+]))
```

### Desugaring Transformations

*Logical Conjunction (`and`)*: The expression `(and e1 e2)` desugars to:
```racket
(if e1 e2 false)
```
This encoding preserves short-circuit evaluation: if `e1` evaluates to `false`, `e2` is never evaluated.

*Logical Disjunction (`or`)*: The expression `(or e1 e2)` desugars to:
```racket
(if e1 true e2)
```
Similarly, if `e1` evaluates to `true`, `e2` is not evaluated.

*Variable Binding (`let`)*: The expression `(let (x e1) e2)` desugars to:
```racket
((lam x e2) e1)
```
This transformation reveals that `let` is syntactic sugar for immediate function
application. The variable `x` is bound to the value of `e1` in the scope of `e2`.
Importantly, `x` is not bound in `e1`, preventing recursive definitions.

<figure id="fig">
<img src="/assets/images/2024-04-01/meme.png" style="width:90.0%" />
<center>
<figcaption>
    <a href="https://cs.brown.edu/courses/csci1730/2020"> source </a>
</figcaption>
</center>
</figure><br>

### Desugaring Pipeline

The `desugar` function recursively transforms `Expr+` trees into `Expr` trees,
replacing all sugar constructs with their core equivalents while preserving program
semantics. This separation of concerns allows the interpreter to remain simple while
the language surface syntax can be extended freely.


## Part III: Static Type Checker

The type checker implements a static type system that verifies program correctness
before execution, catching type errors at compile time rather than runtime.

### Type Language

The type system includes:

```racket
(define-type Type
  (t-num)
  (t-bool)
  (t-str)
  (t-fun [arg-type : Type]
         [return-type : Type])
  (t-list [elem-type : Type]))
```

Function types `(t-fun arg-type return-type)` represent the type of single-argument
functions. List types `(t-list elem-type)` enforce homogeneity: all elements must have
the same type.

### Type Environment

The type checker maintains a type environment mapping identifiers to types:

```racket
(define-type-alias TEnv (Hashof Symbol Type))
```

This parallels the value environment used in the interpreter but operates at the type
level.

### Type Checking Rules

The `type-of` function implements bidirectional type checking using a post-order
traversal:

*Literals*: Numbers have type `(t-num)`, strings have type `(t-str)` and
booleans have type `(t-bool)`.

*Binary Operations*: 
- `+` and `num=` require both operands to have type `(t-num)`
- `++` and `str=` require both operands to have type `(t-str)`
- Type mismatches raise `TypeError` exceptions

*Conditionals*: The condition must have type `(t-bool)`. Both branches must have the
same type, which becomes the type of the entire `if` expression. This ensures type
safety regardless of which branch executes.

<div class="note">
{{ "**note:**
The requirement that both branches of an if expression must have the same type reveals
a fundamental principle of static type systems: they must reason about programs without
executing them.

This design decision addresses a critical challenge in static typing. When the type
checker analyzes an if expression, it cannot determine which branch will execute at
runtime. Which branch will be executed depends on the condition's value, which is
unknown during type checking. The type system must therefore ensure type safety for all
possible execution paths simultaneously. By requiring both branches to share a common
type, the type checker can confidently assign that type to the entire if expression,
regardless of which branch ultimately executes.

This constraint represents a deliberate tradeoff between expressiveness and safety.
Consider an expression like `(if condition 'hello' 42)`, this would be rejected by
miniPlait's type checker because the branches have incompatible types. In a dynamically
typed language, such an expression would be perfectly valid. The program would simply
return different types of values based on the condition. However, this flexibility
comes at the cost of deferring type checking to runtime, when type errors manifest as
runtime failures rather than compile-time rejections.

The restriction also illuminates what static type systems fundamentally cannot express
without additional machinery. To allow branches with different types while maintaining
static safety, we would need more sophisticated type constructs such as union types
(as found in TypeScript or OCaml), which explicitly represent a value of either type
A or type B. Without such features, the type system must be conservative, accepting
only programs where type safety can be guaranteed across all execution 
paths." | markdownify }}
</div>
<br>


*Lambda Abstractions*: Lambda expressions require type annotations on parameters:
```racket
(lam (x : Num) (+ x 1))
```
The type of this function is `(t-fun (t-num) (t-num))`. The type checker verifies the
body in an environment extended with the parameter's declared type.

*Function Application*: When type-checking `(f e)`, the type checker verifies:
1. The function `f` has type `(t-fun arg-type return-type)`
2. The argument `e` has type `arg-type`
3. The application has type `return-type`

*Variable Binding (`let`)*: The expression `(let (x e1) e2)` is type-checked by:
1. Computing the type `t1` of `e1`
2. Type-checking `e2` in an environment extended with `x : t1`
3. The type of the entire expression is the type of `e2`

<div class="note">
{{ "**note:**
The requirement for type annotations on lambda parameters reveals a fundamental
difference between what information is available during interpretation versus type
checking.

The type checker operates in a fundamentally different informational
context. It analyzes programs statically, before execution, which means it has access
only to the program's syntactic structure without any concrete values. When the type
checker encounters a lambda expression like `(lam x (+ x 1))`, it must verify that the
addition operation is valid. However, without knowing the type of parameter x, the type
checker cannot determine whether adding to x is a legitimate operation. The parameter
could represent a number, a string, a boolean, or a function and each possibility would
have different implications for type safety.

The type annotation resolves this informational gap. By explicitly declaring 
`(x : Num)`, the programmer provides the type checker with the information it
needs to proceed. The type checker can then verify that all operations on `x`
within the function body are consistent with numeric types. Furthermore, this
annotation enables the type checker to determine the function's complete type
signature." | markdownify }}
</div>
<br>

### List Operations

Lists introduce several type checking challenges:

*Empty Lists*: The empty list `(empty : t)` requires an explicit type annotation
because it has no elements to infer the type from. The annotation specifies the type
of elements that will eventually be added.

*List Construction*: The expression `(link x xs)` requires:
- `xs` has type `(t-list t)` for some type `t`
- `x` has type `t`
- The result has type `(t-list t)`

*List Deconstruction*:
- `(first xs)` requires `xs` has type `(t-list t)` and returns type `t`
- `(rest xs)` requires `xs` has type `(t-list t)` and returns type `(t-list t)`
- `(is-empty xs)` requires `xs` has type `(t-list t)` and returns type `(t-bool)`

These rules ensure type safety for list operations while maintaining homogeneity.

<div class="note">
{{ "**note:**
The type system is monomorphic: empty lists must be annotated with a specific element
type. A polymorphic type system would allow `empty` to have type `∀α. List α`, enabling
a single empty list to work with any element type. This limitation is acceptable for
pedagogical purposes but would be restrictive in a production language." 
| markdownify }}
</div>


## Part IV: Integrated Typed Interpreter

The typed interpreter combines desugaring, type checking and interpretation into a
unified evaluation pipeline.

### Evaluation Pipeline

Program execution follows four stages:

1. **Parsing**: S-expressions are parsed into `Expr+` (extended abstract syntax)
2. **Desugaring**: `Expr+` is transformed into `Expr` (core abstract syntax)
3. **Type Checking**: `type-of` verifies type correctness and returns the program's type
4. **Interpretation**: If type checking succeeds, `interp` evaluates the program to
    produce a value

This pipeline ensures that only well-typed programs are executed, catching many errors
before runtime.

### Error Delegation

The introduction of static type checking eliminates several classes of runtime errors:

*Errors Caught by Type Checker*:
- Unbound variable references
- Type mismatches in binary operations
- Non-boolean `if` conditions
- Type mismatches in function applications
- List operation type violations

*Errors Remaining at Runtime*:
- Accessing elements of empty lists (`first` and `rest` on empty lists)

The type system cannot distinguish between empty and non-empty lists because they share
the same type `(t-list t)`. This represents a fundamental limitation: proving that a
list is non-empty requires dependent types or refinement types, which are beyond the
scope of this type system.

### Semantic Changes

Type checking introduces several semantic restrictions:

1. *List Homogeneity*: All list elements must have the same type.
    Previously valid programs like:
   ```racket
   (link 0 (link "hello" (empty : Str)))
   ```
   are now rejected.

2. *Conditional Branch Uniformity*: Both branches of an `if` expression must have the
    same type:
   ```racket
   (if true "string" 5)
   ```
   is rejected because the branches have different types.

3. *Type Annotation Requirements*: Lambda abstractions and empty lists require explicit
    type annotations.

These restrictions trade expressiveness for type safety, 
a common tradeoff in statically typed languages.

### Modified `let` Syntax

The integrated interpreter requires type annotations on `let` bindings:

```racket
(let (x 4 : Num) (+ x x))
```
<br>

## Part V: Lazy Evaluation

The final phase implements lazy evaluation, fundamentally changing how and when
expressions are evaluated.

### Lazy Evaluation Semantics

Lazy evaluation (also called call-by-need) delays computation until values are actually
needed. When a function is called, its argument is not evaluated immediately. Instead,
the unevaluated argument (a "suspension" or "thunk") is passed to the function. The
argument is evaluated only when the function's body actually uses it.

This evaluation strategy enables several programming patterns impossible with eager
evaluation:
- Infinite data structures
- Improved modularity through separation of data generation from consumption
- Potential performance improvements by avoiding unnecessary computations

### Suspension and Forcing

The implementation uses suspension to represent delayed computations:

```racket
(define-type Value
  ...
  (v-suspension [expr : Expr]
                [env : Env]))
```

A suspension packages an expression with its environment. When the value is needed, it
is "forced": the expression is evaluated in its environment to produce a concrete value.

*Strictness Points*: Certain language constructs force evaluation:
- Binary operators force both operands
- `if` forces its condition and the selected branch
- `first` forces the head element of a list
- `rest` forces evaluation to determine if the list is empty, 
   but does not force remaining elements
- The top-level `eval` forces the final result for display

Forcing is shallow: evaluating a list to determine it is non-empty does not
recursively force its elements.

### Recursive Definitions with `rec`

Lazy evaluation naturally supports recursive definitions through the `rec` construct:

```racket
(rec (name expr : type) body)
```

The variable `name` is bound in both `expr` and `body`, enabling self-referential
definitions. This is essential for defining recursive functions that work on infinite
structures:

```racket
(rec (nats-from (lam (n : Num)
                     (link n (nats-from (+ 1 n))))
                : (Num -> (List Num)))
  (nats-from 0))
```

This defines an infinite list of natural numbers. With eager evaluation, this would
loop forever. With lazy evaluation, only the elements actually accessed are computed.

### Infinite Streams Example

Consider a function `take` that extracts the first `n` elements from a stream:

```racket
(rec (take (lam (n : Num)
            (lam (s : (List Num))
             (if (num= n 0)
                 (empty : Num)
                 (link (first s)
                       ((take (+ n -1)) (rest s))))))
           : (Num -> ((List Num) -> (List Num))))
  ((take 3) (nats-from 0)))
```

This evaluates to `(link 0 (link 1 (link 2 (empty : Num))))`. The key insight: although
`nats-from` generates an infinite list, only the first three elements are computed
because `take` only accesses three elements.

With eager evaluation, `(nats-from 0)` would attempt to construct the entire infinite
list before passing it to `take`, causing non-termination. Lazy evaluation decouples
data generation from consumption, evaluating only what is needed.

### Performance Considerations

While lazy evaluation enables elegant programming patterns,
it introduces certain disadvantages:
- Each suspension requires memory allocation
- Debugging becomes more difficult because evaluation order is implicit

In practice, lazy evaluation is most valuable when it enables computations that would
be impossible or impractical with eager evaluation, such as working with infinite
structures or avoiding unnecessary expensive computations.

## References

1. Krishnamurthi, S. (2020). *Programming Languages: Application and Interpretation (PLAI)*. Version 3.2.5. Brown University. Available at: https://www.plai.org/3/5/plai-v325.pdf

2. Krishnamurthi, S. (2020). CSCI 1730: Programming Languages. Brown University. Course materials available at: https://cs.brown.edu/courses/csci1730/2020/


### Appendix A: Grammar Summary

**miniPlait Grammar (Eager, Untyped)**:
```
<expr> ::= <num> | <string> | <var>
         | true | false
         | (+ <expr> <expr>)
         | (++ <expr> <expr>)
         | (num= <expr> <expr>)
         | (str= <expr> <expr>)
         | (if <expr> <expr> <expr>)
         | (lam <var> <expr>)
         | (<expr> <expr>)
```

**With Syntactic Sugar**:
```
<expr> ::= ... (as above)
         | (and <expr> <expr>)
         | (or <expr> <expr>)
         | (let (<var> <expr>) <expr>)
```

**Typed miniPlait Grammar**:
```
<expr> ::= ... (as above)
         | (lam (<var> : <type>) <expr>)
         | (let (<var> <expr> : <type>) <expr>)
         | (first <expr>)
         | (rest <expr>)
         | (is-empty <expr>)
         | (empty : <type>)
         | (link <expr> <expr>)

<type> ::= Num | Str | Bool
         | (List <type>)
         | (<type> -> <type>)
```

**Lazy miniPlait Additional Syntax**:
```
<expr> ::= ... (as above)
         | (rec (<var> <expr> : <type>) <expr>)
```
