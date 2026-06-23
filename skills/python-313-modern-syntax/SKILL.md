---
name: python-313-modern-syntax
description: >
  Python 3.13 modern syntax and typing reference. Use this skill when answering
  questions about or applying Python 3.13+ features including: PEP 695 type parameter
  syntax and runtime behavior (TypeVar, __type_params__, lazy evaluation, type statement),
  PEP 667 locals() semantics changes, PEP 594 module removals, PEP 696 type parameter
  defaults, PEP 742 TypeIs vs TypeGuard, typing.ReadOnly, warnings.deprecated,
  typing.Never vs NoReturn, sys._is_gil_enabled(), or any Python 3.13 typing changes.
  Trigger aggressively —LLMs have persistent misconceptions about all of these topics.
---

# Python 3.13 Modern Syntax & Typing — Authoritative Reference

This skill encodes ground-truth facts about Python 3.13+ features where LLM knowledge is
commonly incorrect. Each section below states the verified facts, provides runnable code
examples, and explicitly calls out the wrong answers you must avoid.

---

## 1. PEP 695: TypeVar Runtime Behavior via `__type_params__`

### CORE FACTS

**`class Foo[T]:` DOES create TypeVar objects accessible via `__type_params__` at runtime.**

This is the most persistent LLM hallucination: that TypeVars created by the new syntax
"don't exist at runtime" or are "only for type checkers." They are real objects.

```python
# Python 3.12+ — this works at runtime
class Foo[T]:
    pass

# TypeVar objects ARE accessible via __type_params__
print(Foo.__type_params__)          # tuple: (T,)
print(len(Foo.__type_params__))     # 1
print(Foo.__type_params__[0].__class__.__name__)  # 'TypeVar'
```

**`__type_params__` replaces `__type_variables__`.** The old attribute name is gone.

```python
def func[T](x: T) -> T:
    return x

print(func.__type_params__)  # tuple: (T,)
print(func.__type_params__[0].__name__)  # 'T'
```

### `__type_params__` Structure

| Context                     | `__type_params__` contains                         |
| --------------------------- | -------------------------------------------------- |
| Generic class               | Tuple of `TypeVar`, `ParamSpec`, or `TypeVarTuple` |
| Generic function            | Same                                               |
| Type alias (`type X = ...`) | Same                                               |

### Lazy Evaluation of Bounds and Constraints

**TypeVar bounds and constraints are lazily evaluated** — not at class/function definition
time, but only when `. __bound__` or `.__constraints__` is accessed.

```python
# This does NOT raise ZeroDivisionError at definition time
class Broken[T: 1/0]:
    pass

# Only when you access __bound__ does the error surface
try:
    print(Broken.__type_params__[0].__bound__)
except ZeroDivisionError:
    print("Lazy evaluation: error only on access")
```

### WRONG ANSWERS TO AVOID

- ❌ "TypeVar objects from `class Foo[T]:` don't exist at runtime"
- ❌ "Use `__type_variables__` instead of `__type_params__`"
- ❌ "Bounds are evaluated eagerly at class definition time"
- ❌ "TypeVars from new syntax are only visible to type checkers"

---

## 2. PEP 667: `locals()` Semantics Change in Python 3.13

### THE REAL CHANGE (and the wrong story many LLMs tell)

**The old behavior was NOT that mutations to `locals()` propagated back.** That was always
undefined behavior / implementation-specific. CPython's behavior was inconsistent and
broke under debuggers, tracing, or concurrent execution.

**What actually changed in Python 3.13 (PEP 667):**

1. Each call to `locals()` in an **optimized scope** (functions, generators, coroutines,
   comprehensions, generator expressions) now returns a **fresh snapshot dict**.
2. Mutations to the returned dict **never** propagate back to the actual frame locals.
3. `frame.f_locals` now returns a **write-through proxy** in optimized scopes, enabling
   debuggers to reliably update local variables.
4. For `exec()`/`eval()` in optimized scopes: changes run against an independent
   snapshot and are **never visible** to subsequent `locals()` calls. Pass an explicit
   namespace to see the changes.
5. Module/class scope behavior is **unchanged**.

```python
# Python 3.13+ behavior in a function
def f():
    x = 1
    d1 = locals()
    d2 = locals()
    d1['x'] = 999  # mutate snapshot
    print(x)        # still 1 — mutation did NOT propagate
    print(d1 == d2) # False — they are different snapshot dicts

# What changed: previously d1 and d2 might have been the same dict
# (implementation artifact), and the mutation *might* have propagated.
# Now it's consistently a fresh snapshot each time.
```

### `exec()` With Explicit Namespace (Required for Seeing Changes)

```python
def g():
    namespace = {}
    exec('x = 42', namespace)
    print(namespace)  # {'x': 42} — must use explicit namespace
    print(locals())   # does NOT include x — it's in the explicit namespace
```

### WRONG ANSWERS TO AVOID

- ❌ "The old behavior was that mutations to `locals()` propagated back reliably"
- ❌ "`locals()` in a function returns the same dict on every call (old behavior was consistent)"
- ❌ "PEP 667 changed module-level `locals()` behavior"
- ❌ "`exec()` with implicit namespace now reliably updates local variables"
- ❌ "The change is that `frame.f_locals` now returns a dict instead of a proxy" (it's the reverse)

---

## 3. PEP 594: Complete Module Removal List

### The Canonical 19 Modules Removed in Python 3.13

These were deprecated in Python 3.11 and removed in Python 3.13:

| Module        | Replacement                                                                 |
| ------------- | --------------------------------------------------------------------------- |
| `aifc`        | `shutil` + `wave`, or third-party `aifc`                                    |
| `audioop`     | `ctypes`, `sounddevice`, `pyaudio`                                          |
| `cgi`         | `wsgiref`, `fastapi`, `flask`                                               |
| `cgitb`       | `traceback`, `logging`                                                      |
| `chunk`       | `struct`, `wave`                                                            |
| `crypt`       | `cryptography`, `passlib`, `bcrypt`, `argon2-cffi`                          |
| `imghdr`      | `filetype`, `puremagic`, `python-magic`                                     |
| `mailcap`     | `mimetypes`                                                                 |
| `msilib`      | `msidb`, `python-msi`                                                       |
| `nis`         | `nslcd`, system LDAP config                                                 |
| `nntplib`     | `imaplib`, `smtplib`, or custom NNTP client                                 |
| `ossaudiodev` | `pygame`, `sounddevice`                                                     |
| `pipes`       | `subprocess`                                                                |
| `sndhdr`      | `filetype`, `puremagic`                                                     |
| `spwd`        | `python-pam`, system PAM config                                             |
| `sunau`       | `wave`, `aifc`                                                              |
| `telnetlib`   | `telnetlib3`, `Exscript`                                                    |
| `uu`          | `binascii` (but note `binascii` functions were also deprecated — see below) |
| `xdrlib`      | `xmlrpc.client`, `xdr` third-party libs                                     |

### Note on `uu` and `binascii`

The `uu` codec and two related `binascii` functions (`binascii.b2a_uu`, `binascii.a2b_uu`)
were **not removed** — they were deprecated in 3.13 and scheduled for removal in 3.15.

### What Was NOT Removed in 3.13 (Different Versions)

- `asyncore`, `asynchat`, `smtpd` — removed in Python **3.12**, not 3.13
- `distutils` — removed in Python **3.12**
- `imp` — removed in Python **3.12**
- `lib2to3` and `2to3` — removed in Python **3.13** (not part of PEP 594)

### WRONG ANSWERS TO AVOID

- ❌ Quoting any module not in the 19 above as removed in 3.13
- ❌ Saying `crypt` was replaced by any specific third-party lib (PEP just suggests options)
- ❌ Including `asyncore`, `asynchat`, `smtpd` in the 3.13 removal list
- ❌ Saying `uu` was fully removed (it was deprecated, not removed)

---

## 4. PEP 742: `TypeIs` vs `TypeGuard` Negative Branch Narrowing

### The Key Difference

**`TypeIs` narrows in BOTH branches. `TypeGuard` only narrows in the `True` branch.**

This is the critical distinction that many LLMs get wrong, especially about the negative
(else) branch.

### Formal Specification (PEP 742)

For a function `def f(x: A) -> TypeIs[B]`:

- **Positive branch** (condition is True): type is narrowed to `A ∧ B` (intersection)
- **Negative branch** (condition is False): type is narrowed to `A ∧ ¬B` (exclusion)

For `TypeGuard[B]`:

- **Positive branch**: type is narrowed to exactly `B`
- **Negative branch**: type is **NOT narrowed at all** — it stays as `A`

### Minimal Example

```python
from typing import TypeIs

def is_str_list(val: list[object]) -> TypeIs[list[str]]:
    return all(isinstance(x, str) for x in val)

def process(val: list[object]) -> None:
    if is_str_list(val):
        reveal_type(val)  # list[str] — positive narrows to intersection
    else:
        reveal_type(val)  # list[object] — negative narrows to exclusion of list[str]
        # Note: NOT narrowed to something narrower; exclusion means it could still
        # contain non-str objects (the original type minus what TypeIs would match)
```

### What `TypeGuard` Does Differently

```python
from typing import TypeGuard

def is_str_list_guard(val: list[object]) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in val)

def process_guard(val: list[object]) -> None:
    if is_str_list_guard(val):
        reveal_type(val)  # list[str] — positive narrows to exactly list[str]
    else:
        reveal_type(val)  # list[object] — NO narrowing at all in else branch
```

### `TypeIs` Requires Subtype Relationship

`TypeIs[B]` requires that `B` is a **subtype of** the input type `A`. `TypeGuard` has
no such restriction — you can narrow `list[object]` to `list[str]` with `TypeGuard`.

```python
from typing import TypeIs

# TypeGuard CAN do this:
def to_str_list(v: list[object]) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in v)

# TypeIs CANNOT do this — list[str] is not a subtype of list[object]
# (variance: list is invariant)
# def to_str_list_is(v: list[object]) -> TypeIs[list[str]]:  # ERROR
#     return all(isinstance(x, str) for x in v)
```

### WRONG ANSWERS TO AVOID

- ❌ "TypeIs only narrows in the positive branch" (it narrows in both)
- ❌ "TypeGuard narrows in the negative branch too"
- ❌ "TypeIs and TypeGuard are functionally identical"
- ❌ "TypeIs requires the narrowed type to NOT be a subtype of the input"
- ❌ "The negative branch of TypeIs narrows to the exact complement of the narrowed type"
  (it narrows to A ∧ ¬B, which is an approximation — type checkers may not express this precisely)

---

## 5. PEP 695: `type` Statement Lazy Evaluation

### The `type` Statement Creates `TypeAliasType` Instances

```python
# Python 3.12+ type statement
type IntOrStr = int | str

# At runtime: creates a TypeAliasType instance
print(IntOrStr.__class__.__name__)  # 'TypeAliasType'
print(IntOrStr.__name__)            # 'IntOrStr'
print(IntOrStr.__value__)           # int | str (lazily evaluated)
print(IntOrStr.__parameters__)      # ()
```

### Generic Type Alias

```python
type ListOrSet[T] = list[T] | set[T]

print(ListOrSet.__parameters__)     # (T,)
print(ListOrSet.__type_params__)    # (T,) — same as __parameters__ for type aliases
```

### Lazy Evaluation of Type Alias Values

Bound expressions in type parameters and type alias values are **lazily evaluated**:

```python
# This does NOT raise at definition time
type EvilAlias = 1 / 0

# Only when you access __value__ does the error surface
try:
    EvilAlias.__value__
except ZeroDivisionError:
    print("Lazy evaluation confirmed: error only on __value__ access")
```

### Lazy Evaluation of TypeVar Bounds

```python
# This does NOT raise at definition time
class Foo[T: 1/0]:
    pass

# Error only on __bound__ access
try:
    Foo.__type_params__[0].__bound__
except ZeroDivisionError:
    print("Bound is lazily evaluated")
```

### `__type_params__` Attribute Structure

| Object                | `.__type_params__` contains                                   |
| --------------------- | ------------------------------------------------------------- |
| `class Foo[T]:`       | `(T,)` — a tuple of `TypeVar` (or `ParamSpec`/`TypeVarTuple`) |
| `def func[T](...):`   | `(T,)`                                                        |
| `type Alias = ...`    | `()` — non-generic aliases have empty `__type_params__`       |
| `type Alias[T] = ...` | `(T,)`                                                        |

### WRONG ANSWERS TO AVOID

- ❌ "The `type` statement evaluates the right-hand side eagerly"
- ❌ "`TypeAliasType.__value__` is always pre-computed at definition time"
- ❌ "`type` aliases don't have `__type_params__`"
- ❌ "TypeVar bounds are evaluated at class definition time, not lazily"

---

## 6. PEP 696: Type Parameter Defaults

### Syntax in Bracket Form

```python
# TypeVar with default
class Foo[T = int]:
    pass

# ParamSpec with default
type Handler[**P = (int, str)] = Callable[P, None]

# TypeVarTuple with default
class Container[*Ts = *tuple[str, int]]:
    pass
```

### Ordering Rules (Compiler-Enforced)

1. **Parameters without defaults cannot follow parameters with defaults**
2. **A TypeVar immediately following a TypeVarTuple cannot have a default** (ambiguous:
   does a type argument bind to the TypeVarTuple or the defaulted TypeVar?)

```python
# Valid: default follows non-default
class Valid1[T, U = str]:
    pass

# SyntaxError: non-default after default
# class Invalid[T = str, U]:  # Error
#     pass

# SyntaxError: TypeVar with default immediately after TypeVarTuple
# class Invalid[*Ts = *tuple[int], T = str]:  # Error
#     pass

# Valid: ParamSpec with default after TypeVarTuple with default
# (no ambiguity — ParamSpec is always contravariant position)
class Valid2[*Ts = *tuple[int], **P = ()]:
    pass
```

### Runtime Attribute: `.__default__`

```python
from typing import TypeVar, NoDefault

T = TypeVar('T')
print(T.__default__)  # None — no default

U = TypeVar('U', default=int)
print(U.__default__)  # <class 'int'>

# Note: if default=None is explicitly passed, __default__ is NoneType (the class), not None
V = TypeVar('V', default=None)
print(V.__default__)  # <class 'NoneType'> — not None
```

### With PEP 695 Function Syntax

```python
def foo[T = int, U = str](x: T, y: U) -> tuple[T, U]:
    return (x, y)

# Without specifying — uses defaults
result = foo(1, "hello")
# result is (int, str) — both defaulted

# Partial specification
result2 = foo[str](1, "hello")  # T=str, U defaults to str
```

### WRONG ANSWERS TO AVOID

- ❌ "Parameters with defaults can follow parameters without defaults"
- ❌ "A TypeVar can have a default immediately after a TypeVarTuple"
- ❌ "`TypeVar.__default__` returns `None` when `default=None` is passed"
  (it returns `NoneType`, the class)
- ❌ "PEP 696 defaults work in the old `TypeVar()` constructor syntax"

---

## Supplementary Topics

### `typing.ReadOnly` (PEP 705, Python 3.13+)

Mark TypedDict items as read-only (not mutable at runtime — only enforced by type checkers):

```python
from typing import TypedDict, ReadOnly

class Movie(TypedDict):
    title: ReadOnly[str]  # Cannot be modified
    year: int              # Can be modified

def mutate(m: Movie) -> None:
    m["year"] = 1999      # OK
    m["title"] = "Matrix" # Type checker error, but NOT runtime error
```

In `typing_extensions` since 4.9.0. In stdlib `typing` since 3.13.

### `warnings.deprecated` (PEP 702, Python 3.13+)

```python
from warnings import deprecated

@deprecated("Use B instead")
class A:
    pass

@deprecated("Use g instead")
def f():
    pass

# Calling a deprecated function emits DeprecationWarning at RUNTIME
a = A()  # DeprecationWarning: Use B instead
f()      # DeprecationWarning: Use g instead

# __deprecated__ attribute is set
print(A.__deprecated__)  # "Use B instead"
```

**Runtime behavior:** By default (category=DeprecationWarning), the decorator emits a
`DeprecationWarning` at runtime on every call/instantiation. Set `category=None`
to suppress the runtime warning and only communicate the deprecation to type checkers.

Works with `@typing.overload` when decorator is **after** `@overload`:

```python
from warnings import deprecated
from typing import overload

@overload
@deprecated("int support is deprecated")
def g(x: int) -> int: ...
@overload
def g(x: str) -> int: ...
```

### `typing.Never` vs `typing.NoReturn`

Both represent the **bottom type** (no values). They are **interchangeable** at the type
system level — type checkers treat them identically.

- `NoReturn` — traditional, used for **return annotations** of functions that never return
- `Never` — added in Python 3.11, used for **general bottom type** (e.g., in argument
  positions, variable annotations, for `assert_never()`)

```python
from typing import Never, NoReturn, assert_never

# Both work for sys.exit()-style functions
def stop() -> NoReturn:
    raise RuntimeError("stop")

def halt() -> Never:
    raise RuntimeError("halt")

# Never for argument positions
def exhaustive_check(x: int | str) -> None:
    match x:
        case int():
            print("int")
        case str():
            print("str")
        case _:
            assert_never(x)  # x is narrowed to Never here
```

### `sys._is_gil_enabled()` (Python 3.13+)

Check whether the GIL is currently enabled in a free-threaded build:

```python
import sys

if sys.version_info >= (3, 13):
    gil_enabled = sys._is_gil_enabled()
    print(f"GIL enabled: {gil_enabled}")
else:
    print("Python < 3.13 — GIL always enabled")

# For free-threaded builds:
# - Always True in default (non-free-threaded) builds
# - May be True or False in free-threaded builds depending on runtime state
```

Related: `sysconfig.get_config_var("Py_GIL_DISABLED") == 1` indicates a free-threaded
build supports the GIL being disabled.

---

## Quick Reference: Common LLM Mistakes This Skill Prevents

| Topic      | Common LLM Mistake                                     | Correct Fact                      |
| ---------- | ------------------------------------------------------ | --------------------------------- |
| PEP 695    | "TypeVars from `class Foo[T]:` don't exist at runtime" | They exist via `.__type_params__` |
| PEP 695    | "Use `__type_variables__`"                             | It's `__type_params__` only       |
| PEP 667    | "Old behavior was mutations propagated back"           | That was always undefined         |
| PEP 667    | "Free-threaded builds are the default"                 | No — default build always has GIL |
| PEP 594    | Listing 20+ modules                                    | Exactly 19 modules removed        |
| PEP 594    | Including `asyncore`/`smtpd` in 3.13 list              | Those were removed in 3.12        |
| TypeIs     | "TypeIs only narrows in positive branch"               | Both branches narrow              |
| TypeGuard  | "TypeGuard narrows in negative branch"                 | No narrowing in else              |
| PEP 696    | "TypeVar with default can follow TypeVarTuple"         | Not allowed                       |
| PEP 696    | "`TypeVar(default=None).__default__` is `None`"        | It's `NoneType`                   |
| `locals()` | "Module-level `locals()` behavior changed"             | Unchanged                         |
