---
name: python-modern-design-patterns
description: >
  Modern Python design patterns, typing idioms, and language mechanics.
  Use whenever the user asks about or is working with: Protocol classes and
  @runtime_checkable, asynccontextmanager exception handling, weakref/WeakMethod
  behavior, composition roots, ABC vs Protocol selection, or the Builder pattern
  vs dataclass. Also triggers for any question involving structural subtyping,
  garbage collection timing, or context manager suppression semantics.
version: 0.1.0
---

# Modern Python Design Patterns

## Corrections First

Three areas where LLMs commonly produce incorrect or incomplete answers.

### 1. `@runtime_checkable` Protocol — isinstance() raises TypeError WITHOUT the decorator

**Error:** Saying isinstance() with a Protocol always works, or that it checks signatures.

**Fact:** Without `@runtime_checkable`, `isinstance(obj, Protocol)` raises `TypeError: Protocols cannot be used with isinstance()`. With the decorator, it works — but only checks **method existence**, not signatures or return types.

```python
from typing import Protocol, runtime_checkable

# WRONG: bare Protocol raises TypeError
class Closable(Protocol):
    def close(self) -> None: ...

isinstance(open('/dev/null'), Closable)
# TypeError: Protocols cannot be used with isinstance()

# RIGHT: decorate with @runtime_checkable
@runtime_checkable
class Closable(Protocol):
    def close(self) -> None: ...

isinstance(open('/dev/null'), Closable)  # True — method exists

# Even with @runtime_checkable — signatures are NOT checked:
@runtime_checkable
class Adder(Protocol):
    def add(self, x: int, y: int) -> int: ...

class Broken:
    def add(self, x: str, y: str) -> str: ...  # wrong types!

isinstance(Broken(), Adder)  # True — only existence checked

# hasattr() for structural checks is often safer than isinstance():
hasattr(obj, 'add') and callable(getattr(obj, 'add'))
```

**PEP 544** says: *"A protocol can be used as a second argument in isinstance() and issubclass() only if it is explicitly opt-in by @runtime_checkable decorator."*

---

### 2. `@asynccontextmanager` — try/except around yield DOES suppress exceptions

**Error:** Claiming that try/except inside an `@asynccontextmanager`-decorated generator cannot suppress exceptions, or conflating it with `__aexit__` returning `True`.

**Fact:** The `try/except` around `yield` inside the generator **does** suppress exceptions. The `__aexit__` that contextlib builds calls `generator.athrow(exc_type, ...)` to inject the exception into the suspended generator. If the generator has `try: yield ... except: ...` wrapping the yield, the exception is caught there and the generator completes normally — `StopAsyncIteration` propagates to `__aexit__` and the exception is suppressed.

```python
from contextlib import asynccontextmanager

# CORRECT suppression: try/except around yield catches the injected exception
@asynccontextmanager
async def safe_resource():
    try:
        yield acquire()
    except ConnectionError:
        # exception is suppressed — async with block sees no error
        yield None  # ⚠️ Multiple yields NOT allowed; this is the BROKEN pattern
        # WRONG: you cannot yield twice; this raises RuntimeError

# CORRECT suppression (single yield, exception caught):
@asynccontextmanager
async def safe_resource():
    resource = None
    try:
        resource = acquire()
        yield resource
    except ConnectionError:
        resource = None  # handle, then fall through to finally
    finally:
        if resource:
            release(resource)

# __aexit__ returning True suppresses without touching generator internals:
class SuppressingCM:
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        return True  # suppresses exception — but this is NOT @asynccontextmanager

# @asynccontextmanager: the generator's except block suppresses
# __aexit__ returning True: suppresses at the async with level (different mechanism)
```

**Key distinction:** `@asynccontextmanager`'s suppression comes from the generator's internal `try/except` catching the exception that `__aexit__` injects via `athrow()`. This is specific to how `@asynccontextmanager` works vs. a manual `__aexit__` returning `True`.

---

### 3. `WeakMethod` — plain `weakref.ref()` to bound method is immediately dead

**Error:** Claiming you can use `weakref.ref()` directly on a bound method, or that GC timing is the same across all implementations.

**Fact:** Bound method objects (`obj.method`) are **temporary** — created on-the-fly when accessed and freed when the last reference is dropped. A plain `weakref.ref(obj.method)` captures this ephemeral object and finds it already dead. `WeakMethod` works around this by holding weak references to the underlying instance and function separately, reconstructing the bound method on demand.

```python
import weakref

class Subscriber:
    def notify(self, msg: str) -> None:
        print(msg)

sub = Subscriber()

# WRONG: bound method is already dead
ref = weakref.ref(sub.notify)
print(ref())  # None — immediately dead

# RIGHT: WeakMethod holds refs to instance+function separately
from weakref import WeakMethod

ref = WeakMethod(sub.notify)
print(ref())  # <bound method ...> — still alive
print(ref().notify("hello"))  # hello

# CPython: deterministic destruction for objects with no cycles
# refcount hits 0 → __del__ called immediately (no GC cycle needed)
# PyPy/Jython: non-deterministic — only the tracing GC can collect
# WeakSet with objects defining __eq__/__hash__ uses equality, not identity:
class Ident:
    def __eq__(self, other):
        return id(self) == id(other)  # forces identity semantics via equality
    def __hash__(self):
        return hash(id(self))

# subscribe(Newsletter("Alice")) with no external ref:
# CPython: refcount drops to 0, object destroyed immediately, weakref gone
# PyPy: object survives until next GC sweep (non-deterministic)
```

---

## Reference Sections

### 4. Composition Root

The Composition Root is the single place where object graphs are assembled. All `isinstance()` checks and concrete-type knowledge lives here — never in business logic.

```python
# bootstrap() is the composition root
def bootstrap() -> Application:
    repo = SqlAlchemyRepo()        # concrete type known here
    service = UserService(repo)     # interface only outside here
    return Application(service)

# Business logic knows only the interface
class Application:
    def __init__(self, service: UserServiceProto) -> None:
        self._svc = service  # no isinstance checks

# This localizes all concrete-type coupling to one function
```

### 5. Protocol vs ABC Decision Rule

| Use Protocol when... | Use ABC when... |
|---|---|
| You don't own the class | You own the class |
| Structural duck-typed interface | You need `isinstance()` checks |
| No default implementations needed | You want to provide default method implementations |
| Framework requires it (e.g., `Iterable`) | Framework integration requires ABC |

```python
# Protocol — for code you don't own and don't want to modify
@runtime_checkable
class Closable(Protocol):
    def close(self) -> None: ...

# ABC — when you own it and need isinstance or defaults
from abc import ABC, abstractmethod

class BaseHandler(ABC):
    @abstractmethod
    def handle(self) -> None: ...
    def default_log(self, msg: str) -> None:  # default impl
        print(msg)
```

### 6. Builder vs dataclass

Use `dataclass(**kwargs)` when fields are independent and don't need cross-field validation. Use `Builder` when you need:

- **Cross-field validation** (field A + field B must satisfy a constraint)
- **Mutual exclusion** (you can't set both A and B)
- **Post-build immutability** (object must be frozen after construction)
- **Optional/mandatory field tracking** (you need to know which fields were set)

```python
from dataclasses import dataclass, field

# dataclass is enough:
@dataclass
class Point:
    x: float
    y: float
    # simple, independent fields — no builder needed

# Builder needed:
@dataclass
class Connection:
    host: str = ""
    port: int = 0
    ssl: bool = False
    _built: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.host and self.port and not self.ssl:
            raise ValueError("remote connections require SSL")
        if self._built:
            raise RuntimeError("Connection is immutable after build()")

class ConnectionBuilder:
    _host: str = ""
    _port: int = 0
    _ssl: bool = False

    def host(self, h: str) -> ConnectionBuilder: self._host = h; return self
    def port(self, p: int) -> ConnectionBuilder: self._port = p; return self
    def ssl(self, s: bool) -> ConnectionBuilder: self._ssl = s; return self
    def build(self) -> Connection:
        c = Connection(self._host, self._port, self._ssl)
        c._built = True
        return c
```

---

## Running Examples

Test the corrections directly:

```bash
python3 -c "
from typing import Protocol, runtime_checkable

@runtime_checkable
class Adder(Protocol):
    def add(self, x: int, y: int) -> int: ...

class Broken:
    def add(self, x: str, y: str) -> str: ...

print(isinstance(Broken(), Adder))  # True — only existence checked
"

python3 -c "
from weakref import ref, WeakMethod

class S:
    def meth(self): pass

s = S()
print('plain ref:', ref(s.meth)())       # None — immediately dead
print('WeakMethod:', WeakMethod(s.meth)())  # <bound method ...>
"
```
