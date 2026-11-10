"""One adapter instance serves every request, so it must not hold one.

A voice is loaded once and then synthesizes for whoever asks, concurrently.
An adapter that stashes the current request's reference clip, transcription
or seed on ``self`` hands it to the next caller — the wrong voice, delivered
as a perfectly good WAV, and nothing about it looks like a failure.

The check is on the source: every write to ``self`` in a registered adapter
must either live in a setup method (``BaseOnnxAdapter.SETUP_METHODS``, which
run once from the voice config) or be named attribute by attribute in that
adapter's ``MEMOIZED_WRITES``. Naming it there is a claim that the value is
safe to serve to the next request — a property of the model, or a cache keyed
on what produced it.
"""
import ast
import inspect
import unittest
from typing import Dict, Optional, Set

from phoonnx.engines import _REGISTRY
from phoonnx.engines.base import BaseOnnxAdapter


# Container methods that mutate in place. ``self.x = self.x + [y]`` rebinds
# and is caught as an assignment; ``self.x.append(y)`` is not.
_MUTATORS = frozenset({"append", "extend", "add", "update", "insert",
                       "setdefault", "pop", "clear", "discard", "remove",
                       "popitem", "sort", "move_to_end"})


def _adapter_classes():
    """Every registered adapter class plus its phoonnx-defined bases."""
    classes = {}
    for name, cls in sorted(_REGISTRY.items()):
        for klass in cls.__mro__:
            if klass is BaseOnnxAdapter:
                continue
            if klass.__module__.startswith("phoonnx."):
                classes[f"{klass.__module__}.{klass.__qualname__}"] = klass
    return classes


def _self_writes(klass) -> Dict[str, Set[str]]:
    """``{method name: {attribute written}}`` for every write to ``self``.

    Plain assignment is the easy case. The ones that matter are the quiet
    spellings: an attribute assigned as part of a tuple, ``setattr(self, ...)``
    with the name in a variable, and mutation of a container already on
    ``self`` — ``self.cache[key] = x``, ``self.seen.add(x)``, ``self.x += 1``.
    Every one of them leaves this request's state where the next request
    reads it, and none of them is an ``ast.Assign`` to an ``ast.Attribute``.
    """
    tree = ast.parse(inspect.getsource(klass).lstrip())
    written: Dict[str, Set[str]] = {}

    def record(method: str, attr: str) -> None:
        written.setdefault(method, set()).add(attr)

    def targets_of(node):
        if isinstance(node, ast.Assign):
            return node.targets
        if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            return [node.target]
        return []

    def flatten(target):
        """Assignment targets, unwrapping tuple/list/starred destructuring."""
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                yield from flatten(element)
        elif isinstance(target, ast.Starred):
            yield from flatten(target.value)
        else:
            yield target

    def self_attr(node) -> Optional[str]:
        """The attribute name when ``node`` is ``self.X`` or ``self.X[...]``."""
        while isinstance(node, ast.Subscript):
            node = node.value
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"):
            return node.attr
        return None

    for method in tree.body[0].body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(method):
            for target in targets_of(node):
                for element in flatten(target):
                    attr = self_attr(element)
                    if attr:
                        record(method.name, attr)

            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # setattr(self, "name", value) / setattr(self, name_var, value)
            if (isinstance(func, ast.Name) and func.id == "setattr"
                    and node.args
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "self"):
                name = node.args[1] if len(node.args) > 1 else None
                if isinstance(name, ast.Constant) and isinstance(name.value, str):
                    record(method.name, name.value)
                else:
                    # The attribute is chosen at runtime, so which one is
                    # written cannot be known from the source. Reported under
                    # a marker naming the expression, which an adapter has to
                    # declare verbatim — a dynamic write is exactly the kind
                    # that should be argued for out loud.
                    record(method.name,
                           "*" + (ast.unparse(name) if name else "?"))
            # self.container.append/add/update/extend/setdefault(...)
            elif (isinstance(func, ast.Attribute)
                  and func.attr in _MUTATORS):
                attr = self_attr(func.value)
                if attr:
                    record(method.name, attr)

    return written


class TestAdaptersAreStatelessBetweenRequests(unittest.TestCase):

    def test_no_adapter_writes_undeclared_state(self):
        offenders = []
        for name, klass in _adapter_classes().items():
            for method, attrs in _self_writes(klass).items():
                if method in klass.SETUP_METHODS:
                    continue
                allowed = klass.MEMOIZED_WRITES.get(method, frozenset())
                for attr in sorted(attrs - set(allowed)):
                    offenders.append(f"{name}.{method} writes self.{attr}")
        self.assertEqual(
            offenders, [],
            "adapter state written outside a setup method and not declared "
            "in MEMOIZED_WRITES — carry it on the AdapterSynthesisRequest / "
            "Result instead, or, if it is safe to serve to the next request, "
            "name the attribute in that adapter's MEMOIZED_WRITES with the "
            "reason:\n  " + "\n  ".join(offenders))

    def test_every_declared_exemption_is_still_reached(self):
        """A renamed method or attribute must not leave a stale exemption."""
        for name, klass in _adapter_classes().items():
            writes = _self_writes(klass)
            for method, attrs in klass.MEMOIZED_WRITES.items():
                with self.subTest(adapter=name, method=method):
                    self.assertTrue(
                        callable(getattr(klass, method, None)),
                        f"{name} exempts '{method}', which it does not define")
                    unused = set(attrs) - writes.get(method, set())
                    self.assertEqual(
                        unused, set(),
                        f"{name}.{method} no longer writes {sorted(unused)}")

    def test_a_plain_assignment_is_caught(self):
        class _Leaky(_Stub):
            def build_feed_dict(self, request, session):
                self.reference = request.params["reference_audio"]
                return {}

        self.assertEqual(_self_writes(_Leaky), {"build_feed_dict": {"reference"}})

    def test_a_tuple_assignment_is_caught(self):
        class _Leaky(_Stub):
            def build_feed_dict(self, request, session):
                self.reference, self.rate = request.params["reference_audio"]
                return {}

        self.assertEqual(_self_writes(_Leaky),
                         {"build_feed_dict": {"reference", "rate"}})

    def test_a_starred_assignment_is_caught(self):
        class _Leaky(_Stub):
            def build_feed_dict(self, request, session):
                first, *self.rest = request.phoneme_ids
                return {}

        self.assertEqual(_self_writes(_Leaky), {"build_feed_dict": {"rest"}})

    def test_setattr_is_caught(self):
        class _Leaky(_Stub):
            def build_feed_dict(self, request, session):
                setattr(self, "reference", request.params["reference_audio"])
                return {}

        self.assertEqual(_self_writes(_Leaky), {"build_feed_dict": {"reference"}})

    def test_a_subscript_write_is_caught(self):
        class _Leaky(_Stub):
            def build_feed_dict(self, request, session):
                self.cache[request.speaker_id] = request.params
                return {}

        self.assertEqual(_self_writes(_Leaky), {"build_feed_dict": {"cache"}})

    def test_an_in_place_container_mutation_is_caught(self):
        class _Leaky(_Stub):
            def build_feed_dict(self, request, session):
                self.seen.append(request.speaker_id)
                self.params.update(request.params)
                return {}

        self.assertEqual(_self_writes(_Leaky),
                         {"build_feed_dict": {"seen", "params"}})

    def test_an_augmented_assignment_is_caught(self):
        class _Leaky(_Stub):
            def build_feed_dict(self, request, session):
                self.calls += 1
                return {}

        self.assertEqual(_self_writes(_Leaky), {"build_feed_dict": {"calls"}})

    def test_reading_self_is_not_a_write(self):
        class _Innocent(_Stub):
            def build_feed_dict(self, request, session):
                local = self.params["x"]
                other = {}
                other["y"] = self.params
                return {"a": local, "b": other}

        self.assertEqual(_self_writes(_Innocent), {})


class _Stub(BaseOnnxAdapter):
    """Minimal concrete adapter, so the cases above only differ in the write."""

    def build_feed_dict(self, request, session):
        return {}

    def parse_outputs(self, outputs, request, output_names=None):
        return None

    def default_params(self):
        return {}


if __name__ == "__main__":
    unittest.main()
