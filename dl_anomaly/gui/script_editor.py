"""
gui/script_editor.py - Industrial Vision-style script editor panel with safe AST-based interpreter.

Provides:
- ScriptInterpreter: A safe AST-based interpreter that walks Python AST nodes without
  ever calling eval() or exec(). Supports a DSL of image-processing functions mirroring
  vision operators, with variable assignment, control flow, f-strings, and user-defined
  functions.
- ScriptEditor(ttk.Frame): A toggleable panel with a code editor (left pane), output
  console (right pane), toolbar, line numbers, and real-time syntax highlighting.
- ScriptError: Custom exception carrying a line number for error highlighting.
"""

from __future__ import annotations

import ast
import logging
import math
import operator
import re
import time
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Theme constants                                                        #
# ====================================================================== #

_BG_DARK = "#2b2b2b"
_BG_MEDIUM = "#3c3c3c"
_FG_LIGHT = "#cccccc"
_ACCENT = "#0078d4"
_CANVAS_BG = "#1e1e1e"


# ====================================================================== #
#  ScriptError                                                            #
# ====================================================================== #

class ScriptError(Exception):
    """Script execution error with optional source line number."""

    def __init__(self, message: str, line: int = 0) -> None:
        super().__init__(message)
        self.message = message
        self.line = line


# ====================================================================== #
#  Flow-control sentinel exceptions (internal)                            #
# ====================================================================== #

class _BreakException(Exception):
    """Raised by ast.Break to unwind the loop."""


class _ContinueException(Exception):
    """Raised by ast.Continue to skip to the next iteration."""


class _ReturnException(Exception):
    """Raised by ast.Return to propagate a return value."""

    def __init__(self, value: Any = None) -> None:
        super().__init__()
        self.value = value


# ====================================================================== #
#  MathProxy - safe math module proxy                                     #
# ====================================================================== #

class _MathProxy:
    """Exposes a safe subset of the ``math`` module as attribute access.

    Scripts can write ``math.sqrt(2)``, ``math.pi``, etc.
    """

    sqrt = math.sqrt
    pi = math.pi
    e = math.e
    sin = math.sin
    cos = math.cos
    tan = math.tan
    asin = math.asin
    acos = math.acos
    atan = math.atan
    atan2 = math.atan2
    log = math.log
    log2 = math.log2
    log10 = math.log10
    exp = math.exp
    pow = math.pow
    ceil = math.ceil
    floor = math.floor
    fabs = math.fabs
    hypot = math.hypot
    degrees = math.degrees
    radians = math.radians
    inf = math.inf
    nan = math.nan


# ====================================================================== #
#  ScriptInterpreter - Safe AST-based interpreter                         #
# ====================================================================== #

class ScriptInterpreter:
    """Walk a Python AST with a restricted evaluator.  NEVER uses eval()/exec().

    The interpreter receives a reference to the main application (``app``)
    and an optional dict of extra callable functions.  It builds an internal
    registry of whitelisted DSL functions (image I/O, region operations,
    vision operators, drawing, measurement, barcode, built-in helpers) and
    stores script variables in a flat dict.  Output from ``print()`` calls
    is captured into an internal list.

    Parameters
    ----------
    app : object or None
        The main InspectorApp instance.  Used to obtain the current image
        (``app.pipeline_panel``) and to push results back into the viewer
        (``app.image_viewer``, ``app.pipeline_panel``).
    extra_functions : dict or None
        Additional ``{name: callable}`` entries merged into the registry.
    """

    # Safety limits.
    MAX_WHILE_ITERATIONS: int = 10_000
    MAX_CALL_DEPTH: int = 64

    # ------------------------------------------------------------------ #
    #  Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        app: Any = None,
        extra_functions: Optional[Dict[str, Callable]] = None,
    ) -> None:
        self._app = app
        self._output: List[str] = []

        # Variable scope (flat -- no nested scopes for simplicity).
        self._variables: Dict[str, Any] = {}

        # User-defined functions (def name(args): body).
        self._user_functions: Dict[str, ast.FunctionDef] = {}

        # Call depth counter.
        self._call_depth: int = 0

        # Current AST node (for error reporting).
        self._current_node: Optional[ast.AST] = None

        # Build the allowed built-in function registry.
        self._builtins: Dict[str, Any] = self._build_builtins()

        # Merge any extra functions provided by the caller.
        if extra_functions:
            self._builtins.update(extra_functions)

    # ------------------------------------------------------------------ #
    #  Built-in function registry                                         #
    # ------------------------------------------------------------------ #

    def _build_builtins(self) -> Dict[str, Any]:
        """Return a dict mapping DSL function names to their implementations."""
        b: Dict[str, Any] = {}

        # -- Display / IO --
        b["print"] = self._fn_print
        b["display"] = self._fn_display
        b["display_region"] = self._fn_display_region

        # -- Image acquisition --
        b["read_image"] = self._fn_read_image
        b["current_image"] = self._fn_current_image

        # -- region_ops wrappers --
        b["threshold"] = self._fn_threshold
        b["binary_threshold"] = self._fn_binary_threshold
        b["connection"] = self._fn_connection
        b["select_shape"] = self._fn_select_shape
        b["select_gray"] = self._fn_select_gray
        b["fill_up"] = self._fn_fill_up
        b["shape_trans"] = self._fn_shape_trans
        b["erosion_region"] = self._fn_erosion_region
        b["dilation_region"] = self._fn_dilation_region
        b["opening_region"] = self._fn_opening_region
        b["closing_region"] = self._fn_closing_region
        b["union_region"] = self._fn_union_region
        b["intersection_region"] = self._fn_intersection_region
        b["difference_region"] = self._fn_difference_region
        b["complement_region"] = self._fn_complement_region
        b["count_obj"] = self._fn_count_obj
        b["region_to_mask"] = self._fn_region_to_mask
        b["sort_region"] = self._fn_sort_region
        b["get_single_region"] = self._fn_get_single_region
        b["area_center"] = self._fn_area_center
        b["smallest_rectangle1"] = self._fn_smallest_rectangle1
        b["smallest_circle"] = self._fn_smallest_circle
        b["inner_circle"] = self._fn_inner_circle
        b["region_to_display_image"] = self._fn_region_to_display_image

        # -- vision_ops wrappers (image processing) --
        vision_funcs = [
            "rgb_to_gray", "gauss_filter", "mean_image", "median_image",
            "sharpen_image", "edges_canny", "sobel_filter", "laplace_filter",
            "invert_image", "add_image", "sub_image", "rotate_image",
            "mirror_image", "histogram_eq", "entropy_image", "deviation_image",
            "template_match_ncc", "match_shape",
            "measure_line_profile", "distance_pp",
            "paint_region", "draw_text", "draw_line", "draw_rectangle",
            "draw_circle", "draw_cross", "draw_arrow",
            "find_barcode", "find_qrcode",
        ]
        for name in vision_funcs:
            b[name] = self._make_vision_wrapper(name)

        # -- Safe Python built-ins --
        b["len"] = len
        b["range"] = range
        b["int"] = int
        b["float"] = float
        b["str"] = str
        b["abs"] = abs
        b["min"] = min
        b["max"] = max
        b["sum"] = sum
        b["round"] = round
        b["enumerate"] = enumerate
        b["zip"] = zip
        b["sorted"] = sorted
        b["reversed"] = reversed
        b["type"] = type
        b["isinstance"] = isinstance
        b["list"] = list
        b["tuple"] = tuple
        b["dict"] = dict
        b["set"] = set
        b["bool"] = bool
        b["map"] = map
        b["filter"] = filter

        # -- math module proxy --
        b["math"] = _MathProxy()

        return b

    # ------------------------------------------------------------------ #
    #  vision_ops generic wrapper factory                                   #
    # ------------------------------------------------------------------ #

    def _make_vision_wrapper(self, func_name: str) -> Callable:
        """Create a lazy wrapper that imports from dl_anomaly.core.vision_ops at call time."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                import dl_anomaly.core.vision_ops as vision_ops
            except ImportError:
                raise ScriptError(
                    f"core.vision_ops 模組未安裝，無法呼叫 {func_name}()",
                    line=self._get_current_line(),
                )
            fn = getattr(vision_ops, func_name, None)
            if fn is None:
                raise ScriptError(
                    f"core.vision_ops 中找不到函式 {func_name}()",
                    line=self._get_current_line(),
                )
            return fn(*args, **kwargs)

        return wrapper

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def output(self) -> List[str]:
        """Return captured output lines."""
        return list(self._output)

    def execute(self, source: str) -> List[str]:
        """Parse *source* into an AST and execute it.

        Returns the list of captured output lines.

        Raises ``ScriptError`` on any problem (syntax errors, runtime errors,
        forbidden operations, etc.).
        """
        # Reset state.
        self._variables.clear()
        self._user_functions.clear()
        self._call_depth = 0
        self._output.clear()

        try:
            tree = ast.parse(source, filename="<script>", mode="exec")
        except SyntaxError as exc:
            line = exc.lineno or 0
            raise ScriptError(f"語法錯誤: {exc.msg}", line=line)

        self._exec_body(tree.body)
        return list(self._output)

    # ------------------------------------------------------------------ #
    #  Statement execution                                                #
    # ------------------------------------------------------------------ #

    def _exec_body(self, stmts: List[ast.stmt]) -> None:
        """Execute a list of statements sequentially."""
        for stmt in stmts:
            self._exec_stmt(stmt)

    def _exec_stmt(self, node: ast.stmt) -> None:  # noqa: C901
        """Dispatch a single statement node."""
        self._current_node = node

        # -- ast.Expr (expression statement, e.g. function call) --
        if isinstance(node, ast.Expr):
            self._eval_expr(node.value)

        # -- ast.Assign --
        elif isinstance(node, ast.Assign):
            value = self._eval_expr(node.value)
            for target in node.targets:
                self._assign_target(target, value)

        # -- ast.AugAssign (+=, -=, etc.) --
        elif isinstance(node, ast.AugAssign):
            current = self._eval_expr(node.target)
            rhs = self._eval_expr(node.value)
            op = self._augassign_op(node.op)
            result = op(current, rhs)
            self._assign_target(node.target, result)

        # -- ast.If --
        elif isinstance(node, ast.If):
            test = self._eval_expr(node.test)
            if test:
                self._exec_body(node.body)
            elif node.orelse:
                self._exec_body(node.orelse)

        # -- ast.For --
        elif isinstance(node, ast.For):
            iterable = self._eval_expr(node.iter)
            for item in iterable:
                self._assign_target(node.target, item)
                try:
                    self._exec_body(node.body)
                except _BreakException:
                    break
                except _ContinueException:
                    continue
            else:
                if node.orelse:
                    self._exec_body(node.orelse)

        # -- ast.While --
        elif isinstance(node, ast.While):
            iterations = 0
            while self._eval_expr(node.test):
                iterations += 1
                if iterations > self.MAX_WHILE_ITERATIONS:
                    raise ScriptError(
                        f"while 迴圈超過安全上限 ({self.MAX_WHILE_ITERATIONS} 次迭代)",
                        line=getattr(node, "lineno", 0),
                    )
                try:
                    self._exec_body(node.body)
                except _BreakException:
                    break
                except _ContinueException:
                    continue
            else:
                if node.orelse:
                    self._exec_body(node.orelse)

        # -- ast.Break / ast.Continue --
        elif isinstance(node, ast.Break):
            raise _BreakException()

        elif isinstance(node, ast.Continue):
            raise _ContinueException()

        # -- ast.FunctionDef (user-defined function) --
        elif isinstance(node, ast.FunctionDef):
            self._user_functions[node.name] = node

        # -- ast.Return --
        elif isinstance(node, ast.Return):
            value = self._eval_expr(node.value) if node.value else None
            raise _ReturnException(value)

        # -- ast.Pass --
        elif isinstance(node, ast.Pass):
            pass

        # -- ast.Import / ast.ImportFrom (blocked) --
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ScriptError(
                "import 語句被禁止 (安全限制)",
                line=getattr(node, "lineno", 0),
            )

        else:
            raise ScriptError(
                f"不支援的語句類型: {type(node).__name__}",
                line=getattr(node, "lineno", 0),
            )

    # ------------------------------------------------------------------ #
    #  Expression evaluation                                              #
    # ------------------------------------------------------------------ #

    def _eval_expr(self, node: ast.expr) -> Any:  # noqa: C901
        """Evaluate an expression node and return its value."""
        if node is None:
            return None

        self._current_node = node

        # -- ast.Constant (numbers, strings, booleans, None) --
        if isinstance(node, ast.Constant):
            return node.value

        # -- ast.Name (variable lookup) --
        elif isinstance(node, ast.Name):
            name = node.id
            if name in self._variables:
                return self._variables[name]
            if name in self._builtins:
                return self._builtins[name]
            if name in self._user_functions:
                return self._user_functions[name]
            # Python built-in constants.
            if name == "True":
                return True
            if name == "False":
                return False
            if name == "None":
                return None
            raise ScriptError(
                f"未定義的變數: '{name}'",
                line=getattr(node, "lineno", 0),
            )

        # -- ast.BinOp --
        elif isinstance(node, ast.BinOp):
            left = self._eval_expr(node.left)
            right = self._eval_expr(node.right)
            return self._binop(node.op, left, right)

        # -- ast.UnaryOp --
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_expr(node.operand)
            return self._unaryop(node.op, operand)

        # -- ast.BoolOp (and / or) --
        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result: Any = True
                for v in node.values:
                    result = self._eval_expr(v)
                    if not result:
                        return result
                return result
            else:  # ast.Or
                result = False
                for v in node.values:
                    result = self._eval_expr(v)
                    if result:
                        return result
                return result

        # -- ast.Compare --
        elif isinstance(node, ast.Compare):
            left = self._eval_expr(node.left)
            for op_node, comparator in zip(node.ops, node.comparators):
                right = self._eval_expr(comparator)
                if not self._compare(op_node, left, right):
                    return False
                left = right
            return True

        # -- ast.Call --
        elif isinstance(node, ast.Call):
            func = self._eval_expr(node.func)

            # Build positional args, handling *starred.
            args: List[Any] = []
            has_starred = False
            for a in node.args:
                if isinstance(a, ast.Starred):
                    has_starred = True
                    val = self._eval_expr(a.value)
                    args.extend(val)
                else:
                    args.append(self._eval_expr(a))

            # Build keyword args.
            kwargs: Dict[str, Any] = {}
            for kw in node.keywords:
                if kw.arg is not None:
                    kwargs[kw.arg] = self._eval_expr(kw.value)

            return self._call_function(func, args, kwargs, node)

        # -- ast.Attribute (obj.attr) --
        elif isinstance(node, ast.Attribute):
            obj = self._eval_expr(node.value)
            attr_name = node.attr
            try:
                return getattr(obj, attr_name)
            except AttributeError:
                raise ScriptError(
                    f"物件沒有屬性 '{attr_name}'",
                    line=getattr(node, "lineno", 0),
                )

        # -- ast.Subscript (obj[key]) --
        elif isinstance(node, ast.Subscript):
            obj = self._eval_expr(node.value)
            sl = node.slice

            # Python 3.9+ uses the value directly; older uses ast.Index.
            if isinstance(sl, ast.Index):
                idx = self._eval_expr(sl.value)  # type: ignore[attr-defined]
            elif isinstance(sl, ast.Slice):
                lower = self._eval_expr(sl.lower) if sl.lower else None
                upper = self._eval_expr(sl.upper) if sl.upper else None
                step = self._eval_expr(sl.step) if sl.step else None
                idx = slice(lower, upper, step)
            else:
                # Python 3.9+: sl is the node itself.
                if isinstance(sl, ast.Slice):
                    lower = self._eval_expr(sl.lower) if sl.lower else None
                    upper = self._eval_expr(sl.upper) if sl.upper else None
                    step = self._eval_expr(sl.step) if sl.step else None
                    idx = slice(lower, upper, step)
                else:
                    idx = self._eval_expr(sl)

            try:
                return obj[idx]
            except (IndexError, KeyError, TypeError) as exc:
                raise ScriptError(
                    f"索引錯誤: {exc}",
                    line=getattr(node, "lineno", 0),
                )

        # -- ast.List --
        elif isinstance(node, ast.List):
            return [self._eval_expr(e) for e in node.elts]

        # -- ast.Tuple --
        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_expr(e) for e in node.elts)

        # -- ast.Dict --
        elif isinstance(node, ast.Dict):
            return {
                self._eval_expr(k): self._eval_expr(v)
                for k, v in zip(node.keys, node.values)
            }

        # -- ast.Set --
        elif isinstance(node, ast.Set):
            return {self._eval_expr(e) for e in node.elts}

        # -- ast.IfExp (ternary: a if cond else b) --
        elif isinstance(node, ast.IfExp):
            return (
                self._eval_expr(node.body)
                if self._eval_expr(node.test)
                else self._eval_expr(node.orelse)
            )

        # -- ast.JoinedStr (f-string) --
        elif isinstance(node, ast.JoinedStr):
            parts: List[str] = []
            for v in node.values:
                if isinstance(v, ast.Constant):
                    parts.append(str(v.value))
                elif isinstance(v, ast.FormattedValue):
                    val = self._eval_expr(v.value)
                    if v.format_spec:
                        fmt = self._eval_expr(v.format_spec)
                        parts.append(format(val, fmt))
                    else:
                        conversion = v.conversion
                        if conversion == ord("s"):
                            parts.append(str(val))
                        elif conversion == ord("r"):
                            parts.append(repr(val))
                        elif conversion == ord("a"):
                            parts.append(ascii(val))
                        else:
                            parts.append(str(val))
                else:
                    parts.append(str(self._eval_expr(v)))
            return "".join(parts)

        # -- ast.FormattedValue (used inside f-strings) --
        elif isinstance(node, ast.FormattedValue):
            val = self._eval_expr(node.value)
            if node.format_spec:
                fmt = self._eval_expr(node.format_spec)
                return format(val, fmt)
            return val

        # -- ast.ListComp (basic list comprehension) --
        elif isinstance(node, ast.ListComp):
            return self._eval_listcomp(node)

        # -- ast.Starred (handled in Call, fallback here) --
        elif isinstance(node, ast.Starred):
            return self._eval_expr(node.value)

        # -- ast.NamedExpr (walrus operator :=) --
        elif hasattr(ast, "NamedExpr") and isinstance(node, ast.NamedExpr):
            val = self._eval_expr(node.value)
            self._variables[node.target.id] = val
            return val

        else:
            raise ScriptError(
                f"不支援的表達式類型: {type(node).__name__}",
                line=getattr(node, "lineno", 0),
            )

    # ------------------------------------------------------------------ #
    #  List comprehension                                                 #
    # ------------------------------------------------------------------ #

    def _eval_listcomp(self, node: ast.ListComp) -> List[Any]:
        """Evaluate a list comprehension ``[expr for x in iter if cond]``."""
        result: List[Any] = []
        self._eval_generators(node.generators, 0, node.elt, result)
        return result

    def _eval_generators(
        self,
        generators: List[ast.comprehension],
        idx: int,
        elt: ast.expr,
        result: List[Any],
    ) -> None:
        if idx >= len(generators):
            result.append(self._eval_expr(elt))
            return

        gen = generators[idx]
        iterable = self._eval_expr(gen.iter)
        for item in iterable:
            self._assign_target(gen.target, item)
            if all(self._eval_expr(cond) for cond in gen.ifs):
                self._eval_generators(generators, idx + 1, elt, result)

    # ------------------------------------------------------------------ #
    #  Assignment helpers                                                 #
    # ------------------------------------------------------------------ #

    def _assign_target(self, target: ast.expr, value: Any) -> None:
        """Assign *value* to the given target expression."""
        if isinstance(target, ast.Name):
            self._variables[target.id] = value

        elif isinstance(target, (ast.Tuple, ast.List)):
            # Unpacking: a, b, c = (1, 2, 3)
            try:
                items = list(value)
            except TypeError:
                raise ScriptError(
                    "無法解包賦值目標",
                    line=getattr(target, "lineno", 0),
                )
            if len(items) != len(target.elts):
                raise ScriptError(
                    f"解包數量不匹配: 預期 {len(target.elts)} 個值，實際 {len(items)} 個",
                    line=getattr(target, "lineno", 0),
                )
            for t, v in zip(target.elts, items):
                self._assign_target(t, v)

        elif isinstance(target, ast.Subscript):
            obj = self._eval_expr(target.value)
            sl = target.slice
            if isinstance(sl, ast.Index):
                idx = self._eval_expr(sl.value)  # type: ignore[attr-defined]
            else:
                idx = self._eval_expr(sl)
            obj[idx] = value

        elif isinstance(target, ast.Attribute):
            obj = self._eval_expr(target.value)
            setattr(obj, target.attr, value)

        else:
            raise ScriptError(
                f"不支援的賦值目標: {type(target).__name__}",
                line=getattr(target, "lineno", 0),
            )

    # ------------------------------------------------------------------ #
    #  Operator helpers                                                   #
    # ------------------------------------------------------------------ #

    _BINOP_MAP: Dict[type, Callable] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.BitAnd: operator.and_,
    }

    _UNARYOP_MAP: Dict[type, Callable] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Not: operator.not_,
        ast.Invert: operator.invert,
    }

    _CMPOP_MAP: Dict[type, Callable] = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    _AUGASSIGN_MAP: Dict[type, Callable] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.BitAnd: operator.and_,
    }

    def _binop(self, op: ast.operator, left: Any, right: Any) -> Any:
        fn = self._BINOP_MAP.get(type(op))
        if fn is None:
            raise ScriptError(
                f"不支援的運算子: {type(op).__name__}",
                line=self._get_current_line(),
            )
        try:
            return fn(left, right)
        except Exception as exc:
            raise ScriptError(str(exc), line=self._get_current_line())

    def _unaryop(self, op: ast.unaryop, operand: Any) -> Any:
        fn = self._UNARYOP_MAP.get(type(op))
        if fn is None:
            raise ScriptError(
                f"不支援的一元運算子: {type(op).__name__}",
                line=self._get_current_line(),
            )
        return fn(operand)

    def _compare(self, op: ast.cmpop, left: Any, right: Any) -> bool:
        if isinstance(op, ast.In):
            return left in right
        if isinstance(op, ast.NotIn):
            return left not in right
        if isinstance(op, ast.Is):
            return left is right
        if isinstance(op, ast.IsNot):
            return left is not right
        fn = self._CMPOP_MAP.get(type(op))
        if fn is None:
            raise ScriptError(
                f"不支援的比較運算子: {type(op).__name__}",
                line=self._get_current_line(),
            )
        try:
            return fn(left, right)
        except TypeError:
            return False

    def _augassign_op(self, op: ast.operator) -> Callable:
        fn = self._AUGASSIGN_MAP.get(type(op))
        if fn is None:
            raise ScriptError(
                f"不支援的複合賦值運算子: {type(op).__name__}",
                line=self._get_current_line(),
            )
        return fn

    # ------------------------------------------------------------------ #
    #  Function call dispatch                                             #
    # ------------------------------------------------------------------ #

    def _call_function(
        self,
        func: Any,
        args: List[Any],
        kwargs: Dict[str, Any],
        call_node: ast.Call,
    ) -> Any:
        """Call *func* with the given arguments."""
        # User-defined function (ast.FunctionDef).
        if isinstance(func, ast.FunctionDef):
            return self._call_user_function(func, args, kwargs)

        # Callable (built-in or Python built-in).
        if callable(func):
            try:
                return func(*args, **kwargs)
            except ScriptError:
                raise
            except _ReturnException:
                raise
            except Exception as exc:
                raise ScriptError(
                    f"函式執行錯誤: {exc}",
                    line=getattr(call_node, "lineno", 0),
                )

        raise ScriptError(
            f"'{func}' 不是可呼叫的函式",
            line=getattr(call_node, "lineno", 0),
        )

    def _call_user_function(
        self,
        func_def: ast.FunctionDef,
        args: List[Any],
        kwargs: Dict[str, Any],
    ) -> Any:
        """Execute a user-defined function with a new local scope."""
        self._call_depth += 1
        if self._call_depth > self.MAX_CALL_DEPTH:
            raise ScriptError(
                f"呼叫深度超過上限 ({self.MAX_CALL_DEPTH})",
                line=getattr(func_def, "lineno", 0),
            )

        # Save current variables.
        saved_vars = self._variables.copy()

        try:
            # Bind parameters.
            params = func_def.args
            param_names = [arg.arg for arg in params.args]

            for i, name in enumerate(param_names):
                if i < len(args):
                    self._variables[name] = args[i]
                elif name in kwargs:
                    self._variables[name] = kwargs[name]
                elif params.defaults:
                    default_offset = len(param_names) - len(params.defaults)
                    if i >= default_offset:
                        self._variables[name] = self._eval_expr(
                            params.defaults[i - default_offset]
                        )
                    else:
                        raise ScriptError(
                            f"函式 '{func_def.name}' 缺少參數 '{name}'",
                            line=getattr(func_def, "lineno", 0),
                        )
                else:
                    raise ScriptError(
                        f"函式 '{func_def.name}' 缺少參數 '{name}'",
                        line=getattr(func_def, "lineno", 0),
                    )

            try:
                self._exec_body(func_def.body)
            except _ReturnException as ret:
                return ret.value

            return None

        finally:
            self._variables = saved_vars
            self._call_depth -= 1

    # ------------------------------------------------------------------ #
    #  Helper                                                             #
    # ------------------------------------------------------------------ #

    def _get_current_line(self) -> int:
        if self._current_node and hasattr(self._current_node, "lineno"):
            return self._current_node.lineno
        return 0

    def _get_current_image_from_app(self) -> Optional[np.ndarray]:
        """Retrieve the currently displayed image from the app's pipeline."""
        if self._app is None:
            return None
        try:
            panel = self._app.pipeline_panel
            step = panel.get_current_step()
            if step is not None:
                return step.image
        except AttributeError:
            pass
        return None

    def _add_pipeline_step(
        self,
        name: str,
        image: np.ndarray,
        region: Any = None,
    ) -> None:
        """Push a result into the app's pipeline panel and image viewer."""
        if self._app is None:
            return
        try:
            panel = self._app.pipeline_panel
            if region is not None:
                panel.add_step(name, image, "script", region=region)
            else:
                panel.add_step(name, image, "script")
        except AttributeError:
            pass

    # ================================================================== #
    #  Built-in function implementations                                  #
    # ================================================================== #

    # -- print --
    def _fn_print(self, *args: Any, **kwargs: Any) -> None:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "")
        text = sep.join(str(a) for a in args) + end
        self._output.append(text)

    # -- Image I/O --
    def _fn_read_image(self, path: str) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise ScriptError(
                f"無法讀取影像: {path}",
                line=self._get_current_line(),
            )
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _fn_current_image(self) -> np.ndarray:
        img = self._get_current_image_from_app()
        if img is not None:
            return img
        raise ScriptError("沒有目前影像", line=self._get_current_line())

    # -- Display --
    def _fn_display(self, img: np.ndarray) -> None:
        self._add_pipeline_step("腳本結果", img)

    def _fn_display_region(self, region: Any, color: str = "green") -> None:
        try:
            from dl_anomaly.core.region_ops import region_to_display_image
        except ImportError:
            raise ScriptError(
                "core.region_ops 模組未安裝，無法呼叫 display_region()",
                line=self._get_current_line(),
            )
        src = region.source_image
        display_img = region_to_display_image(region, src)
        self._add_pipeline_step(
            f"區域顯示 ({region.num_regions})", display_img, region=region
        )

    # -- region_ops --
    def _fn_threshold(self, image: np.ndarray, min_val: int, max_val: int) -> Any:
        from dl_anomaly.core.region_ops import threshold
        return threshold(image, int(min_val), int(max_val))

    def _fn_binary_threshold(self, image: np.ndarray, method: str = "otsu") -> Any:
        from dl_anomaly.core.region_ops import binary_threshold
        return binary_threshold(image, method)

    def _fn_connection(self, region: Any) -> Any:
        from dl_anomaly.core.region_ops import connection
        return connection(region)

    def _fn_select_shape(
        self, region: Any, feature: str, min_val: float, max_val: float
    ) -> Any:
        from dl_anomaly.core.region_ops import select_shape
        return select_shape(region, feature, float(min_val), float(max_val))

    def _fn_select_gray(
        self, region: Any, image: np.ndarray, feature: str,
        min_val: float, max_val: float
    ) -> Any:
        """Select regions by gray-value features.

        Filters regions whose gray-value *feature* (``mean_value``,
        ``min_value``, ``max_value``) inside the given *image* falls within
        [min_val, max_val].  This re-computes gray statistics against *image*
        before filtering.
        """
        from dl_anomaly.core.region_ops import compute_region_properties
        from dl_anomaly.core.region import Region

        gray_props = compute_region_properties(region.labels, image)
        feature_map = {
            "mean": "mean_value",
            "min": "min_value",
            "max": "max_value",
            "mean_value": "mean_value",
            "min_value": "min_value",
            "max_value": "max_value",
        }
        feat = feature_map.get(feature, feature)

        passing: List[int] = []
        for p in gray_props:
            val = getattr(p, feat, None)
            if val is not None and float(min_val) <= val <= float(max_val):
                passing.append(p.index)

        return region._keep_indices(passing)

    def _fn_fill_up(self, region: Any) -> Any:
        """Fill holes in region masks using flood-fill from the border."""
        from dl_anomaly.core.region_ops import compute_region_properties
        from dl_anomaly.core.region import Region

        mask = region.to_binary_mask()
        h, w = mask.shape[:2]
        filled = mask.copy()
        # Flood-fill from a border pixel (top-left corner of a padded image).
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        inv = cv2.bitwise_not(filled)
        cv2.floodFill(inv, flood_mask, (0, 0), 0)
        filled = filled | inv

        num, labels = cv2.connectedComponents(filled, connectivity=8)
        labels = labels.astype(np.int32)
        props = compute_region_properties(labels, region.source_image)
        return Region(
            labels=labels,
            num_regions=num - 1,
            properties=props,
            source_image=region.source_image,
            source_shape=region.source_shape,
        )

    def _fn_shape_trans(self, region: Any, trans_type: str) -> Any:
        """Shape transformation (convex, rectangle1, rectangle2, circle, ellipse)."""
        from dl_anomaly.core.region_ops import compute_region_properties
        from dl_anomaly.core.region import Region

        mask = region.to_binary_mask()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_mask = np.zeros_like(mask)

        for cnt in contours:
            if trans_type == "convex":
                hull = cv2.convexHull(cnt)
                cv2.fillPoly(result_mask, [hull], 255)
            elif trans_type == "rectangle1":
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(result_mask, (bx, by), (bx + bw, by + bh), 255, -1)
            elif trans_type == "circle":
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                cv2.circle(result_mask, (int(cx), int(cy)), int(r), 255, -1)
            elif trans_type == "ellipse" and len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(result_mask, ellipse, 255, -1)
            else:
                cv2.drawContours(result_mask, [cnt], -1, 255, -1)

        num, labels = cv2.connectedComponents(result_mask, connectivity=8)
        labels = labels.astype(np.int32)
        props = compute_region_properties(labels, region.source_image)
        return Region(
            labels=labels,
            num_regions=num - 1,
            properties=props,
            source_image=region.source_image,
            source_shape=region.source_shape,
        )

    def _fn_erosion_region(self, region: Any, ksize: int = 3) -> Any:
        return self._morphology_op(region, cv2.MORPH_ERODE, int(ksize))

    def _fn_dilation_region(self, region: Any, ksize: int = 3) -> Any:
        return self._morphology_op(region, cv2.MORPH_DILATE, int(ksize))

    def _fn_opening_region(self, region: Any, ksize: int = 3) -> Any:
        return self._morphology_op(region, cv2.MORPH_OPEN, int(ksize))

    def _fn_closing_region(self, region: Any, ksize: int = 3) -> Any:
        return self._morphology_op(region, cv2.MORPH_CLOSE, int(ksize))

    def _morphology_op(self, region: Any, morph_type: int, ksize: int) -> Any:
        """Apply a morphological operation to a region mask."""
        from dl_anomaly.core.region_ops import compute_region_properties
        from dl_anomaly.core.region import Region

        mask = region.to_binary_mask()
        k = max(ksize, 1)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

        if morph_type == cv2.MORPH_ERODE:
            result_mask = cv2.erode(mask, kernel)
        elif morph_type == cv2.MORPH_DILATE:
            result_mask = cv2.dilate(mask, kernel)
        else:
            result_mask = cv2.morphologyEx(mask, morph_type, kernel)

        num, labels = cv2.connectedComponents(result_mask, connectivity=8)
        labels = labels.astype(np.int32)
        props = compute_region_properties(labels, region.source_image)
        return Region(
            labels=labels,
            num_regions=num - 1,
            properties=props,
            source_image=region.source_image,
            source_shape=region.source_shape,
        )

    def _fn_union_region(self, a: Any, b: Any) -> Any:
        return a.union(b)

    def _fn_intersection_region(self, a: Any, b: Any) -> Any:
        return a.intersection(b)

    def _fn_difference_region(self, a: Any, b: Any) -> Any:
        return a.difference(b)

    def _fn_complement_region(self, region: Any) -> Any:
        return region.complement()

    def _fn_count_obj(self, region: Any) -> int:
        return region.num_regions

    def _fn_region_to_mask(self, region: Any) -> np.ndarray:
        return region.to_binary_mask()

    def _fn_sort_region(
        self, region: Any, feature: str, order: str = "ascending"
    ) -> Any:
        """Sort regions by a feature value."""
        from dl_anomaly.core.region import Region
        import copy

        if not region.properties:
            return region

        reverse = (order.lower() in ("descending", "desc"))
        sorted_props = sorted(
            region.properties,
            key=lambda p: getattr(p, feature, 0),
            reverse=reverse,
        )
        old_to_new = {}
        new_props = []
        for new_idx, p in enumerate(sorted_props, 1):
            old_to_new[p.index] = new_idx
            p_copy = copy.copy(p)
            p_copy.index = new_idx
            new_props.append(p_copy)

        new_labels = np.zeros_like(region.labels)
        for old_idx, new_idx in old_to_new.items():
            new_labels[region.labels == old_idx] = new_idx

        return Region(
            labels=new_labels,
            num_regions=region.num_regions,
            properties=new_props,
            source_image=region.source_image,
            source_shape=region.source_shape,
        )

    def _fn_get_single_region(self, region: Any, index: int) -> Any:
        return region.get_single_region(int(index))

    def _fn_area_center(self, region: Any) -> Tuple[int, float, float]:
        if region.num_regions == 0 or not region.properties:
            return (0, 0.0, 0.0)
        total_area = sum(p.area for p in region.properties)
        if total_area == 0:
            return (0, 0.0, 0.0)
        cx = sum(p.centroid[0] * p.area for p in region.properties) / total_area
        cy = sum(p.centroid[1] * p.area for p in region.properties) / total_area
        return (total_area, cx, cy)

    def _fn_smallest_rectangle1(self, region: Any) -> Tuple[int, int, int, int]:
        """Return bounding box (x, y, w, h) of the combined region."""
        mask = region.to_binary_mask()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (0, 0, 0, 0)
        all_pts = np.vstack(contours)
        bx, by, bw, bh = cv2.boundingRect(all_pts)
        return (bx, by, bw, bh)

    def _fn_smallest_circle(self, region: Any) -> Tuple[float, float, float]:
        """Return minimum enclosing circle (cx, cy, radius)."""
        mask = region.to_binary_mask()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (0.0, 0.0, 0.0)
        all_pts = np.vstack(contours)
        (cx, cy), r = cv2.minEnclosingCircle(all_pts)
        return (float(cx), float(cy), float(r))

    def _fn_inner_circle(self, region: Any) -> Tuple[float, float, float]:
        """Approximate the largest inscribed circle using distance transform."""
        mask = region.to_binary_mask()
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist)
        return (float(max_loc[0]), float(max_loc[1]), float(max_val))

    def _fn_region_to_display_image(
        self, region: Any, source_image: Any = None
    ) -> np.ndarray:
        from dl_anomaly.core.region_ops import region_to_display_image
        if source_image is None:
            source_image = region.source_image
        return region_to_display_image(region, source_image)


# ====================================================================== #
#  ScriptEditor - The UI Panel                                            #
# ====================================================================== #

_DEFAULT_SCRIPT = '''\
# Blob Analysis Script
img = current_image()
gray = rgb_to_gray(img)
blurred = gauss_filter(gray, 1.5)
region = threshold(blurred, 0, 128)
regions = connection(region)
n = count_obj(regions)
print(f"Found {n} regions")

for i in range(n):
    r = get_single_region(regions, i + 1)
    a, cx, cy = area_center(r)
    if a > 100:
        print(f"Region {i+1}: area={a}, center=({cx:.1f}, {cy:.1f})")

display_region(regions, "green")
'''

# Syntax-highlighting word lists.
_KEYWORDS = [
    "if", "elif", "else", "for", "while", "in", "not", "and", "or",
    "True", "False", "None", "return", "def", "break", "continue",
]

_BUILTINS = [
    # region_ops
    "read_image", "current_image", "threshold", "binary_threshold",
    "connection", "select_shape", "select_gray", "fill_up", "shape_trans",
    "erosion_region", "dilation_region", "opening_region", "closing_region",
    "union_region", "intersection_region", "difference_region",
    "complement_region", "count_obj", "region_to_mask", "sort_region",
    "get_single_region", "area_center", "smallest_rectangle1",
    "smallest_circle", "inner_circle", "region_to_display_image",
    # vision_ops
    "rgb_to_gray", "gauss_filter", "mean_image", "median_image",
    "sharpen_image", "edges_canny", "sobel_filter", "laplace_filter",
    "invert_image", "add_image", "sub_image", "rotate_image",
    "mirror_image", "histogram_eq", "entropy_image", "deviation_image",
    "template_match_ncc", "match_shape",
    "measure_line_profile", "distance_pp",
    "paint_region", "draw_text", "draw_line", "draw_rectangle",
    "draw_circle", "draw_cross", "draw_arrow",
    "find_barcode", "find_qrcode",
    # display
    "display", "display_region", "print",
    # safe python
    "len", "int", "float", "str", "abs", "min", "max", "sum", "round",
    "range", "enumerate", "zip", "sorted", "reversed", "type",
    "isinstance", "list", "tuple", "dict", "set", "bool",
]


class ScriptEditor(ttk.Frame):
    """Industrial Vision-style script editor panel.

    Embeddable in the main application.  Provides a horizontal PanedWindow
    with a code editor (left) and output console (right), a toolbar across
    the top, and real-time syntax highlighting with line numbers.

    Parameters
    ----------
    master : tk.Widget
        Parent widget.
    app : object or None
        The main InspectorApp instance, used to obtain the current image and
        to push script results into the pipeline panel / image viewer.
    """

    def __init__(
        self,
        master: tk.Widget,
        app: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, **kwargs)

        self._app = app

        # Debounce ID for syntax highlighting.
        self._highlight_after_id: Optional[str] = None

        self._build_ui()
        self._bind_events()

        # Load the default script.
        self._code_text.insert("1.0", _DEFAULT_SCRIPT)
        self._apply_syntax_highlighting()
        self._update_line_numbers()

    # ================================================================== #
    #  UI Construction                                                    #
    # ================================================================== #

    def _build_ui(self) -> None:
        """Create the complete editor UI: toolbar, paned code/output areas."""

        # ---- Toolbar ----
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=2, pady=(2, 0))

        btn_style: Dict[str, Any] = {
            "bg": _BG_MEDIUM, "fg": "#e0e0e0",
            "activebackground": _ACCENT, "activeforeground": "#ffffff",
            "bd": 0, "padx": 8, "pady": 3,
            "font": ("Consolas", 9),
        }

        self._btn_run = tk.Button(
            toolbar, text="\u25b6 \u57f7\u884c (F9)",
            command=self.execute, **btn_style,
        )
        self._btn_run.pack(side=tk.LEFT, padx=2)

        self._btn_clear = tk.Button(
            toolbar, text="\u6e05\u9664\u8f38\u51fa",
            command=self._clear_output, **btn_style,
        )
        self._btn_clear.pack(side=tk.LEFT, padx=2)

        self._btn_open = tk.Button(
            toolbar, text="\u958b\u555f",
            command=self._open_script, **btn_style,
        )
        self._btn_open.pack(side=tk.LEFT, padx=2)

        self._btn_save = tk.Button(
            toolbar, text="\u5132\u5b58",
            command=self._save_script, **btn_style,
        )
        self._btn_save.pack(side=tk.LEFT, padx=2)

        # ---- Paned window (horizontal split) ----
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # -- Left: Code editor with line numbers --
        editor_frame = ttk.Frame(paned)
        paned.add(editor_frame, weight=7)

        # Line numbers canvas (drawn via Canvas for pixel-perfect alignment).
        self._line_canvas = tk.Canvas(
            editor_frame,
            width=40,
            bg=_CANVAS_BG,
            bd=0,
            highlightthickness=0,
        )
        self._line_canvas.pack(side=tk.LEFT, fill=tk.Y)

        # Code text widget.
        self._code_text = tk.Text(
            editor_frame,
            bg=_CANVAS_BG,
            fg="#e0e0e0",
            insertbackground="#ffffff",
            selectbackground="#264f78",
            selectforeground="#ffffff",
            font=("Consolas", 11),
            undo=True,
            tabs="    ",
            bd=0,
            padx=4,
            pady=4,
            wrap=tk.NONE,
            highlightthickness=0,
        )
        self._code_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Vertical scrollbar synced with both line canvas and code editor.
        code_scroll = ttk.Scrollbar(editor_frame, orient=tk.VERTICAL)
        code_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        def on_scrollbar(*args: Any) -> None:
            self._code_text.yview(*args)
            self._update_line_numbers()

        def on_text_scroll(*args: Any) -> None:
            code_scroll.set(*args)
            self._update_line_numbers()

        code_scroll.configure(command=on_scrollbar)
        self._code_text.configure(yscrollcommand=on_text_scroll)

        # Configure syntax-highlighting tags.
        self._code_text.tag_configure("keyword", foreground="#569cd6")
        self._code_text.tag_configure("builtin", foreground="#dcdcaa")
        self._code_text.tag_configure("string", foreground="#ce9178")
        self._code_text.tag_configure("number", foreground="#b5cea8")
        self._code_text.tag_configure("comment", foreground="#6a9955")
        self._code_text.tag_configure("error_line", background="#5a1d1d")

        # -- Right: Output console --
        output_frame = ttk.LabelFrame(paned, text="\u8f38\u51fa")
        paned.add(output_frame, weight=3)

        self._output_text = tk.Text(
            output_frame,
            state=tk.DISABLED,
            bg=_CANVAS_BG,
            fg=_FG_LIGHT,
            font=("Consolas", 11),
            bd=0,
            padx=4,
            pady=4,
            wrap=tk.WORD,
            highlightthickness=0,
        )
        self._output_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        output_scroll = ttk.Scrollbar(
            output_frame, orient=tk.VERTICAL, command=self._output_text.yview,
        )
        output_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._output_text.configure(yscrollcommand=output_scroll.set)

        # Output tags.
        self._output_text.tag_configure("success", foreground="#4ec9b0")
        self._output_text.tag_configure("error", foreground="#f44747")
        self._output_text.tag_configure("normal", foreground="#4ec9b0")
        self._output_text.tag_configure("warning", foreground="#cca700")

    # ================================================================== #
    #  Event Bindings                                                     #
    # ================================================================== #

    def _bind_events(self) -> None:
        """Bind keyboard and scroll events."""
        self._code_text.bind("<KeyRelease>", self._on_key_release)
        self._code_text.bind("<F9>", lambda e: self.execute())
        self._code_text.bind("<MouseWheel>", self._on_mousewheel)
        self._code_text.bind("<Configure>", lambda e: self._update_line_numbers())
        # Bind F9 at the frame level as well.
        self.bind_all("<F9>", lambda e: self.execute())

    def _on_key_release(self, event: tk.Event) -> None:
        """Schedule syntax highlighting with debounce."""
        self._update_line_numbers()
        if self._highlight_after_id is not None:
            self.after_cancel(self._highlight_after_id)
        self._highlight_after_id = self.after(200, self._apply_syntax_highlighting)

    def _on_mousewheel(self, event: tk.Event) -> None:
        """Sync line numbers when mouse wheel scrolls the code editor."""
        # Schedule a line-number update after the scroll event is processed.
        self.after(10, self._update_line_numbers)

    # ================================================================== #
    #  Line Numbers (Canvas-based)                                        #
    # ================================================================== #

    def _update_line_numbers(self) -> None:
        """Redraw the line-number gutter on the Canvas, aligned with visible
        lines in the code editor."""
        self._line_canvas.delete("all")

        # Measure font metrics from the code text widget.
        font = self._code_text.cget("font")

        # Walk visible lines.
        idx = self._code_text.index("@0,0")
        while True:
            dline_info = self._code_text.dlineinfo(idx)
            if dline_info is None:
                break
            y = dline_info[1]
            line_num = int(str(idx).split(".")[0])
            self._line_canvas.create_text(
                36, y + 2,
                anchor="ne",
                text=str(line_num),
                fill="#888888",
                font=("Consolas", 11),
            )
            idx = self._code_text.index(f"{idx}+1line")
            if self._code_text.compare(idx, ">=", tk.END):
                break

    # ================================================================== #
    #  Syntax Highlighting                                                #
    # ================================================================== #

    def _apply_syntax_highlighting(self) -> None:
        """Remove all tags then re-apply using regex patterns."""
        self._highlight_after_id = None
        code = self._code_text.get("1.0", tk.END)

        # Remove existing tags.
        for tag in ("keyword", "builtin", "string", "number", "comment", "error_line"):
            self._code_text.tag_remove(tag, "1.0", tk.END)

        # Build patterns.  Order matters: comments and strings first so they
        # take precedence over keywords inside them.
        patterns: List[Tuple[str, str]] = []

        # Comments: # to end of line.
        patterns.append(("comment", r"#[^\n]*"))

        # Strings: triple-quoted first, then single/double.
        patterns.append(("string", r'"""[\s\S]*?"""'))
        patterns.append(("string", r"'''[\s\S]*?'''"))
        patterns.append(("string", r'f?"[^"\n]*"'))
        patterns.append(("string", r"f?'[^'\n]*'"))

        # Keywords (whole-word match).
        kw_pattern = r"\b(?:" + "|".join(_KEYWORDS) + r")\b"
        patterns.append(("keyword", kw_pattern))

        # Built-in function names (whole-word match).
        bi_pattern = r"\b(?:" + "|".join(_BUILTINS) + r")\b"
        patterns.append(("builtin", bi_pattern))

        # Numbers: integers and floats.
        patterns.append(("number", r"\b\d+\.?\d*(?:[eE][+-]?\d+)?\b"))

        # Track regions already tagged by higher-priority patterns to avoid
        # conflicts (e.g. keywords inside strings or comments).
        occupied: List[Tuple[int, int]] = []

        for tag, pattern in patterns:
            for match in re.finditer(pattern, code):
                start_offset = match.start()
                end_offset = match.end()

                # Skip if this region overlaps with a higher-priority tag.
                if tag not in ("comment", "string"):
                    overlap = False
                    for occ_start, occ_end in occupied:
                        if start_offset < occ_end and end_offset > occ_start:
                            overlap = True
                            break
                    if overlap:
                        continue

                # Convert character offset to tk line.col index.
                start_idx = self._offset_to_index(code, start_offset)
                end_idx = self._offset_to_index(code, end_offset)
                self._code_text.tag_add(tag, start_idx, end_idx)

                if tag in ("comment", "string"):
                    occupied.append((start_offset, end_offset))

    @staticmethod
    def _offset_to_index(text: str, offset: int) -> str:
        """Convert a character offset in *text* to a Tk ``line.col`` index."""
        line = text.count("\n", 0, offset) + 1
        col = offset - text.rfind("\n", 0, offset) - 1
        return f"{line}.{col}"

    # ================================================================== #
    #  Execution                                                          #
    # ================================================================== #

    def execute(self, _event: Any = None) -> None:
        """Parse and execute the script in the code editor."""
        code = self._code_text.get("1.0", "end-1c")
        if not code.strip():
            return

        # Remove previous error highlighting.
        self._code_text.tag_remove("error_line", "1.0", tk.END)

        self._clear_output()
        self._print_output("--- 執行開始 ---", tag="success")

        start_time = time.perf_counter()

        interpreter = ScriptInterpreter(app=self._app)

        try:
            output_lines = interpreter.execute(code)

            # Display captured output.
            for line in output_lines:
                self._print_output(line, tag="normal")

            elapsed = time.perf_counter() - start_time
            self._print_output(
                f"\n--- 執行完成 ({elapsed:.3f} 秒) ---", tag="success",
            )
            self._set_app_status(f"腳本執行完成 ({elapsed:.3f}s)")

        except ScriptError as exc:
            # Show any output captured before the error.
            for line in interpreter.output:
                self._print_output(line, tag="normal")

            self._print_output(
                f"\n錯誤 (第 {exc.line} 行): {exc.message}", tag="error",
            )
            if exc.line > 0:
                self._highlight_error_line(exc.line)
            self._set_app_status(f"腳本錯誤: {exc.message}")

        except Exception as exc:
            # Show any output captured before the error.
            for line in interpreter.output:
                self._print_output(line, tag="normal")

            tb = traceback.format_exc()
            self._print_output(f"\n未預期錯誤:\n{tb}", tag="error")
            self._set_app_status(f"腳本未預期錯誤: {exc}")

    # ================================================================== #
    #  Output Console                                                     #
    # ================================================================== #

    def _print_output(self, text: str, tag: str = "") -> None:
        """Append *text* to the output console."""
        self._output_text.configure(state=tk.NORMAL)
        if tag:
            self._output_text.insert(tk.END, text + "\n", tag)
        else:
            self._output_text.insert(tk.END, text + "\n")
        self._output_text.configure(state=tk.DISABLED)
        self._output_text.see(tk.END)

    def _clear_output(self) -> None:
        """Clear the output console."""
        self._output_text.configure(state=tk.NORMAL)
        self._output_text.delete("1.0", tk.END)
        self._output_text.configure(state=tk.DISABLED)

    # ================================================================== #
    #  Error Highlighting                                                 #
    # ================================================================== #

    def _highlight_error_line(self, line_num: int) -> None:
        """Add the ``error_line`` tag to the given line in the editor."""
        start = f"{line_num}.0"
        end = f"{line_num}.end"
        self._code_text.tag_add("error_line", start, end)
        self._code_text.see(start)

    # ================================================================== #
    #  File Operations                                                    #
    # ================================================================== #

    def _open_script(self) -> None:
        """Open a script file and load it into the editor."""
        path = filedialog.askopenfilename(
            title="開啟腳本",
            filetypes=[
                ("Python 腳本", "*.py"),
                ("文字檔案", "*.txt"),
                ("所有檔案", "*"),
            ],
        )
        if not path:
            return

        try:
            content = Path(path).read_text(encoding="utf-8")
            self._code_text.delete("1.0", tk.END)
            self._code_text.insert("1.0", content)
            self._apply_syntax_highlighting()
            self._update_line_numbers()
            self._set_app_status(f"腳本已載入: {Path(path).name}")
        except Exception as exc:
            self._print_output(f"開啟腳本失敗: {exc}", tag="error")

    def _save_script(self) -> None:
        """Save the editor content to a file."""
        path = filedialog.asksaveasfilename(
            title="儲存腳本",
            defaultextension=".py",
            filetypes=[
                ("Python 腳本", "*.py"),
                ("文字檔案", "*.txt"),
                ("所有檔案", "*"),
            ],
        )
        if not path:
            return

        try:
            content = self._code_text.get("1.0", "end-1c")
            Path(path).write_text(content, encoding="utf-8")
            self._set_app_status(f"腳本已儲存: {Path(path).name}")
        except Exception as exc:
            self._print_output(f"儲存腳本失敗: {exc}", tag="error")

    # ================================================================== #
    #  App status helper                                                  #
    # ================================================================== #

    def _set_app_status(self, message: str) -> None:
        """Update the main application's status bar, if available."""
        if self._app is not None:
            try:
                self._app.set_status(message)
            except AttributeError:
                pass

    # ================================================================== #
    #  Public helpers                                                     #
    # ================================================================== #

    def get_code(self) -> str:
        """Return the current script content."""
        return self._code_text.get("1.0", "end-1c")

    def set_code(self, code: str) -> None:
        """Replace the editor content with *code*."""
        self._code_text.delete("1.0", tk.END)
        self._code_text.insert("1.0", code)
        self._apply_syntax_highlighting()
        self._update_line_numbers()

    def focus_editor(self) -> None:
        """Set keyboard focus to the code editor."""
        self._code_text.focus_set()
