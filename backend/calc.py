"""Deterministic calculator that shows its working.

The problem statement asks for "calculations with steps shown". An LLM must
not be trusted to do arithmetic on engineering numbers, so expressions are
evaluated here with Python's AST -- no eval() -- and every intermediate
sub-expression is recorded so the output is auditable by an engineer.
"""
import ast
import math
import operator

_BINOPS = {
    ast.Add: ("+", operator.add),
    ast.Sub: ("-", operator.sub),
    ast.Mult: ("*", operator.mul),
    ast.Div: ("/", operator.truediv),
    ast.Pow: ("**", operator.pow),
    ast.Mod: ("%", operator.mod),
    ast.FloorDiv: ("//", operator.floordiv),
}

_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "radians": math.radians,
    "pi": math.pi, "e": math.e,
}


class CalcError(Exception):
    pass


def _checked(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalcError("only real numeric values are supported")
    if isinstance(value, int) and value.bit_length() > 1024:
        raise CalcError("integer size limit exceeded")
    if isinstance(value, float) and not math.isfinite(value):
        raise CalcError("result must be finite")
    return value


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.12g}"
    return str(v)


def _walk(node, steps):
    if isinstance(node, ast.Expression):
        return _walk(node.body, steps)
    if isinstance(node, ast.Constant):
        return _checked(node.value)
    if isinstance(node, ast.Name):
        if node.id in _FUNCS and not callable(_FUNCS[node.id]):
            return _FUNCS[node.id]
        raise CalcError(f"unknown name '{node.id}'")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _walk(node.operand, steps)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in _BINOPS:
            raise CalcError("unsupported operator")
        sym, fn = _BINOPS[op]
        left, right = _walk(node.left, steps), _walk(node.right, steps)
        if op is ast.Pow and (abs(right) > 1000 or (isinstance(left, int) and right > 0
                                                    and left.bit_length() * right > 1024)):
            raise CalcError("exponent exceeds the calculation limit")
        try:
            val = _checked(fn(left, right))
        except (ArithmeticError, ValueError) as exc:
            raise CalcError(str(exc)) from exc
        steps.append(f"{_fmt(left)} {sym} {_fmt(right)} = {_fmt(val)}")
        return val
    if isinstance(node, ast.Call):
        if (not isinstance(node.func, ast.Name) or not callable(_FUNCS.get(node.func.id))
                or node.keywords or len(node.args) > 16):
            raise CalcError("use a whitelisted function with positional numeric arguments")
        name = node.func.id
        args = [_walk(a, steps) for a in node.args]
        try:
            val = _checked(_FUNCS[name](*args))
        except (ArithmeticError, ValueError, TypeError) as exc:
            raise CalcError(str(exc)) from exc
        steps.append(f"{name}({', '.join(_fmt(a) for a in args)}) = {_fmt(val)}")
        return val
    raise CalcError(f"unsupported expression element: {type(node).__name__}")


def evaluate(expression, label=""):
    """Return (result, steps). Raises CalcError on anything unsafe."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 4096:
        raise CalcError("expression must contain 1 to 4096 characters")
    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 256:
            raise CalcError("expression is too complex")
        steps = []
        result = _checked(_walk(tree, steps))
        return result, steps
    except (SyntaxError, RecursionError) as exc:
        raise CalcError("invalid or overly nested expression") from exc


def evaluate_text(expression, label=""):
    """Human-readable calculation report with every step shown."""
    result, steps = evaluate(expression)
    head = f"Calculation{': ' + label if label else ''}\nExpression: {expression}"
    body = "\n".join(f"  step {i+1}: {s}" for i, s in enumerate(steps)) or "  (single value)"
    return f"{head}\nSteps:\n{body}\nResult: {_fmt(result)}"
