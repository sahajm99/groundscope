"""A tiny local MCP server (stdio) exposing utility tools.

Demonstrates Groundscope as an MCP *server* (v2.7 seed) and gives the agent's
MCP *client* something real to call. Run standalone: python mcp_servers/util_server.py
Or let the agent spawn it via mcp.json.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("groundscope-utils")

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.FloorDiv: operator.floordiv,
}


def _eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '47 * 89' or '(2+3)**4'."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


@mcp.tool()
def current_datetime() -> str:
    """Return the current UTC date and time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    mcp.run(transport="stdio")
