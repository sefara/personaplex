"""Exercise the merged server's disconnect and optional Mimi branches without weights."""

import ast
import asyncio
import os
import pathlib
import subprocess
import sys
import types
import unittest


SOURCE = pathlib.Path(__file__).parents[1] / "moshi" / "moshi" / "server.py"
SERVER = ast.parse(SOURCE.read_text(encoding="utf-8"))
STATE = next(node for node in SERVER.body if isinstance(node, ast.ClassDef) and node.name == "ServerState")
CHAT = next(node for node in STATE.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "handle_chat")


def extract_opus_loop(namespace):
    loop = next(node for node in ast.walk(CHAT) if isinstance(node, ast.AsyncFunctionDef) and node.name == "opus_loop")
    module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["opus_loop"]


def extract_optional_reset():
    for node in ast.walk(CHAT):
        if not isinstance(node, ast.If):
            continue
        if ast.unparse(node.test) != "self.other_mimi is not None":
            continue
        if any("reset_streaming" in ast.unparse(child) for child in node.body):
            function = ast.FunctionDef(
                name="reset_optional", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")],
                kwonlyargs=[], kw_defaults=[], defaults=[]), body=[node], decorator_list=[]
            )
            module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
            namespace = {}
            exec(compile(module, str(SOURCE), "exec"), namespace)
            return namespace["reset_optional"]
    raise AssertionError("No optional Mimi reset found in handle_chat")


class FakeReader:
    def __init__(self, values):
        self.values = iter(values)

    def read_pcm(self):
        return next(self.values)


class StreamRecoveryTests(unittest.TestCase):
    def test_ended_opus_stream_releases_session_for_next_client(self):
        async def scenario():
            lock = asyncio.Lock()
            model = types.SimpleNamespace(
                mimi=types.SimpleNamespace(parameters=lambda: iter([types.SimpleNamespace(dtype="bf16")]))
            )
            for _ in range(2):
                namespace = {"asyncio": asyncio, "close": False, "self": model,
                             "opus_reader": FakeReader([None])}
                opus_loop = extract_opus_loop(namespace)
                async with asyncio.timeout(1):
                    async with lock:
                        self.assertIsNone(await opus_loop())
            self.assertFalse(lock.locked())

        asyncio.run(scenario())

    def test_other_mimi_reset_is_optional(self):
        reset_optional = extract_optional_reset()
        reset_optional(types.SimpleNamespace(other_mimi=None))
        calls = []
        other = types.SimpleNamespace(reset_streaming=lambda: calls.append("reset"))
        reset_optional(types.SimpleNamespace(other_mimi=other))
        self.assertEqual(calls, ["reset"])

    def test_fast_rejects_disabled_torch_compile(self):
        entrypoint = SOURCE.parents[2] / "scripts" / "dgx-spark-entrypoint.py"
        environment = dict(os.environ, NO_TORCH_COMPILE="0")
        result = subprocess.run(
            [sys.executable, str(entrypoint), "python", "-m", "moshi.server", "--fast"],
            env=environment, capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NO_TORCH_COMPILE to be unset", result.stderr)


if __name__ == "__main__":
    unittest.main()
