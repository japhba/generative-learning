"""Offline checks for billing bounds, replay, and separation of test histories."""

import asyncio
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from openrouter_learning import OpenRouter, SpendingLedger, run_world
from permutation_task import make_world


class SpendingTests(unittest.TestCase):
    def test_pending_requests_count_against_cap_and_survive_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ledger = SpendingLedger(path, Decimal("0.01"))
            ledger.reserve("one", Decimal("0.009"), "mock")
            with self.assertRaises(RuntimeError): ledger.reserve("two", Decimal("0.002"), "mock")
            ledger.close()
            resumed = SpendingLedger(path, Decimal("0.01"))
            self.assertEqual(resumed.committed, Decimal("0.009"))
            with self.assertRaises(RuntimeError): resumed.reserve("one", Decimal("0.001"), "mock")
            resumed.settle("one", Decimal("0.001"))
            resumed.reserve("two", Decimal("0.009"), "mock")
            self.assertEqual(resumed.committed, Decimal("0.01"))
            resumed.close()


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_world_waits_for_sibling_evaluation_to_settle(self):
        sibling_started, sibling_finished = asyncio.Event(), asyncio.Event()

        async def respond(request):
            payload = json.loads(request.content)
            history = payload["messages"]
            is_test = "Independent test questions" in history[-1]["content"]
            has_feedback = any("Correct worked examples" in message["content"] for message in history)
            if is_test and has_feedback:
                if any(message["role"] == "assistant" for message in history):
                    sibling_started.set()
                    await asyncio.sleep(0.02)
                    sibling_finished.set()
                else:
                    await sibling_started.wait()
                    return httpx.Response(400, json={"error": "Mock study evaluation failed"})
            questions = payload["response_format"]["json_schema"]["schema"]["properties"]["answers"]["maxItems"]
            return httpx.Response(200, json={"provider": "offline_mock", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"answers": ["A"] * questions})}}], "usage": {"cost": 0.00001, "prompt_tokens": 300, "completion_tokens": 20}})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ledger = SpendingLedger(path / "ledger", Decimal("0.02"))
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                api = OpenRouter(client, ledger, 2)
                with self.assertRaises(ExceptionGroup) as caught:
                    await run_world(api, "qwen/qwen3-30b-a3b-instruct-2507", make_world(123), path)
                self.assertTrue(sibling_finished.is_set())
                self.assertEqual(len(caught.exception.exceptions), 1)
                self.assertIsInstance(caught.exception.exceptions[0], httpx.HTTPStatusError)
                self.assertEqual(len(ledger.reserved), 4)
                self.assertEqual(len(ledger.paid), 3)
                self.assertEqual(sum(ledger.paid.values()), Decimal("0.00003"))
                api.progress.close()
            ledger.close()

    async def test_recovery_cache_retains_original_unresolved_reservation(self):
        requests = []

        def respond(request):
            requests.append(request)
            raise httpx.ReadError("Original request outcome is unknown", request=request)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ledger = SpendingLedger(path, Decimal("0.02"))
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                api = OpenRouter(client, ledger, 1)
                arguments = ("qwen/qwen3-30b-a3b-instruct-2507", [{"role": "user", "content": "1?"}], [1], 1)
                with self.assertRaises(httpx.ReadError): await api.predict(*arguments)
                original_id, = ledger.reserved
                recovery_id = original_id + "_recovery_1"
                ledger.reserve(recovery_id, Decimal("0.001"), "offline_mock")
                body = {"provider": "offline_mock", "choices": [{"finish_reason": "stop", "message": {"content": '{"answers": ["A"]}'}}], "usage": {"cost": 0.00001, "prompt_tokens": 300, "completion_tokens": 20}}
                (path / f"{original_id}.json").write_text(json.dumps({"call_id": original_id, "http_status": 200, "response": body, "recovery_attempt_call_id": recovery_id}))
                recovered = await api.predict(*arguments)
                self.assertEqual(recovered["call_id"], original_id)
                self.assertNotIn(original_id, ledger.paid)
                self.assertEqual(ledger.paid, {recovery_id: Decimal("0.00001")})
                self.assertEqual(ledger.committed, ledger.reserved[original_id] + Decimal("0.00001"))
                self.assertEqual(await api.predict(*arguments), recovered)
                self.assertEqual(len(requests), 1)
                api.progress.close()
            ledger.close()

    async def test_concurrent_identical_requests_are_charged_once(self):
        requests = []

        async def respond(request):
            requests.append(request)
            await asyncio.sleep(0.01)
            return httpx.Response(200, json={"provider": "offline_mock", "choices": [{"finish_reason": "stop", "message": {"content": '{"answers": ["A"]}'}}], "usage": {"cost": 0.00001, "prompt_tokens": 300, "completion_tokens": 20}})

        with tempfile.TemporaryDirectory() as directory:
            ledger = SpendingLedger(Path(directory), Decimal("0.02"))
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                api = OpenRouter(client, ledger, 2)
                calls = [api.predict("qwen/qwen3-30b-a3b-instruct-2507", [{"role": "user", "content": "1?"}], [1], 1) for _ in range(2)]
                first, second = await asyncio.gather(*calls)
                self.assertEqual(first, second)
                self.assertEqual(len(requests), 1)
                self.assertEqual(ledger.committed, Decimal("0.00001"))
                api.progress.close()
            ledger.close()

    async def test_full_world_is_cached_and_evaluation_never_enters_practice(self):
        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            questions = payload["response_format"]["json_schema"]["schema"]["properties"]["answers"]["maxItems"]
            body = {"provider": "offline_mock", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"answers": ["A"] * questions})}}], "usage": {"cost": 0.00001, "prompt_tokens": 300, "completion_tokens": 20}}
            return httpx.Response(200, json=body)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "world_results").mkdir()
            ledger = SpendingLedger(path / "ledger", Decimal("0.02"))
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                api = OpenRouter(client, ledger, 2)
                world = make_world(123)
                rows = await run_world(api, "qwen/qwen3-30b-a3b-instruct-2507", world, path)
                self.assertEqual(len(requests), 13)
                self.assertEqual(len(rows), 10)
                self.assertEqual(rows[0]["call_id"], rows[1]["call_id"])
                self.assertEqual(ledger.committed, Decimal("0.00013"))
                for payload in requests:
                    history = payload["messages"]
                    self.assertFalse(any("Independent test questions" in message["content"] for message in history[:-1]))
                    self.assertEqual(payload["provider"]["max_price"], {"prompt": 0.11, "completion": 0.45, "request": 0})
                final_study, final_generate = [row for row in rows if row["budget"] == 8]
                self.assertEqual(final_study["practice_cost_usd"], 0)
                self.assertAlmostEqual(final_generate["practice_cost_usd"], 0.00004)
                replay = await run_world(api, "qwen/qwen3-30b-a3b-instruct-2507", world, path)
                self.assertEqual(rows, replay)
                self.assertEqual(len(requests), 13)
                self.assertEqual(ledger.committed, Decimal("0.00013"))
                api.progress.close()
            ledger.close()


if __name__ == "__main__": unittest.main()
