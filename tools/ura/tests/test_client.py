import json
import unittest
import urllib.error

from uradata.client import BROWSER_UA, URAClient, URAError, decode_payload


class FakeOpener:
    """Records requests and replays scripted responses (bytes or exceptions)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers):
        self.calls.append((url, headers))
        response = self.responses.pop(0) if self.responses else b'{"Status":"Success","Result":[]}'
        if isinstance(response, Exception):
            raise response
        return response


def payload(**kwargs):
    return json.dumps({"Status": "Success", "Message": "", **kwargs}).encode()


TOKEN = payload(Result="tok-123")


class TestDecode(unittest.TestCase):
    def test_plain_utf8(self):
        self.assertEqual(decode_payload(b'{"a": 1}'), {"a": 1})

    def test_latin1_bytes_do_not_raise(self):
        # URA emits accented project names as Latin-1 inside otherwise valid JSON.
        raw = '{"project": "CAFÉ"}'.encode("latin-1")
        self.assertEqual(decode_payload(raw)["project"], "CAFÉ")

    def test_utf8_is_preferred_over_latin1(self):
        raw = '{"project": "CAFÉ"}'.encode("utf-8")
        self.assertEqual(decode_payload(raw)["project"], "CAFÉ")


class TestClient(unittest.TestCase):
    def test_requires_a_key(self):
        with self.assertRaises(URAError):
            URAClient(access_key="", opener=FakeOpener())

    def test_reads_key_from_environment(self):
        import os
        os.environ["URA_ACCESS_KEY"] = "env-key"
        try:
            self.assertEqual(URAClient(opener=FakeOpener()).access_key, "env-key")
        finally:
            del os.environ["URA_ACCESS_KEY"]

    def test_sends_browser_user_agent(self):
        # Without this header the live endpoint returns 403.
        opener = FakeOpener(TOKEN)
        URAClient("k", opener=opener).token
        self.assertEqual(opener.calls[0][1]["User-Agent"], BROWSER_UA)

    def test_token_is_memoised(self):
        opener = FakeOpener(TOKEN, payload(Result=[]), payload(Result=[]))
        client = URAClient("k", opener=opener)
        client.service("A")
        client.service("B")
        token_calls = [c for c in opener.calls if "insertNewToken" in c[0]]
        self.assertEqual(len(token_calls), 1)

    def test_service_sends_token_header(self):
        opener = FakeOpener(TOKEN, payload(Result=[]))
        URAClient("k", opener=opener).service("PMI_Resi_Transaction", batch=1)
        url, headers = opener.calls[1]
        self.assertIn("service=PMI_Resi_Transaction", url)
        self.assertIn("&batch=1", url)
        self.assertEqual(headers["Token"], "tok-123")
        self.assertEqual(headers["AccessKey"], "k")

    def test_retries_transient_failures_with_backoff(self):
        slept = []
        opener = FakeOpener(urllib.error.URLError("reset"), urllib.error.URLError("reset"), TOKEN)
        client = URAClient("k", opener=opener, sleep=slept.append)
        self.assertEqual(client.token, "tok-123")
        self.assertEqual(slept, [2, 4])

    def test_gives_up_after_retries(self):
        opener = FakeOpener(*[urllib.error.URLError("down")] * 4)
        client = URAClient("k", opener=opener, sleep=lambda _: None)
        with self.assertRaises(URAError):
            client.token

    def test_rejects_error_status(self):
        opener = FakeOpener(TOKEN, json.dumps(
            {"Status": "Error", "Message": "Invalid service."}).encode())
        client = URAClient("k", opener=opener, sleep=lambda _: None)
        with self.assertRaises(URAError) as caught:
            client.service("PMI_Resi_Rental_Contract")
        self.assertIn("Invalid service", str(caught.exception))

    def test_rejects_token_without_result(self):
        opener = FakeOpener(json.dumps({"Status": "Success", "Result": ""}).encode())
        with self.assertRaises(URAError):
            URAClient("k", opener=opener, sleep=lambda _: None).token

    def test_transactions_fetches_every_batch(self):
        opener = FakeOpener(TOKEN, *[payload(Result=[{"project": f"P{i}"}]) for i in range(4)])
        seen = []
        result = URAClient("k", opener=opener).transactions(progress=lambda b, n: seen.append(b))
        self.assertEqual(len(result), 4)
        self.assertEqual(seen, [1, 2, 3, 4])
