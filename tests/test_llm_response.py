from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.envstate import llm_response


class FakeHTTPError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class SequenceCompletions:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def client_with(outcomes):
    completions = SequenceCompletions(outcomes)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


class ModelRouteRetryTests(unittest.TestCase):
    marker = (
        'Model "gpt-5.6-luna" is not supported by any configured account '
        "in this group"
    )

    def test_only_matching_account_group_404_is_retryable(self):
        self.assertTrue(
            llm_response._is_retryable_model_route_error(
                FakeHTTPError(404, self.marker)
            )
        )
        self.assertFalse(
            llm_response._is_retryable_model_route_error(
                FakeHTTPError(404, "model does not exist")
            )
        )
        self.assertFalse(
            llm_response._is_retryable_model_route_error(
                FakeHTTPError(500, self.marker)
            )
        )

    @patch.object(llm_response, "_sleep_backoff")
    def test_route_404_retries_then_returns_success(self, _sleep):
        success = object()
        client, completions = client_with(
            [FakeHTTPError(404, self.marker), success]
        )
        result = llm_response._create_with_backoff(
            client,
            "gpt-5.6-luna",
            [],
            {},
            attempts=4,
            route_404_attempts=8,
            base=0,
            cap=0,
        )
        self.assertIs(result, success)
        self.assertEqual(completions.calls, 2)

    @patch.object(llm_response, "_sleep_backoff")
    def test_route_404_exhaustion_returns_none(self, _sleep):
        client, completions = client_with(
            [FakeHTTPError(404, self.marker) for _ in range(3)]
        )
        result = llm_response._create_with_backoff(
            client,
            "gpt-5.6-luna",
            [],
            {},
            attempts=1,
            route_404_attempts=3,
            base=0,
            cap=0,
        )
        self.assertIsNone(result)
        self.assertEqual(completions.calls, 3)

    @patch.object(llm_response, "_sleep_backoff")
    def test_ordinary_404_still_raises_immediately(self, _sleep):
        client, completions = client_with(
            [FakeHTTPError(404, "model does not exist")]
        )
        with self.assertRaises(FakeHTTPError):
            llm_response._create_with_backoff(
                client,
                "bad-model",
                [],
                {},
                attempts=4,
                route_404_attempts=8,
                base=0,
                cap=0,
            )
        self.assertEqual(completions.calls, 1)


if __name__ == "__main__":
    unittest.main()
