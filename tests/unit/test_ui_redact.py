"""`helix ui` masks secrets in every text it shows."""

from __future__ import annotations

import pytest

from helix.ui import redact as redact_module
from helix.ui.redact import MASK, add_env_names, redact


@pytest.mark.parametrize(
    "secret",
    [
        "sk-abcdefghijklmnop1234",
        "sk-ant-api03-AbC_d-e",
        "ghp_" + "a1" * 12,
        "xoxb-1234-abcd-EFGH",
        "AKIA" + "ABCDEFGHIJKLMNOP",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl",
    ],
)
def test_token_shapes_are_masked(secret: str) -> None:
    out = redact(f"before {secret} after")
    assert secret not in out
    assert out == f"before {MASK} after"


def test_short_sk_prefix_is_not_a_secret() -> None:
    assert redact("task-list sk-short") == "task-list sk-short"


def test_bearer_keeps_scheme() -> None:
    assert redact("Authorization: Bearer abc.def/ghi+jk==") == f"Authorization: Bearer {MASK}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("OPENAI_API_KEY=hunter2", f"OPENAI_API_KEY={MASK}"),
        ("export GH_TOKEN=abc123 && run", f"export GH_TOKEN={MASK} && run"),
        ('DB_PASSWORD="two words"', f"DB_PASSWORD={MASK}"),
        ("MY_SECRET='x y'", f"MY_SECRET={MASK}"),
        ("PASS=1 AUTH_HEADER=z", f"PASS={MASK} AUTH_HEADER={MASK}"),
        ("MUJOCO_GL=egl", "MUJOCO_GL=egl"),
    ],
)
def test_secret_looking_assignments_are_masked(text: str, expected: str) -> None:
    assert redact(text) == expected


def test_backend_auth_env_values_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "plain-looking-credential")
    assert redact("echo plain-looking-credential") == f"echo {MASK}"


def test_registered_env_names_are_masked_but_short_values_are_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(redact_module, "_env_names", set(redact_module._env_names))
    monkeypatch.setenv("MY_ENDPOINT", "https://internal.example/v1")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    add_env_names(["MY_ENDPOINT", "MUJOCO_GL"])
    assert redact("url https://internal.example/v1 gl egl") == f"url {MASK} gl egl"


def test_plain_text_is_unchanged() -> None:
    text = "score: 0.75 · feedback: grasp slipped at t=3.2"
    assert redact(text) == text
