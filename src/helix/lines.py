"""Line splitting for newline-delimited machine formats.

Why this exists
---------------
``str.splitlines()`` is the wrong tool for every format whose grammar says
"records are separated by a line feed".  Python splits on eight boundaries a
line feed is not::

    \\v  U+000B  LINE TABULATION       \\f  U+000C  FORM FEED
    \\x1c U+001C FILE SEPARATOR        \\x1d U+001D GROUP SEPARATOR
    \\x1e U+001E RECORD SEPARATOR      \\x85 U+0085 NEXT LINE (NEL)
    \\u2028      LINE SEPARATOR        \\u2029      PARAGRAPH SEPARATOR

JSON permits U+0085, U+2028 and U+2029 unescaped inside a string literal —
they are not C0 control characters, so RFC 8259 does not require escaping —
and the serializers the agent backends use do not escape them (Rust's
``serde_json`` for Codex, ``JSON.stringify`` for the Node CLIs).  The
remaining five are escaped as ``\\uXXXX`` by any conforming serializer, but
arrive verbatim in transcripts and evaluator streams that are not JSON.

The consequence for a JSONL stream is that one agent message, or one blob of
captured command output, containing a single NEL byte turns a valid record
into two invalid fragments.  Downstream that reads either as a hard parse
failure or as a silently wrong record count.

Use :func:`split_lf_lines` wherever the line boundary is defined by the
format (JSONL, a ``KEY=`` protocol line on stdout, a transcript file).  Keep
``str.splitlines()`` for human-authored text where breaking on any Unicode
line boundary is the intent.
"""

from __future__ import annotations


__all__ = ["split_lf_lines"]


def split_lf_lines(text: str) -> list[str]:
    """Split *text* on U+000A only, the way a newline-delimited format defines it.

    Unlike ``str.splitlines()`` this never breaks a record on an embedded
    Unicode line boundary such as U+0085 / U+2028 / U+2029, and it never
    silently rewrites one of those characters into a newline when the parts
    are rejoined.

    Callers that also need to tolerate CRLF should ``strip()`` each line, as
    the JSONL parsers here do; ``json.loads`` ignores trailing whitespace in
    any case.

    Args:
        text: The raw stream or file contents to split.

    Returns:
        The line feed separated pieces of *text*.  A trailing line feed
        yields a final empty string, which callers skip along with any other
        blank line.
    """
    return text.split("\n")
