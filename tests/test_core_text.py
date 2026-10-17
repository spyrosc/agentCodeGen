"""Tests for :mod:`genai.core.text`."""

from genai.core.text import truncate_middle


def test_short_text_is_unchanged():
    assert truncate_middle("hello", 10) == "hello"
    assert truncate_middle("hello", 5) == "hello"


def test_long_text_keeps_head_and_tail_and_reports_dropped_count():
    text = "H" * 50 + "M" * 900 + "T" * 50
    out = truncate_middle(text, 100)
    head, marker, tail = out.split("\n")
    assert head == "H" * 50
    assert tail == "T" * 50
    assert marker == "... [truncated 900 chars] ..."


def test_odd_limit_reports_exact_dropped_count():
    out = truncate_middle("x" * 20, 5)
    head, marker, tail = out.split("\n")
    assert len(head) == len(tail) == 2
    assert "[truncated 16 chars]" in marker


def test_limit_one_does_not_leak_the_whole_text():
    # text[-0:] would return everything; the helper must not.
    out = truncate_middle("abcdefghij", 1)
    assert "abcdefghij" not in out
    assert "[truncated 10 chars]" in out


def test_non_positive_limit_disables_truncation():
    text = "y" * 1000
    assert truncate_middle(text, 0) == text
    assert truncate_middle(text, -1) == text
