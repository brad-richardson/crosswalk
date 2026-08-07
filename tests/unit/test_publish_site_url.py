"""The publish path must never ship its placeholder site URL.

`--site-url` is written into `index.json` (which the live dashboard reads and
links from) and into the credibility page's query examples. On 2026-08-07 a real
publish ran without it and shipped `https://bridges.example.com` to production —
a plausible-looking dead link, which is worse than an obviously broken one
because nothing about it invites a second look. These pin both halves of the fix.
"""

from __future__ import annotations

from crosswalk.factory.publish import DEFAULT_SITE_URL


def test_placeholder_site_url_is_unmistakably_invalid():
    """The default must not look like a real host.

    RFC 2606 reserves `.invalid` as guaranteed-never-resolvable, so a placeholder
    that escapes is obviously broken. An `example.com`-style default renders as a
    normal link and reads as intentional.
    """
    assert DEFAULT_SITE_URL.endswith(".invalid"), (
        f"DEFAULT_SITE_URL is {DEFAULT_SITE_URL!r}. Use a .invalid host so an "
        "un-overridden placeholder cannot be mistaken for a real URL."
    )
    assert "example.com" not in DEFAULT_SITE_URL


def test_no_dry_run_requires_explicit_site_url(tmp_path):
    """`publish --no-dry-run` without `--site-url` must exit non-zero.

    This is the guard that makes the placeholder unreachable from a published
    artifact. Asserted against the CLI rather than the helper, because the check
    lives in the command and that is what a release runs.

    ``--factory-root`` points at an EMPTY directory on purpose. The guard has to
    fire on argument validation alone, before any filesystem check -- an earlier
    revision sat after the factory-root existence check, so on a machine without
    factory output (CI) the command reported the missing output instead and this
    test failed for the right reason. Passing a nonexistent root here would let
    that regression pass again on a developer machine that happens to have
    data/factory populated.
    """
    from typer.testing import CliRunner

    from crosswalk.cli import app

    empty_root = tmp_path / "no-factory-output-here"

    result = CliRunner().invoke(
        app,
        ["factory", "publish", "--all", "--no-dry-run", "--factory-root", str(empty_root)],
    )
    assert result.exit_code != 0, (
        "publish --no-dry-run succeeded without --site-url; the placeholder can "
        "reach production again"
    )
    assert "--site-url is required" in result.output, (
        "Expected the --site-url guard to fire first, but got:\n" + result.output
    )


def test_targets_publish_is_exempt_from_the_site_url_guard(tmp_path):
    """``--targets`` must NOT be blocked: that flow never reads ``site_url``.

    Guards that over-fire get deleted by the next person in a hurry. Pin the
    exemption so the boundary is deliberate rather than incidental.
    """
    from typer.testing import CliRunner

    from crosswalk.cli import app

    result = CliRunner().invoke(
        app,
        ["factory", "publish", "--all", "--no-dry-run", "--targets", "--raw-dir", str(tmp_path)],
    )
    assert "--site-url is required" not in result.output, (
        "The --site-url guard fired for a --targets publish, which does not use "
        "site_url at all:\n" + result.output
    )
