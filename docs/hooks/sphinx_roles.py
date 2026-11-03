"""MkDocs hook: render the Sphinx roles of the docstrings.

The docstrings follow the numpydoc standard and refer to other objects with
Sphinx roles (``:func:`pils.store.save_version```). mkdocstrings renders
the docstrings as Markdown, which leaves ``:func:`` in front of the name.
This hook turns each role into an optional cross-reference
(mkdocs-autorefs): a link when the object is documented, plain code
otherwise, and drops the reST literal-block marker (a paragraph ending in
``::``), which Markdown shows as text.
"""

from __future__ import annotations

import html
import re
from typing import Any

ROLES = "func|meth|mod|class|attr|data|obj|exc|const"
PATTERN = re.compile(rf":(?:py:)?(?:{ROLES}):<code>(?P<target>[^<]+)</code>")
LITERAL_MARKER = re.compile(r"\s*::</p>")


def _reference(match: re.Match[str]) -> str:
    target = html.unescape(match["target"])
    # "~pils.store.save_version" shows the last component only (Sphinx)
    shown = target.rsplit(".", 1)[-1] if target.startswith("~") else target
    identifier = target.lstrip("~.").removesuffix("()")
    return (
        f'<autoref identifier="{html.escape(identifier)}" optional>'
        f"<code>{html.escape(shown.lstrip('~'))}</code></autoref>"
    )


def on_page_content(html_text: str, **kwargs: Any) -> str:
    """Replace the Sphinx roles left in the rendered page."""
    html_text = LITERAL_MARKER.sub(":</p>", html_text)
    return PATTERN.sub(_reference, html_text)
