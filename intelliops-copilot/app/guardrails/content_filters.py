import re

DESTRUCTIVE_PATTERNS = [
    re.compile(r"\brm\s+-(?:[a-z]*r[a-z]*f|[a-z]*f[a-z]*r)\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+(?:table|database|schema)\b", re.IGNORECASE),
    re.compile(r"\bdelete\s+from\b(?!.*\bwhere\b)", re.IGNORECASE),
    re.compile(r"\btruncate\s+table\b", re.IGNORECASE),
    re.compile(r"\bkubectl\s+delete\s+(?:ns|namespace|all|pv|pvc)\b", re.IGNORECASE),
    re.compile(r"\bhelm\s+uninstall\b", re.IGNORECASE),
    re.compile(r"\bdd\s+if=\b", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
]

DESTRUCTIVE_WARNING_PREFIX = "⚠️ Destructive action — review carefully before running: "


def filter_destructive_commands(suggested_fix: str) -> str:
    """Scans suggested_fix for destructive terminal or database commands.

    If destructive commands are detected and no warning is present, prepends
    a safety warning notice rather than blocking outright.
    """
    if not suggested_fix:
        return suggested_fix

    is_destructive = any(pattern.search(suggested_fix) for pattern in DESTRUCTIVE_PATTERNS)

    if is_destructive and DESTRUCTIVE_WARNING_PREFIX not in suggested_fix:
        return f"{DESTRUCTIVE_WARNING_PREFIX}{suggested_fix}"

    return suggested_fix
