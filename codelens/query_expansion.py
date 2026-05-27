from __future__ import annotations

EXPANSIONS = {
    "token": ["jwt", "access token", "auth", "authentication"],
    "superuser": ["admin", "permissions", "role"],
    "database": ["db", "session", "sqlalchemy", "postgres"],
    "training plan": ["plan", "training unit", "owner"],
}


def expand_query(query: str) -> str:
    q = query.lower()

    additions: list[str] = []

    for key, values in EXPANSIONS.items():
        if key in q:
            additions.extend(values)

    if "токен" in q:
        additions.extend(["jwt", "access token", "authentication"])

    if "суперпольз" in q:
        additions.extend(["superuser", "admin", "permissions"])

    if "база данных" in q:
        additions.extend(["database", "sqlalchemy", "session"])

    return query + " " + " ".join(additions)