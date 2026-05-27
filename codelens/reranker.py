from __future__ import annotations


def rerank_score(
    query: str,
    chunk_id: str,
    semantic_score: float,
    lexical_score: float,
) -> float:
    score = semantic_score * 0.7 + lexical_score * 0.3

    q = query.lower()
    cid = chunk_id.lower()

    bonuses = [
        ("token", ["token", "auth", "jwt"]),
        ("superuser", ["superuser", "admin"]),
        ("training", ["training_plan", "training_unit"]),
        ("database", ["database", "session", "crud"]),
    ]

    for trigger, keywords in bonuses:
        if trigger in q:
            if any(k in cid for k in keywords):
                score += 0.08

    return score