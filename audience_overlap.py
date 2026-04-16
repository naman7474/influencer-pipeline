def compute_audience_overlap(cip_list: list[dict]) -> dict:
    """
    Compute audience overlap between all pairs of creators.

    Uses commenter handles from comment scrapes.
    If two creators share many commenters, their audiences overlap.

    This is CRITICAL for brand campaigns:
    - High overlap = avoid pairing these creators (same eyeballs)
    - Low overlap = good combo for reach maximization
    """
    creator_commenters = {}

    for cip in cip_list:
        handle = cip.get("profile", {}).get("handle")
        if not handle:
            continue
        commenters = set(
            cip.get("comments", {}).get("unique_commenters", [])
        )
        commenters.discard("")
        creator_commenters[handle] = commenters

    handles = list(creator_commenters.keys())
    overlaps = {}

    for i in range(len(handles)):
        for j in range(i + 1, len(handles)):
            h1, h2 = handles[i], handles[j]
            set1 = creator_commenters[h1]
            set2 = creator_commenters[h2]

            if not set1 or not set2:
                continue

            shared = set1 & set2
            union = set1 | set2
            jaccard = len(shared) / max(len(union), 1)
            overlap_coeff = len(shared) / max(min(len(set1), len(set2)), 1)

            pair_key = f"{h1}::{h2}"
            overlaps[pair_key] = {
                "creator_a": h1,
                "creator_b": h2,
                "shared_commenters": len(shared),
                "jaccard_similarity": round(jaccard, 4),
                "overlap_coefficient": round(overlap_coeff, 4),
                "recommendation": _overlap_recommendation(overlap_coeff),
            }

    return overlaps


def _overlap_recommendation(overlap_coeff: float) -> str:
    if overlap_coeff > 0.3:
        return "HIGH_OVERLAP — avoid pairing for same campaign"
    elif overlap_coeff > 0.1:
        return "MODERATE_OVERLAP — some shared audience"
    else:
        return "LOW_OVERLAP — good for reach maximization"
