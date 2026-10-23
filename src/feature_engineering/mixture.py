"""Source ordering and sampling weights for the tokenizer training corpus."""

from typing import Final

# Part of the corpus fingerprint: bump when a change to allocate_budget alters the sampling.
VERSION = 1

# Deduplication keeps the first occurrence of a document in this order, so
# curated sources win over the web crawls that frequently copy from them.
SOURCE_PRIORITY: Final[list[str]] = [
    "hr-wikipedia",
    "multi-eurlex-hr",
    "parlamint-hr-5",
    "parlaspeech-hr",
    "classla-web-hr-2",
    "macocu-hr-2",
    "hrwac-2.1",
    "fineweb-2-hrv-latn",
    "hplt-3.0-hrv-latn",
    "cc-100-hr",
]

# Target share of training bytes per source. Web crawls are downweighted
# relative to their raw size and edited text (encyclopedic, legal,
# parliamentary) is upweighted. Budget a source cannot fill is redistributed
# proportionally to the remaining sources.
MIXTURE_WEIGHTS: Final[dict[str, float]] = {
    "hr-wikipedia": 0.05,
    "multi-eurlex-hr": 0.02,
    "parlamint-hr-5": 0.04,
    "parlaspeech-hr": 0.01,
    "classla-web-hr-2": 0.12,
    "macocu-hr-2": 0.10,
    "hrwac-2.1": 0.08,
    "fineweb-2-hrv-latn": 0.18,
    "hplt-3.0-hrv-latn": 0.30,
    "cc-100-hr": 0.10,
}


def allocate_budget(available_bytes: dict[str, int], target_bytes: int) -> dict[str, float]:
    """Return the sampling fraction per source that best matches the mixture weights.

    Sources whose weighted budget exceeds what they have are taken in full and
    their leftover budget is shared among the others by weight.
    """

    fractions: dict[str, float] = {}
    remaining = {source: MIXTURE_WEIGHTS.get(source, 0.0) for source, size in available_bytes.items() if size > 0}
    budget = float(target_bytes)

    while remaining:
        total_weight = sum(remaining.values())
        if total_weight <= 0:
            break
        saturated = {
            source for source, weight in remaining.items() if budget * weight / total_weight >= available_bytes[source]
        }
        if not saturated:
            for source, weight in remaining.items():
                fractions[source] = budget * weight / total_weight / available_bytes[source]
            break
        for source in saturated:
            fractions[source] = 1.0
            budget -= available_bytes[source]
            del remaining[source]

    return {source: fractions.get(source, 0.0) for source in available_bytes}
