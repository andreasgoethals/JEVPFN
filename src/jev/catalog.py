"""Current extractor catalogue, separate from fingerprinted historical counters.

Extending this facade must not invalidate completed token audits for older models.
The historical registry and tokenizer implementations remain byte-for-byte frozen.
"""

from src.jev import arena_decisions, registry

ADDED = registry.ADDED  # These five use ExtendedRuntime, not Liquid's native runner.
SELF_HOSTED = (*registry.SELF_HOSTED, "liquid_lfm3", "glide", "mammoth", "drex_max", "knowline", "sieve")
HOSTED_UNAVAILABLE = registry.HOSTED_UNAVAILABLE
PINS = {**registry.PINS, **{key: spec[1] for key, spec in arena_decisions.MODELS.items()}, "liquid_lfm3": "cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41",
        "glide": "4d32743ae96b74f0116f050d37742bbd43c04e7d",
        "mammoth": "61cec44c281c693992e3d6ebeb44dd3402dfa186",
        "drex_max": "005d4eb46cd2f2f56ae0e82c6444fba8363c9c17"}


def asset_urls(provider):
    if provider in arena_decisions.MODELS:
        return arena_decisions.asset_urls(provider)
    if provider in {"mammoth", "drex_max"}:
        from src.jev.frontier_decisions import asset_urls as urls

        return urls(provider)
    if provider == "glide":
        from src.jev.glide import asset_urls as urls

        return urls()
    if provider == "liquid_lfm3":
        from src.jev.liquid_lfm3 import asset_urls as urls

        return urls()
    return registry.asset_urls(provider)


def stage_tokenizers(providers=SELF_HOSTED):
    for provider in providers:
        if provider in arena_decisions.MODELS:
            arena_decisions.stage_tokenizer(provider)
        elif provider in {"mammoth", "drex_max"}:
            from src.jev.frontier_decisions import stage_tokenizer

            stage_tokenizer(provider)
        elif provider == "glide":
            from src.jev.glide import stage_tokenizer

            stage_tokenizer()
        elif provider == "liquid_lfm3":
            from src.jev.liquid_lfm3 import stage_tokenizer

            stage_tokenizer()
        else:
            registry.stage_tokenizers([provider])


def ExactCounter(provider):
    if provider in arena_decisions.MODELS:
        return arena_decisions.Counter(provider)
    if provider in {"mammoth", "drex_max"}:
        from src.jev.frontier_decisions import Counter

        return Counter(provider)
    if provider == "glide":
        from src.jev.glide import Counter

        return Counter()
    if provider == "liquid_lfm3":
        from src.jev.liquid_lfm3 import Counter

        return Counter()
    return registry.ExactCounter(provider)
