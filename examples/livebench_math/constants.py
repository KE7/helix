"""Immutable provenance and subset pins for the LiveBench-Math demo."""

GEPA_BLOG_COMMIT = "121084499247e7ddfa05ec453a53e0d644838b7a"
GEPA_RELEASE_COMMIT = "8b0ce6cd99a234f6b74daf37558a2ac0ce18f975"
TERRARIUM_COMMIT = "e2c8b59079ed26de2d38e8aaf4ac2b4437703fe9"
LIVEBENCH_CODE_COMMIT = "1de6a43e82a137beeeaf2b92d683eedb67f0cf97"
LIVEBENCH_DATA_REVISION = "bb66571c8ccf32d3df9e6f48b920d3770ff4aacb"

FULL_SPLIT_SIZES = {"train": 100, "val": 100, "test": 168}
FULL_SPLIT_SHA256 = {
    "train": "9c447b064f021422e0a922f3024988f5b9185cd29113769fda07ef13e1f0e4af",
    "val": "b55b6cb084d2cfe3f77dda5506cb159eb5eae9cf7d6e86cd0bb2bf6346468860",
    "test": "b26a04a55a05ae293076065e4ed5d8fb6b509ba666d889f416ddd4b05362e971",
}

# One row from each major official scorer family: AIME, AMC, AMPS-Hard,
# and olympiad proof-step ordering. Order is part of the adapter contract.
SMOKE_IDS = {
    "train": (
        "dc1e7754534de44adc73fb52a5bb8669fe2828e61e0069b834a8a6942ad952c5",
        "64950f925b29282781b04e4daeeb3ecf96f1558f18ff2747bb7be0a8be05ec14",
        "c3f6b7718cc440106b768588cd530da88a34d226ee584cc356d1d9e9cd769e3a",
        "c6675bf6647188f84ee445590a494a8d516635bad77ace98ec61d28746839a8d",
    ),
    "val": (
        "4dc5a69ba4f2038bd73182b69e13d3669a77bfdc5fdaf8e41e615fafc51eb359",
        "8825eba85dd830d58905b458d977cb25f9940c6a21746d8250ec85c9e21154df",
        "758e2bd2e027b0e775a8c7795eaef44fe1cb9b7f9868989ed818c7c4acaf67ec",
        "11f95734f602e7d1481f9887ca7fc8bed83258e22fd5c443449ac159a4732115",
    ),
}

SOLVER_MODEL = "gpt-4.1-mini-2025-04-14"
SOLVER_TEMPERATURE = 1.0
SOLVER_MAX_TOKENS = 32_000
SOLVER_TIMEOUT_SECONDS = 180
SOLVER_RETRIES = 0

# The mutation engine only edits prompt.txt. It never sees ground truth and is
# not part of the scored pipeline: the solver below is called directly by the
# protected sidecar and scored by the official LiveBench scorers. Swapping the
# proposer backend therefore cannot change what the benchmark measures.
PUBLICATION_PROPOSER_MODEL = "gpt-5-mini"
# Explicit smoke deviation. The lane originally proposed through a codex runner
# that was never published to GHCR, so it was not reproducible off this host.
# The published, registry-resolvable claude runner is used instead; this also
# retires the codex-only HTTP 400 workaround the gpt-5.4 pin existed for.
SMOKE_PROPOSER_BACKEND = "claude"
SMOKE_PROPOSER_MODEL = "haiku"
