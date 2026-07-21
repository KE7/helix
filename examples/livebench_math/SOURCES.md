# LiveBench-Math source manifest

Every moving input is pinned. The evaluator image downloads the dataset during
the image build; neither benchmark data nor generated results belong in Git.

| Input | Immutable revision | Role |
| --- | --- | --- |
| GEPA parallel-proposals article source | `gepa-ai/gepa@121084499247e7ddfa05ec453a53e0d644838b7a` | Publication configuration and reported comparison |
| GEPA 0.1.4 | `gepa-ai/gepa@8b0ce6cd99a234f6b74daf37558a2ac0ce18f975` | Release associated with the article |
| Terrarium | `gepa-ai/terrarium@e2c8b59079ed26de2d38e8aaf4ac2b4437703fe9` | LiveBench-Math prompt, split procedure, and evaluator semantics |
| LiveBench scorer code | `LiveBench/LiveBench@1de6a43e82a137beeeaf2b92d683eedb67f0cf97` | Official AIME, contest, AMPS-Hard, and olympiad scoring functions |
| LiveBench-Math data | `livebench/math@bb66571c8ccf32d3df9e6f48b920d3770ff4aacb` | The 368-row Hugging Face dataset |

The exact smoke subset is ordered as follows. HELIX IDs `0` through `3` are
positions in each list, not raw question IDs.

| Split | HELIX ID | Question ID | Family |
| --- | ---: | --- | --- |
| train | 0 | `dc1e7754534de44adc73fb52a5bb8669fe2828e61e0069b834a8a6942ad952c5` | `aime_i_2024` |
| train | 1 | `64950f925b29282781b04e4daeeb3ecf96f1558f18ff2747bb7be0a8be05ec14` | `amc_12a_2023` |
| train | 2 | `c3f6b7718cc440106b768588cd530da88a34d226ee584cc356d1d9e9cd769e3a` | `amps_hard_characteristic_polynomial` |
| train | 3 | `c6675bf6647188f84ee445590a494a8d516635bad77ace98ec61d28746839a8d` | `imo` |
| val | 0 | `4dc5a69ba4f2038bd73182b69e13d3669a77bfdc5fdaf8e41e615fafc51eb359` | `aime_i_2024` |
| val | 1 | `8825eba85dd830d58905b458d977cb25f9940c6a21746d8250ec85c9e21154df` | `amc_12a_2023` |
| val | 2 | `758e2bd2e027b0e775a8c7795eaef44fe1cb9b7f9868989ed818c7c4acaf67ec` | `amps_hard_characteristic_polynomial` |
| val | 3 | `11f95734f602e7d1481f9887ca7fc8bed83258e22fd5c443449ac159a4732115` | `imo` |

The full Terrarium-derived split has 100 train, 100 validation, and 168 test
rows. `prepare_data.py` reconstructs it with seed 0 and rejects it unless the
ordered question-ID SHA-256 digests match those in `constants.py`.

## Attribution and licenses

- GEPA is MIT-licensed; copyright belongs to its contributors.
- LiveBench code is distributed under Apache License 2.0; copyright belongs to
  the LiveBench contributors.
- The `livebench/math` dataset card and repository are the authoritative source
  for dataset terms and provenance.
- The pinned Terrarium tree did not contain a top-level license file when this
  demo was prepared. This demo independently implements the documented split
  and routing behavior and does not redistribute the Terrarium repository.

This repository's license does not replace upstream licenses. See the linked
upstream repositories and dataset card before redistributing their materials.
