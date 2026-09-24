# Benchmark manifests

The dataset manifests contain repository names, immutable commit pins, and supplied test counts. Third-party source repositories are fetched at runtime, not redistributed in this package.

| Manifest | Repositories | Composition |
| --- | ---: | --- |
| `ratbench100.json` | 100 | Original 50 followed by additional 50, preserving row contents and order |
| `envbench100.json` | 100 | Selected EnvBench Python repositories |

`full_name` identifies the upstream repository, `commit` pins its revision, and `test_count` supplies the reference denominator for assertion-level evaluation. The evaluation code is not included in this method release; these manifests are supplied as input metadata. The single-repository builder does not read them automatically.

Test-count certification artifacts and verification results are not distributed in this release. Dataset repository names, upstream owners, and commit pins are retained because they are required for reproduction.
