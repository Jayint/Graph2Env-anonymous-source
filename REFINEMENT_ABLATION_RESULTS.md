# Refinement Ablation

## Setup

We evaluate Graph2Env without feedback-driven refinement on the 200 repository revisions in `datasets/graph2env200.json`, using a 7,200-second construction budget per repository. The ablation retains repository analysis, dependency resolution, probing, initial DepGraph construction, and setup-script compilation. It executes the initial setup and verifies test collection in a fresh container, stopping on failure without Agent-driven diagnosis or further graph, script, or runtime repair.

## Results

| Variant | Repositories | EBSR | ESSR |
| --- | ---: | ---: | ---: |
| Graph2Env  w/o  refinement | 200 | 22.50% (45/200) | 15.52% |

EBSR requires successful construction, an independent Docker image build, and successful test collection. ESSR is the mean test pass ratio across all 200 repositories, with unsuccessful or unmeasurable cases contributing zero. 

