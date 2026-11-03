# DepGraph Ablation

## Setup

We evaluate Graph2Env without DepGraph on the 200 repository revisions in `datasets/graph2env200.json`. The ablation removes DepGraph construction, graph-to-script compilation, and subsequent graph queries and updates. Instead, the model directly generates the initial Build Script and Runtime Configuration from repository files, without reusing a graph-derived script. The Agent retains feedback-driven execution and repair of the script and runtime configuration. Final artifacts are replayed in fresh containers for independent evaluation. We use `deepseek-v4-flash`, a 7,200-second construction budget per repository, and a budget of 100 LLM calls, as in the full method.

## Results

| Variant | Repositories | EBSR | ESSR |
| --- | ---: | ---: | ---: |
| Graph2Env | 200 | 81.0% | 59.3% |
| Graph2Env w/o DepGraph | 200 | 71.0% (142/200) | 52.9962% |

Compared with the paper-reported full method, this ablated workflow has lower EBSR and ESSR by 10.0 and 6.3038 percentage points, respectively. These results suggest that explicitly representing and maintaining dependency requirements through DepGraph contributes to more reliable environment construction.
