# SWE-rebench-V2

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/swe-rebench-v2)

## Description

SWE-rebench-V2 is an OpenReward port of the [SWE-rebench V2](https://github.com/SWE-rebench/SWE-rebench-V2) dataset by Badertdinov et al. (Nebius AI). It evaluates agents on real-world software engineering tasks across multiple programming languages. Agents are given a repository checked out to a specific commit and a problem statement, and must modify the source code so that previously-failing tests pass without breaking existing tests. The dataset covers 32K+ instances across Python, JavaScript, Go, Rust, Java, Ruby, and many more languages.

## Community

You can reach out with any questions in Discord: https://discord.gg/V8FqXQ4CgU

## Capabilities

- Multi-language codebase navigation and understanding
- Bug diagnosis from problem statements and test failures
- Source code editing to fix defects
- Reasoning about test expectations and code behavior

## Compute Requirements

Agents are given a sandboxed Docker environment with a pre-built instance image for each task. Default sandbox size is 1 CPU and 2 GB RAM.

## License

[MIT](https://opensource.org/licenses/MIT). The underlying SWE-rebench V2 dataset is subject to its own license terms.

## Tasks

There is one split in this environment:

- **Train**: 32K+ software engineering tasks

Each task provides a repository, base commit, problem statement, and a set of tests that should transition from failing to passing after the agent's fix. Tasks span issue-based and PR-based scenarios across dozens of programming languages and frameworks.

## Reward Structure

This is a multi-turn environment with binary reward:

- **1.0** — All FAIL_TO_PASS tests now pass and all PASS_TO_PASS tests remain passing
- **0.0** — Any required test fails or regresses

On submission, the environment applies the held-out test patch, runs the task's test command, and parses the output using a language/framework-specific log parser to determine per-test pass/fail status.

## Data

Data is loaded from parquet files uploaded to the environment's data directory. Each row contains the instance ID, repository, base commit, test patch, problem statement, Docker image name, language, test expectations (FAIL_TO_PASS and PASS_TO_PASS lists), and install/test configuration. The dataset is derived from the SWE-rebench V2 collection on HuggingFace (`nebius/SWE-rebench-V2`).

## Tools

| Tool | Description |
|------|-------------|
| `bash` | Run bash commands in the sandbox container. |
| `str_replace` | Replace a unique string in a file with another string. |
| `view` | View file contents or directory listings. |
| `create_file` | Create a new file with specified content. |
| `submit_answer` | Submit the solution. Applies the test patch, runs the test suite, and returns reward. |

## Time Horizon

SWE-rebench-V2 is a multi-turn environment. Agents explore the repository, read code, diagnose the issue, make edits, and optionally run tests before submitting. A typical task may involve 10-50+ tool calls depending on complexity.

## Environment Difficulty

SWE-rebench V2 is a challenging benchmark spanning many languages and difficulty levels. Tasks are annotated with difficulty codes. Performance varies significantly by language, framework, and problem complexity. As of the paper's publication, frontier models solve a modest fraction of tasks, with Python tasks being the most commonly attempted.

## Safety

Agents operate within sandboxed Docker containers with no network access to external services. The environment does not involve private data or production systems. Agents can only modify files within the repository checkout; the test patch is applied automatically at submission time and cannot be tampered with.

## Citations

```bibtex
@misc{badertdinov2026swerebenchv2languageagnosticswe,
      title={SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale},
      author={Ibragim Badertdinov and Maksim Nekrashevich and Anton Shevtsov and Alexander Golubev},
      year={2026},
      eprint={2602.23866},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2602.23866},
}
```
