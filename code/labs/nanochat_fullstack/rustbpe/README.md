# Component - rustbpe

## Summary
Lightweight Rust tokenizer-training library that complements the broader NanoChat stack. It is a component doc, not a benchmark-pair lab.

## Problem
Tokenizer training is often either too slow and simple or too feature-heavy and opaque. `rustbpe` exists to keep the implementation lightweight, reasonably fast, and easy to understand.

## What This Component Is
- a Rust library for training a GPT-style tokenizer
- part of the broader NanoChat project tree
- focused on simple implementation and practical speed, not benchmark-harness integration

## Why This Is Not A Benchmark Pair
`rustbpe` is a supporting component, not a baseline/optimized benchmark lab. The right contract here is build/test clarity plus how it fits into NanoChat, not a fabricated performance delta section.

## Learning Goals
- Keep the tokenizer-training component visible and understandable inside the larger NanoChat tree.
- Document how to build and test the Rust component without pretending it is a harness benchmark.
- Make the component's role in the broader full-stack project easy to find.

## Directory Layout
| Path | Description |
| --- | --- |
| `Cargo.toml`, `Cargo.lock` | Rust package metadata and dependency lockfile. |
| `src/lib.rs` | Tokenizer-training implementation. |
| `../README.md`, `../README_FAST.md` | Broader NanoChat project docs that explain how this component fits into the full stack. |

## Building and Testing
Use Cargo directly for this component.
```bash
cd labs/nanochat_fullstack/rustbpe
cargo build --release
cargo test
```
- This is a Rust-native component workflow, not a harness-target workflow.
- The crate is pinned to Rust Edition 2021 so it builds on stable Cargo toolchains used by the repo test environment.
- Use the parent NanoChat docs for end-to-end training/inference context.

## Validation Checklist
- `cargo build --release` should compile the library cleanly on the repo's supported stable Rust toolchain.
- `cargo test` should keep the component healthy as the NanoChat tree evolves.

## How It Fits Into NanoChat
`rustbpe` is the tokenizer-training companion inside the broader [labs/nanochat_fullstack/README.md](/home/cfregly/ai-performance-engineering/code/labs/nanochat_fullstack/README.md) tree.

- Use [labs/nanochat_fullstack/README.md](/home/cfregly/ai-performance-engineering/code/labs/nanochat_fullstack/README.md) for the measured inference-stack story.
- Use [labs/nanochat_fullstack/README_FAST.md](/home/cfregly/ai-performance-engineering/code/labs/nanochat_fullstack/README_FAST.md) for the quicker project walkthrough.
- Use this component doc when you only need the tokenizer-training piece.

## Notes
- This doc intentionally uses the same generator path as the benchmark-facing labs so the repo stays tidy, even when the component itself is not a benchmark pair.
- The crate no longer requires a nightly-or-newer Cargo parser just to read `Cargo.toml`; keep it 2021-compatible unless the code actually needs a newer edition feature.
