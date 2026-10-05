# Contributing

- Rust: cargo fmt + clippy -D warnings; tests in tests/ (cargo test).
- Python: ruff check (line 100); pytest tests_ci/ must stay green.
- Any change to model.py / export_*.py / netq-relevant layout MUST run the
  parity harnesses before push (see ARCHITECTURE.md Contracts).
- Armed pipeline scripts in /home/wyatt/data are live: never edit a script
  a running process may re-read; kill, edit, relaunch.
- Research verdicts: pre-register gates in the engine repo docs/ before
  measuring; append results to docs/RESULTS.md.
