# Pattern adoption

Declared per references/adoption-format.md; IDs resolve against
references/catalog.md in pattern-standards.

| Pattern | Status | Evidence |
|---|---|---|
| P102 | partial | PROVENANCE.md:3 (dataset provenance and license boundaries documented; no warehouse ontology layer) |
| P110 | adopted | src/shard.rs:1 (shard format v1 spec; version refusal at src/shard.rs:186) |
| P111 | adopted | trainer/model.py:3 (architecture contract mirrored bit-for-bit by the engine's C++ inference) |
