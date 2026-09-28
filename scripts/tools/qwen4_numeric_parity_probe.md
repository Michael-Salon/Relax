# Isolated Qwen4 online sync value probe

The producer change is based on Relax `2a2f8294`; the paired SGLang branch
`relax/qwen4exp-numeric-parity` is based on `352f6704`. The two probe modules
follow the same sampling contract. Their source byte hashes differ because
the repositories enforce different import spacing, so check the contract and
the focused CPU tests before a runtime test.

Set `QWEN4_NUMERIC_PARITY=1` in both producer and SGLang worker environments,
along with existing `RELAX_QWEN4_SYNC_AUDIT_DIR` and
`SGLANG_QWEN4_SYNC_AUDIT_DIR`, using two fresh directories. With the numeric
flag unset, audit payloads and weight loading follow the prior behavior.

For each weight version and SGLang IPC TP rank, the producer records the
Bridge-exported HF values before IPC. The consumer records the received values
and reads the actual parameter after its loader. Three BF16 norm vectors are
hashed in full on all eight ranks: visual block 0 norm1, language layer 3 QSA
indexer q_layernorm, and language layer 3 self-attention q_norm. On SGLang TP
rank 0, the probe also hashes 68 cells of layer 1 PLE shard 0 after copying
global rows 0..125014 into the local embedding. The exact coordinates are
17 evenly spaced rows and columns 0, 53, 106, 159. The probe is pinned to
the reduced 8L VL checkpoint shapes, BF16 dtype, and one colocated IPC engine.

Run `scripts/tools/check_qwen4_numeric_parity.py PRODUCER_DIR CONSUMER_DIR`
after a successful two-version run. It requires v1/v2 x ranks 0..7, 92 buckets
and 12,939 observed keys per manifest, correct source-rank mapping, loaded
status, and equal source/received/stored hashes. It reports which selected
samples differ between versions; an empty changed list does not prove an
optimizer update. Run the existing full metadata parity and fixed HF index
coverage checkers as separate gates.

The consumer manifest records the actual destination parameter name supplied
at the loader call. The checker requires the pinned visual/QSA/backbone/PLE
mapping in addition to matching bytes, so a same-shaped unrelated parameter
cannot make a sampled value look valid.

This is a value *sample* check, not full-tensor numerical equality. The three
small norms are fully covered; the large PLE shard has 68 sampled cells.
Vision, QSA, and backbone hashes alone do not prove generation used the new
weights. The runtime gate should include a rollout after v2 if that behavior
is needed.
