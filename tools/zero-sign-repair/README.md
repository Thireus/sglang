# Zero-sign repair for low-bit quants

_This guide explains a quiet defect of some low-bit quant types (IQ2_S, IQ3_S and a few others) and how to check and repair it. It hits weights that contain many exact zeros, which is the case of every model whose weights come from a 4-bit float format such as MXFP4 or NVFP4 (about 11 % of the weights are exactly 0), and of pruned weights. The repair changes one sign bit per affected weight and nothing else: it takes seconds per matrix, needs no GPU and no requantisation._

_The tools work on the hot/cold checkpoints that SGLang loads (hot experts in native MXFP4, cold experts as ik_llama.cpp quants, see `python/sglang/srt/model_loader/hotcold_manifest.py`) and on GGUF files._

## In one minute

- The ggml quantisers of IQ2_S and IQ3_S have no zero level, and they write every exact zero with a **positive** sign.
- With 11 % of exact zeros, every row of every matrix ends up with a small positive bias: about **+5.6 % of the mean weight magnitude** for IQ2_S.
- The usual quality number of a quantised matrix, its NMSE, does not change at all, so nothing looks wrong, but the model breaks: on DeepSeek-V4.1-Flash the damaged checkpoint read a corpus KLD of **4.99**, the repaired one **0.056**.
- The fix: give each such weight the sign of its source zero (MXFP4 and NVFP4 have +0 and -0), or alternate the signs when the source only has +0. The bias goes to about 0, the NMSE stays the same to the last digit.

## What goes wrong

Take one row of ten source weights, two of them exactly zero (one +0, one -0), and a quant type whose smallest level is 0.25 (no 0). To keep the example simple, the eight other weights are stored exactly:

```text
source       0.5  -1.0   +0   1.5  -0.5   -0   2.0  -1.5   1.0  -0.5      mean |w| = 0.85
as built     0.5  -1.0 +0.25  1.5  -0.5 +0.25  2.0  -1.5   1.0  -0.5      error sum +0.50
repaired     0.5  -1.0 +0.25  1.5  -0.5 -0.25  2.0  -1.5   1.0  -0.5      error sum  0.00
```

- As built: mean error = +0.05, offset = 0.05 / 0.85 = **+0.059**; squared error 0.125.
- Repaired: mean error = 0, offset = **0**; squared error still 0.125 (that is why the NMSE cannot see the problem).

The real IQ2_S blobs of DeepSeek-V4.1-Flash measured an offset of +0.054 to +0.058 on every one of their 23,040 cold matrices, and 100 % of their rows had a positive summed error.

Why a small bias matters: a layer computes `out = W @ in`. An error `E` with a mean `b` (the offset times the mean weight magnitude) adds `b x sum(in)` to every output, the same number for every output row. Hidden states are not centred, so `sum(in)` over 5,120 channels is large, and the layer answers the *mean of its input* with a constant vector:

```text
   input (5120 channels, mean m != 0)
        |
        v
  [ W + E ]   E = random error (harmless, averages out)  +  b on every weight (the zero signs)
        |
        v
  output = W.in  +  random noise  +  b * 5120 * m  on EVERY output row
                                     ^^^^^^^^^^^^^
                    the same constant vector, whatever the text says;
                    the next layer sees it in its input mean, adds its own,
                    and after a few dozen layers it dominates the hidden state
```

On DeepSeek-V4.1-Flash, an all-ones input of unit route weight gave a cold expert of layer 20 an output mean of +0.20 to +0.23 with the damaged blobs, against -0.004 to +0.002 for the source weights. Served, the model produced a nearly constant output distribution on 37 % of the positions (corpus KLD 4.99, PPL 15,439).

## Who is affected

A model is exposed when its source weights contain many exact zeros: weights dequantised from MXFP4 / NVFP4 (the experts of DeepSeek-V4.1-Flash: 11.4 to 11.8 % exact zeros; any fp4 release converted to a BF16 GGUF), or pruned / sparsified weights. A BF16 model trained in BF16 normally has almost no exact zeros and is not affected.

The type decides the rest. Measured on DeepSeek-V4.1-Flash experts (MXFP4 source, 11.5 % exact zeros, three matrices per type, quantised by ik_llama.cpp's libggml as built, unit importance weights):

| type | bpw | offset as built | zeros that come out positive | verdict | this tool |
|---|---:|---|---|---|---|
| IQ2_S | 2.5625 | +0.0561 to +0.0572 | 100 % | affected | repairs it (its bytes equal those of the repaired checkpoint measured below, checked on a full layer) |
| IQ3_S | 3.4375 | +0.0247 to +0.0252 | 100 % | affected | repairs it (one sign bit per weight, like IQ2_S; tested on matrices, not on a served model) |
| IQ3_KT | 3.1313 | +0.0214 to +0.0220 | 94 % (6 % decode to 0) | affected | refuses: trellis code, no sign bits; requantise |
| IQ2_XXS | 2.0625 | +0.0204 to +0.0210 | 67 % | affected | refuses: parity-signed; requantise |
| IQ2_XS | 2.3125 | +0.0191 to +0.0198 | 67 % | affected | refuses: parity-signed; requantise |
| IQ3_XXS | 3.0625 | +0.0092 to +0.0094 | 67 % | affected | refuses: parity-signed; requantise |
| IQ2_K | 2.375 | +0.0006 to +0.0007 | 50 % | clean | nothing to do |
| IQ2_KT | 2.1313 | +0.0001 to +0.0003 | 48 % (4 % decode to 0) | clean | nothing to do |
| IQ2_KS | 2.1906 | -0.0002 to +0.0002 | 50 % | clean | nothing to do |
| IQ2_KL | 2.6906 | -0.0002 to +0.0001 | 50 % | clean | nothing to do |
| IQ3_K | 3.4375 | +0.0000 to +0.0001 | 50 % | clean | nothing to do |
| IQ3_KS | 3.1906 | -0.0001 to +0.0000 | 50 % | clean | nothing to do |
| IQ1_S | 1.5625 | -0.0005 to -0.0003 | 50 % | clean | nothing to do |

"Parity-signed" (IQ2_XXS, IQ2_XS, IQ3_XXS): these types store 7 signs per group of 8 weights and the 8th sign is implied (an even number of negatives), so one zero cannot change sign alone. Other types (Q4_K, Q8_0, ...) have a zero level or no sign field this tool knows; check them anyway, the check works for every type libggml can dequantise.

Use the check whatever the type: it measures, it does not assume.

## Requirements

```text
CPU: YES
GPU: NO
RAM: about 0.3 GB per worker (one matrix at a time; 8 workers by default)
Python 3 with numpy (nothing else)
ik_llama.cpp's libggml.so (any build: CPU only is enough)
```

Make sure you edit and paste these env variables in any terminal session you'll be using:

```bash
WORKING_DIRECTORY='/AI'                                       # Full path please!
IK_LLAMA="$WORKING_DIRECTORY/ik_llama.cpp"                    # ik_llama.cpp sources
IK_BUILD="$IK_LLAMA/build"                                    # where libggml.so is built
export IK_LIBGGML="$IK_BUILD/ggml/src/libggml.so"             # the tools read this variable
TOOLS="$WORKING_DIRECTORY/sglang/tools/zero-sign-repair"      # this folder of the SGLang repository
CKPT="$WORKING_DIRECTORY/models/MyModel-hotcold"              # the hot/cold checkpoint to check
SOURCE="$WORKING_DIRECTORY/models/MyModel"                    # its full-precision source (MXFP4 / BF16 safetensors)
FIXED="$WORKING_DIRECTORY/models/MyModel-hotcold-fixed"       # where the repaired checkpoint goes (new directory)
LAYERS='all'                                                  # or 22, or 0-3,7
```

## Prepare libggml

Get the ik_llama.cpp sources (skip if you already have them):

```sh
cd "$WORKING_DIRECTORY"
git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp --recursive ik_llama.cpp
```

Build only the ggml shared library, CPU only (about 10 minutes on 16 cores):

```bash
cmake -S "$IK_LLAMA" -B "$IK_BUILD" -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=ON && \
cmake --build "$IK_BUILD" --config Release -j$(nproc) --target ggml
ls -la "$IK_LIBGGML"
```

You should see `[100%] Built target ggml` and the file `libggml.so`. A CUDA build of ik_llama.cpp works too: point `IK_LIBGGML` at its `libggml.so` (often under `build/bin/`).

Check that the tools can load it, on a synthetic matrix:

```bash
python3 "$TOOLS/zero_sign_check.py" demo
```

You should see:

```text
synthetic matrix 256 x 1024, 11.9 % exact zeros (half +0, half -0), quantised to IQ2_S
as quantised: offset +0.06034, rows with a positive summed error 1.000, zeros decoded positive 1.000, NMSE 0.076684 -> OFFSET
repaired:     offset +0.00050, rows with a positive summed error 0.531, zeros decoded positive 0.500, NMSE 0.076684 -> PASS
```

The NMSE is the same number on both lines; only the offset moves. `--type IQ3_S` shows the same for IQ3_S, `--type IQ3_KS` shows a clean type.

## Step 1: check

A quick look first, 8 cold experts per layer:

```bash
python3 "$TOOLS/zero_sign_check.py" hotcold --ckpt "$CKPT" --source "$SOURCE" --layers "$LAYERS" --sample 8
```

You should see one row per layer and projection. A damaged IQ2_S checkpoint looks like this (layer 22 of the first DeepSeek-V4.1-Flash hot/cold build):

```text
layer   proj     type   units    zeros        offset min..max   rows_pos   zero_neg       NMSE  verdict
22        w1    IQ2_S       8   11.57%     +0.05574..+0.05685      1.000      0.000    0.06870  OFFSET:8
22        w3    IQ2_S       8   11.63%     +0.05536..+0.05677      1.000      0.000    0.06854  OFFSET:8
22        w2    IQ2_S       8   11.73%     +0.05654..+0.05770      1.000      0.000    0.06883  OFFSET:8
RESULT: FAIL (run zero_sign_repair.py hotcold)
```

How to read it:

- `zeros`: the share of source weights that are exactly 0. Under about 1 % the problem cannot be large.
- `offset`: mean(quant - source) / mean|source|. Healthy: within +/-0.005 (change with `--max-offset`).
- `rows_pos`: the share of rows whose summed error is positive. Healthy: near 0.5. `1.000` means every row leans the same way.
- `zero_neg`: the share of source zeros that decode negative. `0.000` is the signature of this defect.

The exit code is 0 when everything passes, 1 when a matrix has an offset, 2 when the inputs could not be read. Drop `--sample 8` to check every cold matrix: about 2 minutes per layer of 576 matrices with 8 workers (`--workers N` to change); the repair takes 6 to 9 minutes per such layer. `--json FILE` writes every number.

## Step 2: repair

See what would happen, nothing is written:

```bash
python3 "$TOOLS/zero_sign_repair.py" hotcold --ckpt "$CKPT" --source "$SOURCE" --layers "$LAYERS" --out "$FIXED" --dry-run
```

Then repair into a new directory. The repaired cold files are written there; every other file is hard-linked from `$CKPT` (no extra disk space; `--link copy` if the new directory is on another disk):

```bash
python3 "$TOOLS/zero_sign_repair.py" hotcold --ckpt "$CKPT" --source "$SOURCE" --layers "$LAYERS" --out "$FIXED"
```

You should see, per layer:

```text
layer 22: model-cold-00023-of-00040.safetensors repaired and verified; 576 matrices, offset +0.05416..+0.05825 -> -0.00033..+0.00028; sign bits set 395,774,369, cleared 0; sha256 21900f26c4c0fa32 (514 s)
hotcold.json written with the new sha256 of 3 cold tensors
RESULT: PASS (run zero_sign_check.py hotcold on the destination to see it)
```

What it did, for each cold matrix:

1. read the source matrix and find its exact zeros, with their sign (`+0` / `-0`);
2. pick the rule (`--rule auto`, the default): the source's own sign when 25-75 % of the zeros are -0 (native MXFP4 / NVFP4 checkpoints), else the alternating rule (sources with +0 only, see the questions below);
3. set the sign bit of those weights, and of no other weight; grid indices and scales are untouched;
4. re-read the written file and verify every matrix: |offset| <= 0.005, NMSE unchanged, no byte outside the sign fields changed. Only then is the file renamed into place.

Every matrix goes into `$FIXED/zero-sign-repair.jsonl` (offset before and after, bits set, rule). `hotcold.json` gets the new sha256 of the cold files and tensors and a `zero_sign_repair` record. Interrupted? Run the same command again: the repair is idempotent.

Rather repair in place? Use `--in-place` instead of `--out "$FIXED"`. Each file is still written to `<file>.zsrepair.partial` first and renamed only after its verification; another hard link to the old file (a sibling checkpoint) keeps the old bytes:

```bash
python3 "$TOOLS/zero_sign_repair.py" hotcold --ckpt "$CKPT" --source "$SOURCE" --layers "$LAYERS" --in-place
```

## Step 3: verify

```bash
python3 "$TOOLS/zero_sign_check.py" hotcold --ckpt "$FIXED" --source "$SOURCE" --layers "$LAYERS" --verify-sha
```

You should see every row PASS, `zero_neg` near 0.5, `rows_pos` near 0.5, the same NMSE as before, and the files matching `hotcold.json`:

```text
layer   proj     type   units    zeros        offset min..max   rows_pos   zero_neg       NMSE  verdict
22        w1    IQ2_S     192   11.59%     -0.00033..+0.00026      0.499      0.500    0.06868     PASS
22        w3    IQ2_S     192   11.64%     -0.00026..+0.00025      0.500      0.500    0.06858     PASS
22        w2    IQ2_S     192   11.72%     -0.00030..+0.00028      0.500      0.500    0.06879     PASS
layer 22: model-cold-00023-of-00040.safetensors sha256 21900f26c4c0fa32 matches hotcold.json (file and its 3 cold tensors)
576 matrices, 0 with an offset beyond 0.005; largest |offset| -0.00033 (22.w1.305); 122 s
RESULT: PASS
```

What it gave on DeepSeek-V4.1-Flash (192 hot experts per layer in MXFP4 + 192 cold in IQ2_S; KLD against the unpruned model and PPL on `imatrix-calibration-corpus-v02.txt`, 812 windows of 512 tokens, scored on tokens 257-512):

| checkpoint | cold offset | corpus KLD | PPL | size |
|---|---|---:|---:|---:|
| as built (zeros all positive) | +0.054 to +0.058 | 4.989 | 15,439 | 354.65 GB |
| repaired (zeros take their source sign; this tool writes the same bytes) | -0.0004 to +0.0004 | **0.0564** | **3.7708** | 354.65 GB |
| for reference: the same model with 64 experts per layer deleted | - | 0.1637 | 4.0695 | 363.82 GB |
| for reference: the unpruned model | - | 0 | 3.6977 | - |

The repaired checkpoint has the same size: only sign bits changed.

## GGUF files

The same check and repair work on GGUF files: a quantised GGUF (or a set of shards) against its source GGUF (F32, F16, BF16 or MXFP4), tensors matched by name. Set:

```bash
QUANT_GGUF="$WORKING_DIRECTORY/MyModel-IQ2_S-SPECIAL_SPLIT/MyModel-IQ2_S-SPECIAL_TENSOR-00042-of-00321.gguf"   # one or more files
SOURCE_GGUF="$WORKING_DIRECTORY/MyModel-BF16-SPECIAL_SPLIT/MyModel-BF16-SPECIAL_TENSOR-00042-of-00321.gguf"   # the matching source file(s)
FIXED_GGUF_DIR="$WORKING_DIRECTORY/MyModel-IQ2_S-fixed"
```

Check (exit code 0 / 1 / 2 as above):

```bash
python3 "$TOOLS/zero_sign_check.py" gguf --quant $QUANT_GGUF --source $SOURCE_GGUF
```

Repair into a new folder (only the files that change are written there), then check the result:

```bash
python3 "$TOOLS/zero_sign_repair.py" gguf --quant $QUANT_GGUF --source $SOURCE_GGUF --out "$FIXED_GGUF_DIR"
python3 "$TOOLS/zero_sign_check.py" gguf --quant "$FIXED_GGUF_DIR"/*.gguf --source $SOURCE_GGUF
```

You should see one line per repaired tensor, for example (one DeepSeek-V4.1-Flash expert, BF16 source made from its MXFP4 weights):

```text
  blk.22.ffn_gate_exps.0.weight (IQ2_S, rule source): offset +0.05588 -> -0.00001, rows_pos 1.000 -> 0.510, NMSE 0.069101 (unchanged), sign bits set 672,123 cleared 0 -> verified
```

Quantised tensors of a type the tool cannot repair are checked and listed as `NOT REPAIRED` with the reason; the exit code is then 1. Float tensors are skipped. `--tensors REGEX` limits the work to some tensors; `--in-place` replaces the files where they are (through a `.zsrepair.partial` copy).

Good to know: a BF16 GGUF converted from an MXFP4 checkpoint keeps the -0 of the fp4 codes (BF16 has a sign bit), so the `source` rule applies. An MXFP4 GGUF written by ggml's own quantiser has +0 only (its quantiser never writes code 8: a real MXFP4 GGUF tensor of a 27B model had 8.2 % exact zeros, none of them -0); the `auto` rule then picks the alternating rule, which took its IQ2_S copy from an offset of +0.0397 to -0.00003.

## Tests

```bash
python3 "$TOOLS/test/test_zero_sign.py"
```

You should see `OK`, in seconds. Without `IK_LIBGGML` the tests that need libggml are reported as skipped.

## Questions and answers

**Why not just requantise with a fixed quantiser?** You can, for the types with a sign field the result is the same bytes (proved on two full layers of DeepSeek-V4.1-Flash: requantised with the sign rule inside the quantiser = repaired by this tool, sha256 identical). The repair is much faster (the quantiser search is the slow part) and leaves everything else bit for bit as it was, so nothing else needs re-validating.

**Why the source's sign and not "always alternate"?** The sign of an fp4 zero is the sign of the small value that was rounded to it, so it is the best guess of the true sign; it also keeps about half of the zeros on each side in these models (49.9-50.0 % are -0). Alternating is the fallback for sources that kept no sign.

**What is the alternating rule exactly?** Along each row the zeros take turns, +, -, +, - ..., and the first sign alternates from one row to the next, so rows with an odd number of zeros do not all lean the same way. It is deterministic: the same matrix always gets the same bits. On a synthetic matrix with 12 % zeros, all +0, it takes the IQ2_S offset from above +0.02 to within +/-0.003 (the test `test_repair_plus_zero_only_with_the_alternating_rule`).

**Does an imatrix help?** No: the quantiser picks the sign with `x >= 0` before it looks at the importance weights, so a build with an imatrix gets the same positive zeros. Check it and repair it the same way.

**My checkpoint is a GGUF with IQ2_XXS / IQ2_XS / IQ3_XXS. What now?** These types cannot be repaired by setting bits. Requantise from a source whose exact zeros were first given alternating signs, as tiny values (for example +/-1e-20), so the quantiser sees a sign to keep; then check the result with `zero_sign_check.py gguf`. That route is not part of this tool and was not measured for these types.

**Is the 0.005 threshold right?** Every clean type measured within +/-0.0007, every affected one at +0.009 or more. A repaired IQ2_S matrix lands within +/-0.0004.

**Does it change speed or size?** No. Same files, same sizes, same kernels; only sign bits differ.

## Limits

- Repairs IQ2_S and IQ3_S only (one explicit sign bit per weight). The IQ3_S repair is tested on synthetic matrices and on one real DeepSeek-V4.1-Flash expert matrix (offset +0.0246 -> -0.00001, NMSE unchanged), not on a full IQ3_S checkpoint or in the engine.
- The source must be available: the tool needs to know where the exact zeros were. Without the source the zeros cannot be told apart from small weights.
- Hot/cold checkpoints: the source is read with the expert names `layers.{L}.ffn.experts.{E}.{w}` (MXFP4 `weight` + `scale`, or a BF16 / F16 / F32 `weight`); `--source-pattern` changes the pattern.
- GGUF: tested on files written by this tool's own small writer and on a real MXFP4 GGUF tensor written by llama.cpp tooling (as the source); not yet on a quantised GGUF written by `llama-quantize`. The reader was cross-checked against `gguf-py` on 212 real tensors (same offsets, types, shapes, sizes).
- The rows of a quantised tensor must be a multiple of 256 values (always true for these types).
- The pass mark is the offset of the whole matrix (0.005 at most). A quantiser can carry a small bias of its own that has nothing to do with zeros (seen with IQ3_S on a matrix whose weights have a non-zero mean: +0.007 with no zero in it at all). Such a matrix still reads OFFSET after a correct repair, and in hot/cold mode the layer file that holds it is then left as it was. Look at `zero_neg` in the table: about 0.500 after the repair means the zero signs are fixed and what remains is the type's own bias; requantise with another type if it matters.
