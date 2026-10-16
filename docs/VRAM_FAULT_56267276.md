# GPU memory fault on Vast 56267276 (2026-10-16)

GPU: NVIDIA GeForce RTX 4090, UUID GPU-168b0ee8-2947-feac-8089-e16aa6bc4123, 24,564 MiB, driver 615.71.09,
ECC not available (GeForce). Instance kept running (user order: no Stop/Destroy). Evidence copies with sha256:
`D:\Workspace\DMRL_artifacts\p1\from_56267276\` (and `/workspace/apsp/vram_diag/` on the instance).

## 1. What was measured

| Test | Code | Sync | Result |
|---|---|---|---|
| A. `scripts/vram_check.py` (p1_env, 16:2xZ) | torch 2.4.0+cu121; 21 separate `torch.empty` float32 blocks of 1 GiB, each `fill_(i+1)` + strided write of `arange % 977` every 4096th element | `torch.cuda.synchronize()` after all writes, before verification | pattern errors in blocks 4–9 (verdict `check`) |
| B. `scripts/vram_diag_torch.py` (16:3xZ) | torch; 19 blocks, plain `fill_(i+1)` only; GPU compare in 16 Mi-element chunks **and** host compare of the first bad 1 MiB window after `.cpu()` with numpy | same | 2/2 repetitions: 6 corrupted blocks each; GPU 22k–259k wrong values per block; host re-check confirms (7–512 wrong in the 1 MiB window) |
| C. `scripts/cuda_memtest.cu` (16:32:57Z), **independent of torch** | nvcc 12.8, sm_89; one `cudaMalloc` of 24.29 GB; write kernel of an index-derived xorshift 32-bit word (normal/inverted, new seed each pass); separate verify kernel recomputes the word from the index; first mismatches re-read with `cudaMemcpy` and compared on the CPU | `cudaDeviceSynchronize()` (checked) between write and verify kernels | 4/4 passes: 1,025,739 – 1,100,549 mismatching words (≈ 0.017 % of 6.07 G words), total 4,249,955; first errors at byte offset ≈ 4.66 GiB of the allocation |

Examples (expected → actual):
- B, block 5 (`data_ptr 0x7ef9a4000000`, expected 6.0 = `40c00000`): byte offset 35,840 (`0x7ef9a4008c00`) reads
  `40800000` = 4.0. Block 8 (expected 9.0 = `41100000`), offset 700,842,784: `41980088` = 19.000259.
- C, pass 0: offset 5,002,081,536: expect `37b2141e`, GPU `37b3141b`, host re-read `37b3141b` (xor `00010005`).
  Pass 1, offset 5,015,432,484: GPU read `487f8874`, host re-read `487f8875` (two reads of the same word differ).

| D. **Third-party** `cuda_memtest` (ComputationalRadiationPhysics, commit `e94e1ee5…`, v1.2.3, 16:37:52Z) | Test1 own-address, Test2 moving inversions ones/zeros, Test4 moving inversions random; 1 pass | tool's own | **Test1: 564,833 errors in 35 blocks of 1 MiB, first block 4736 (≈ 4.6 GiB, same region as C)**; second read equals the wrong value (e.g. expected `0x7f59a0cfdfe8`, read `0x107f19a08bdfb9`, diff `0x10004000440051`); Test2/Test4 (uniform patterns): 0 errors. Log sha256 `88723b13…` |
| E. `scripts/vram_gate.sh` self-test (D + C combined, 16:4xZ) | as D and C | — | FAIL (third-party: 455 blocks with errors in that run) |

The fault is data-dependent: patterns that vary from word to word (own address, xorshift, neighbouring blocks with
different fill values) fail; uniform all-ones/all-zeros/single random word moving inversions pass.

## 2. Review of the original script (`vram_check.py`)

- Aliasing / reuse: each block is its own allocation held in a list until the end; the caching allocator cannot
  hand the same memory out twice while referenced. B asserts distinct pointers and no overlap (`ptr_overlap: false`).
- Indexing: 268,435,456 elements = 65,536 rows × 4096; `view(-1, 4096)[:, 1:]` covers every element except the
  strided column, which is checked separately against the same `arange % 977`. No off-by-one found.
- Read before write completes: all writes and reads are on the default stream (stream-ordered) and a device-wide
  synchronize separates them; B and C also verify after an explicit sync, C with an unrelated pattern.
- Weakness: A only reports block indices; B and C add offsets/values. The script itself is not the cause: C uses no
  torch code, other patterns and a different allocation, and still fails.

## 3. Suspected cause and confidence

- **Measured (high confidence):** this GPU returns wrong data from a region of its VRAM (≈ 4.6 GiB onward),
  reproducibly: 6/6 torch repetitions, 4/4 passes of our CUDA test and the third-party cuda_memtest Test1, confirmed
  by host-side reads, with occasional unstable re-reads.
- **Suspected (moderate):** a faulty memory device / data lane: the flipped bits concentrate on fixed positions within
  each 32-bit word (xor masks drawn from bits 0, 2, 4, 6, 16, 18, 20, 22, e.g. `00550055`, `00400000`), and wrong
  values often equal a neighbouring block's fill value. Overheating/overclock cannot be excluded from inside the
  container (idle 50 °C, P8 at the time of reading; memory clocks not adjustable here).
- **Not established:** whether the host provider's other GPUs are affected.

## 4. Consequence

No calibration, bank or benchmark runs on this GPU: latents, images and scores could be silently wrong. Proposal:
per the user's rule (independent check also fails → 1× RTX 4090 24 GB on another host). On the new machine
`scripts/launch_project.sh` runs `scripts/vram_gate.sh` (third-party + own CUDA test, torch-free) BEFORE any setup or
model download, and `p1_env` repeats the torch check and the CUDA test.
