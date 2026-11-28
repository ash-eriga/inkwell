# Front-4E Phase 1: K8V4 layout microbench

Decision gate passed.  The bit-exact winner is `hm_vec2`: head-major 16-row
paired tiles with one `uint4` K load and one `uint2` V load covering rows
`s` and `s+8` for each stream.  It uses the same 816 MiB as token-major K8V4
and produced `maxabs=0 relL2=0` against production `row128` at chunk 256.

| variant | chunk 256 run A ms | chunk 256 run B ms | bit-exact vs row128 |
|---|---:|---:|---|
| token-major `row128` control | 6.832 | 6.826 | yes (control) |
| head-major `hm_row128` | 6.312 | 6.313 | yes |
| head-major `hm_m16` | 6.442 | 6.439 | yes |
| head-major packed tile8 `hm_pack` | 6.220 | 6.219 | yes |
| head-major paired tile16 `hm_vec2` | **6.127** | **6.131** | yes |
| `hm_vec2`, chunk 512 | 5.696 | — | yes vs same-chunk control, but different split reduction partitioning |
| memory-only token rows | 5.967 | 5.969 | diagnostic |
| memory-only head-major rows | 4.935 | 4.939 | diagnostic |
| memory-only vec2 rows | 4.887 | 4.876 | diagnostic |
| contiguous sequential read | 4.835 | 4.837 | diagnostic |

`hm_row64` was slightly faster (6.113 ms at chunk 256) but reassociates the
stream reduction (`maxabs=3.34e-06`) and was rejected. `hm_row256` regressed
to 7.9 ms. `hm_dp4a1` is only a diagnostic ceiling: it removes the low-bit
DP4A correction, is not quality-equivalent (`maxabs=0.057`), and at 6.312 ms
shows the low-bit DP4A work is not the bottleneck.

Raw logs: `layout128.log`, `layout256.log`, `layout256b.log`, `layout512.log`.
