# Challenge 3b — whole decode loop

- `SOLUTION_3b.md`: floor ≥ 1.2308 c/w, construction 1.2387 c/w (−14.18% vs shipped 1.4434), bit-exact.
- `candidate_3b.c`: CPU model of the construction (challenge3 `cand_*` interface).

Verify (challenge3 ref harness lives on branch `t3/exl3-decode-instr-bound-problem`):

    d=$(mktemp -d); for f in Makefile exl3_ref.h exl3_ref.c harness.c candidate_ch2.c; do
      git show t3/exl3-decode-instr-bound-problem:opt/challenge3/ref/$f > $d/$f; done
    cp opt/challenge3b/candidate_3b.c $d/ && make -C $d -s CAND=candidate_3b.c test
    # expect: ALL TESTS PASSED (BIT-EXACT)
