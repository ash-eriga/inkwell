# Inkling-2 forward pass — op-level recipe (extracted from upstream)

Sources: `~/lfm-gtx1060-bench/src-inkling/src/models/inkling2.cpp`,
`inkling2-loop.cpp`, `llama-graph.cpp` (build_moe_ffn, build_attn).

## Driver (inkling2_loop::graph)
- `cur = build_inputs()` → embed get_rows + tok_norm rms_norm; `tau` tensor:
  `tau = reshape3d(scale_bias(log(clamp(scale(cast(pos,F32),1/floor),1,inf)), alpha, 1), 1,1,n_tokens)`
- blocks = []; layers 0..core_start: `cur = build_layer(il, il, cur, blocks)`
- prelude = cur; prelude_blocks = blocks
- for pass in 0..n_passes:
  - if pass>0: bank = blocks[prelude_blocks.len+1..]; bank_out = cur; cur = prelude; blocks = prelude_blocks
  - for ip in core_start..core_end:
    - il = pass==0 ? ip : n_phys + (pass-1)*(end-start) + (ip-start)  ← KV cache slot
    - pass==0 → build_layer(il,ip,cur,blocks)
    - else → messages = mix(bank, bank_out, loop.msg_query); build_layer(il,ip,cur,blocks,messages,loop)
- layers core_end..n_phys: build_layer(il, il); then build_output(cur, blocks)

## build_layer(il, ip, cur, blocks, messages?, loop?)
- inputs[3] = {cur,cur,cur}
- if attn_res_query: mixed = mix(blocks, cur, attn_res_query); inputs[j] = row j (is_swa → only row 0 used for all 3? no: SWA uses inputs[0] shared; conv-gated uses 3 rows)
- if messages: inputs[j] = control(inputs[j], message(j), loop, j) for j in 0..3 (Q,K,V)
- norms: is_swa&&messages → all 3 get attn_norm; is_swa → inputs[0] only, K,V share; else 3 inp_norms
- attn:
  - Qcur = clip(reshape3d(mm(wq,in0)), attn_q_clip, qk_clip)
  - Kcur = clip(reshape3d(convolve(mm(wk,in1), k_conv, conv? mm(k_conv_proj,in1)), head,hkv), None, qk_clip)
  - Vcur = rms_norm(reshape3d(convolve(mm(wv,in2), v_conv, conv? mm(v_conv_proj,in2)), head,hkv), eps)
  - Rcur = clip(reshape3d(mm(attn_rel,in0), n_rel,h), attn_rel_clip, rel_clip)
  - !is_swa → Qcur*=tau; Rcur*=tau
  - sinks = mm(attn_sink_proj, in0)
  - table = mm(pad(attn_rel_bias,0,1,0,0), Rcur); table -= reshape3d(sinks,1,n_head,n_tok); table = cont(permute(table,1,0,2,3))
  - sinks2 = fill(f32[n_head], 0)
  - cur = build_attn(Qcur,Kcur,Vcur,mask=table,sinks=sinks2,scale=1/sqrt(head_dim))
  - cur *= sigmoid(mm(wqkv_gate, in0))
  - cur = convolve(mm(wo,cur), attn_post_conv, None, offset=2*n_embd_k_gqa)
- residual: if ip % block_size == 0 → blocks.push(inpSA) else cur += inpSA
- ffn: cur = reshape2d(mix(blocks, cur, ffn_res_query)); messages → control(cur, msg(FFN), loop, FFN); cur = ffn_norm(cur)
- cur = build_moe_ffn(SIGMOID gating, exp_probs_b, gate_up_exps merged, down_exps, norm_w=true, w_scale=expert_weights_scale)
- cur = convolve(cur, ffn_post_conv, None, offset=2*n_embd_k_gqa + n_embd)
- cur += ffn_inp

## Helpers
- mix(blocks, cur, query): states = concat([reshape3d(b) for b in blocks] + [reshape3d(cur)], dim1);
  states = rms_norm(states, 1e-6); queries = repeat(query, [n_embd, ne1, n_tok]);
  scores = softmax(mm(states, queries), scale=1/sqrt(n_embd)); out = mm(cont(transpose(states)), scores)
- control(cur, msg, loop, ctrl): gate = mm(ctrl_down[ctrl], concat([rms(cur),rms(msg)],0));
  gain = tanh(mm(ctrl_gain[ctrl], gate)); forget = tanh(mm(ctrl_forget[ctrl], gate));
  out = cur - forget*cur + gain*msg
- clip(cur, scale?, thr): cur *= scale?; norm = sqrt(sum_rows(sqr(cur))) + 1e-6;
  cur *= clamp(thr/norm, 0, 1)
- convolve(cur, weight, dynamic?, history, offset):
  x = transpose(reshape3d(cur, ch, n_seq_tok, n_seqs)); x = concat(past(history offset), x, 0)
  store conv state (last n_conv-1 cols back to state at offset)
  out = ssm_conv(x, weight) → reshape2d
  dynamic taps: for lag in 0..n_conv: out += window(x, n_conv-1-lag) * tap(dynamic, lag) per head
  return scale(cur + out, 1/sqrt((dyn?2:1)*n_conv + 1))
- build_moe_ffn (sigmoid): logits = mm(gate_inp, cur); probs = sigmoid(logits);
  sel = probs + exp_probs_b; top_k = argsort_top_k(sel, n_used); weights = get_rows(probs, top_k);
  weights /= clamp(sum_rows(weights), 6.1e-5, inf); weights *= w_scale; ids = top_k
  fused gate_up: gate_up = mul_mat_id(gate_up_exps, cur, ids); split → cur=gate[..n_ff], up=[n_ff..];
  SITU: act = beta*tanh(gate/beta) * sigmoid(gate); up = lb*tanh(up/lb); cur = act*up
  down = mul_mat_id(down_exps, cur, ids); cur = sum_rows(down * reshape3d(weights,1,n_used,n_tok))
- build_attn: per upstream llm_graph_context — K/V write into KV cache views at slot il,
  scores = mm(K,Q)·scale + mask(table) → soft_max_ext(mask) → out = mm(V attn) ;
  sinks via add_sinks. For our runner: keep K,V per-visit in DevBuf arenas, build the same ops.
- build_output: output_res_query mix → output_norm rms → mm(output) → tanh softcap 30 → view unpadded vocab
- conv state: per visit slot, history [n_conv-1, n_sconv_channels]; offsets: K=0, V=n_embd_k_gqa, attn_post=2*gqa, ffn_post=2*gqa+n_embd

## Shim ops still needed
- ggml_mul_mat_id (expert gather-GEMV), ggml_argsort_top_k, ggml_set_rows (for masked select? optional),
  ggml_softplus (unused for sigmoid path), ggml_sigmoid/tanh/sqr/sqrt/log/clamp/pad/permute/cont/concat — DONE
- attn path: manual scores/softmax/kqv OR flash_attn_ext with mask=table+sinks — DONE in shim
- Memory: recurrent conv state + KV arenas as DevBuf-backed tensors via ink_tensor_assign

## End-to-end runner plan (inkwell-run)
1. Load GgufFile → ModelWeights (done)
2. ink_ctx + cuda backend + weight buf (device for dense, host for experts later)
3. For each tensor: create ggml tensor in ctx, assign into buf at offset, ink_tensor_set from mmap
4. build_forward(pos, token_ids): Rust port of the recipe above
5. KV arenas from inkwell-kv plan; conv states per visit
6. Logits → inkwell-sampler → token
