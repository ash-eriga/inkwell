# REXL3 port — wiring map (estado 2026-11-22)

## Estado
- Modelo: `~/models/inkling/rexl3-00001.safetensors` (7.8GB) + `rexl3-00002.safetensors` (555MB) baixados.
- `index.json` (weight_map name→shard), `config.json`, headers em `opt/rexl3/*.header.json`.
- Kernels EXL3 escritos no fim de `crates/inkwell-cuda/src/kernels.cu` (~linha 1560+):
  `exl3_window` (extrator do ring, prefixo INCLUSIVO!), `exl3_mul1`, `inkwell_exl3_had_in`,
  `inkwell_exl3_gemv`, `inkwell_exl3_had_out`, launcher `ink_launch_exl3_gemv(trel,suh,svh,x,y,k,n,ka,halfk,scratch,stream)` + debug `ink_launch_exl3_dbg`, `ink_launch_exl3_dbg_had`.
- GOLD test: `crates/inkwell-cuda/examples/gold_exl3.rs` — PASSA (maxdiff ~5e-4 fp16) para K∈{1.5,2,2.5,3.5,4}. CPU ref = port de trellis.py.
- Crate nova `inkwell-st` (mmap safetensors, HashMap de tensores). Byte math: GGUF ~2491 MiB/tok → REXL3 ~1650 (−34%).

## Formato trellis (verificado contra GPU)
- trellis I16 [TK, TN, 16K]: K=shape[2]/16; k=TK·16 (in), n=TN·16 (out).
- Tile = ring de R=256K bits; position i window = bits [S(i)-16, S(i)) MSB-first, S INCLUSIVO: int K→S=(i+1)K; half K→S=(i+1)·KA+((i+1)>>1).
- Packed: u32 words, bit 0 = MSB word 0; stored como u16 pares (u16[2j]=low half word j).
- Extrator kernel validado: b0=S-16(+R se <16); i0=b0>>5; s0=48-(b0&31); v=(ring[i0]<<32|ring[(i0+1)%nw])>>s0 &0xffff.
- mul1: w*=0x83DCD12D; bytesum+0x6400 → fp16 bits; v=fp16(h·0x1eee+0xc931). (0x6400+bsum bits = valor 1024+bsum.)
- tile_perm i=8t+j → r=(t%4)*2+(j&1)+(8 se j&2), c=t/4+(8 se j&4).
- W = diag(suh)·H·Ŵ·H·diag(svh), H=Hadamard/√128; GEMV: xh=H(suh⊙x); t=Ŵᵀxh; y=svh⊙Ht.
- Butterfly correto: `u = (t&s) ? sm[t-s]-sm[t] : sm[t]+sm[t+s]` (ordem da subtração importa!).

## Mapa de nomes HF → GGUF (mesmos bytes; HF [out,in] row-major = GGUF [in,out] col-major)
attn: wq_du→attn_q [2304→3072], gate_attn→attn_gate [2304→3072], wk_dv→attn_k [2304→1024],
wv_dv→attn_v [2304→1024], wr_du→attn_rel [2304→384], wo_ud→attn_output [3072→2304].
Dense BF16: attn.attn_sink_proj.weight[24,2304]→attn_sink_proj; attn.rel_logits_proj.proj[16,1024]→attn_rel_bias;
attn.q_l2_clip.scale[24,128]→attn_q_clip; attn.rel_l2_clip.scale[24,16]→attn_rel_clip;
attn.k_sconv.weight[1024,1,4]→attn_k_conv; attn.v_sconv.weight→attn_v_conv; attn_sconv.weight[2304,1,4]→attn_post_conv;
mlp_sconv.weight→ffn_post_conv; attn_norm.weight→attn_norm; mlp_norm.weight→ffn_norm;
mlp.gate.weight[128,2304]→ffn_gate_inp; mlp.gate.bias F32[128]→exp_probs_b;
attn_res.attn_queries.N→attn_res_query; attn_res.mlp_queries.N→ffn_res_query; attn_res.output_query→output_res_query.
Conv layers (l%6==5): attn.k_sconv.static_w[4,1024]→attn_k_conv; k_sconv.weight_proj[32,2304]→attn_k_conv_proj;
v_sconv análogo; q_input_norm→attn_q_inp_norm; k_input_norm→attn_k_inp_norm; v_input_norm→attn_v_inp_norm.
Controllers (loop layers 6..17): controllers.{attn,k,mlp,v}.{down[64,4608],forget[2304,64],gain[2304,64]}_weight
→ loop_{q,k,v,ffn}_ctrl_{down,forget,gain} (attn→q, mlp→ffn).
middle_cycle_cross_pass_queries [24,2304] F32 → loop_msg_query (por visita, row index).
Globals: embed.weight BF16[201024,2304]→token_embd (host-only, row gather bf16→f32);
embed_norm.weight→token_embd_norm; norm.weight→output_norm; unembed.{trellis[144,12568,64],suh,svh}→output (K=4, n=201088 padded).
Experts: mlp.experts.E.{gate_proj,up_proj}[2304→1152],down_proj[1152→2304] — K por-expert varia (SAGE). Sem .scale → has_sgu=has_sdn=false.
Ignorar: model.visual.*, markov_head.*, mtp.safetensors (spec decode depois).

## Wiring pendente (em ordem)
1. inkwell-st: abrir repo multi-shard via index.json (name→file).
2. InklingConfig::from_hf(config.json) — campos estão em text_config/loop_config.
3. Runner::load_st: classificar tensors; dense BF16→upload raw bytes ty=30 (ggml), F32 ty=0;
   exl3 → dev buffer (trel+suh+svh separados) + desc struct; experts→pinned + desc tables [128]{trel,suh,svh,ka|halfk};
   embed host-only; attn_res queries F32.
4. mmq call sites em layer(): wq/gate/rel/wk/wv/wo/output → t::exl3_mm custom op (userdata=Box<Exl3W>, srcs=[x]);
   concat trick q+gate+rel: v1 três ops separados.
5. moe_cb_exl3: novos kernels com desc arrays — had_in_moe/gemv_moe/had_out_moe grid=(tiles, slots)
   (tn fixo: gate/up 72, down 144; tk 144/72); situ_pair + expert_wsum reutilizam.
6. embd row: bf16→f32 em host (não ink_dequant).
7. E2E run → golden text → bench.sh → ledger.
