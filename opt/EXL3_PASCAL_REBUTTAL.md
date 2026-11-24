# Resposta ao EXL3_PASCAL_CEILING.md: o teto real fica entre 22 e 34 TPS

O agente conclui que 30 TPS em decode serial é "inalcançável" com o checkpoint
REXL3 2.26bpw num GTX 1060. A conclusão está errada. A medição no próprio
hardware põe o piso serial em **29–46 ms/tok (22–34 TPS)**, mesmo checkpoint,
sem quant nova. Os 127 ms de hoje estão **3–4× acima** desse piso. Isso vem de
kernel subocupado, de launches e de falta de fusão, não de limite físico.

O "beco sem saída do DP4A" saiu de um kernel com bug de sinal. O ledger de
2026-11-23 descartou o dp4a por causa dele.

Tudo abaixo foi medido no GTX 1060 6GB desta máquina, com os apps de desktop
ligados (2026-11-23).

## 1. Teto de ALU do decode: medido

Probe: `opt/kbench/probe_ceiling.cu`. A GPU fica cheia (640 blocos × 256), as
palavras do trellis ficam em registrador e a geometria é a do `tile_row`
(4 pesos por funnel shift). O decode não tem desvio que dependa do valor do
peso, então o custo por peso é o mesmo com pesos reais.

| formulação | G pesos/s |
|---|---|
| V0 fp32 (`tile_row` de hoje: IDP + I2F + FFMA) | 251 |
| V1 `tile_row_d4` (splat via PRMT por peso) | 316 |
| V2 dp4a enxuto (x já em splat no smem) | 347 |
| V3 enxuto + passe de resíduo (x com ~15-16 bits) | 303 |

Por token passam ~5,6 G pesos trellis: 3,05 G de MoE (medido no ledger),
~2,1 G densos (estimado de 634 MiB a ~2,5 bpw) e 0,46 G de unembed. As GEMVs
custam no mínimo **16–22 ms/tok**. Hoje custam ~79 ms (45 de MoE + 34 densos).

## 2. Os claims contra os critérios que o próprio agente escreveu

### Claim 1: "instruction-bound, não bandwidth-bound". Refutado pelo critério dele.
O critério dele: "a kernel variant that exceeds ~15GB/s effective on the SAME
sm_61 hardware". O `gemv_v2_bench`, kernel do próprio agente, já passa disso.
A GEMV densa 2304×1152 com ka=3, ntok=1 e nky=5 roda em **30,7 µs**, ou seja
995 KB / 30,7 µs = **32 GB/s efetivos**, 2× o critério.

Em pesos, isso dá 86 G pesos/s, **3× abaixo** do teto de 251 da mesma
formulação. Com m=1 o kernel não esbarra na contagem de instruções. Ele esbarra
em latência e ocupação: 18–90 blocos para 10 SMs, cada lane presa a uma cadeia
dependente LDS→SHF→XMAD×3→IDP→I2F→FFMA.

O experimento de residência total em VRAM prova só que o gargalo não é PCIe.
Ele não prova que o kernel está no limite de issue.

Os "~17 instr/peso" vieram do `probe_decode.cu`. Ali cada peso faz 2 LDS e
calcula índice em tempo de execução para ler de um anel em smem. Esse número
mede o probe, não o piso.

### Claim 2: "~5 instruções por peso são estruturais". Correto.
Testei 5 formulações em PTX (pedaços de 16 bits, constantes separadas, `mad`
com constante `<<16`, `mov.b32 {0,h}`). O ptxas sempre emite 3 XMAD para
`w16 * 0x83DCD12D`. Duas XMAD (H0 × lo e depois `.PSL` com hi) só saem com SASS
escrito à mão. O piso fica em SHR + 3 XMAD + IDP, mais LDS amortizado. Essa é a
única parte do doc que se sustenta.

### Claim 3: "não existe dot inteiro legal para o MAC". Falso.
A premissa "os pesos precisariam virar int8" confunde peso com ativação. Os
plane-bytes de `p = w16*M` são 4 bytes **sem sinal** e o peso é a soma deles.
Assim, `dp4a.s32.u32(splat(x8), p)` devolve exatamente `x8 * bytesum(p)`. O peso
continua exato, só a ativação vira int8, e com o passe de resíduo ela fica com
~15-16 bits. É o caminho `EXL3_INT8_GEMV` do exllamav3 stock.

Verificado no device: **16.711.680 pares (w16, x8), 0 erros** contra a
referência inteira.

Isso troca IDP + I2F + FFMA (3 instruções) por 1 IDP no MAC, que é 3× a taxa
do caminho fp32, acima dos "≥2x" que o critério dele pede. O ganho no teto vai
de +26% (V1) a +38% (V2).

O `__dp2a_lo` não ser um dot int16×int16 é verdade e não importa. Ninguém
precisa de lanes int16.

### Claim 4: "30 TPS serial é inalcançável". A conta está errada.
O doc supõe no máximo "2x em todo gemv". O teto medido dá **3,6–4,9×** sobre os
79 ms de GEMV de hoje. O próprio ledger do agente diz "MoE-VRAM ≈ 33 ms
(hoje 113)", que já é 3,4×, e o Claim 4 contradiz esse número.

Os outros termos do orçamento não são limite físico:
- had/swarm ~20 ms: são launches pequenos sem fusão. O Hadamard cabe no
  prólogo e no epílogo da GEMV.
- host ~13 ms: um passo inteiro dentro de um CUDA graph leva isso para ~1–2 ms.
  O path GGUF já fez isso neste projeto.

Piso honesto com o checkpoint idêntico:

| termo | piso | base |
|---|---|---|
| GEMVs trellis | 16–22 ms | probe_ceiling |
| PCIe dos experts das 12 camadas fora do loop | 8–15 ms | 8,6–12 GB/s medidos; prefetch esconde uma parte |
| atenção, Hadamard, router, ctrl | 4–7 ms | atenção fundida = 2,3 ms medidos; o resto é estimativa |
| launches com CUDA graph | 1–2 ms | estimativa |
| **total** | **29–46 ms → 22–34 TPS** | |

Os <70 ms do critério dele ainda não foram demonstrados de ponta a ponta. O que
a medição mostra é que nada no hardware impede chegar lá: o piso fica na metade
desse critério.

### Claim 5: "um EXL3 com DP4A não resolve neste tamanho". Meia verdade.
É correto que o dp4a não dá 2×, porque o custo está no decode, não no MAC. O
argumento de "pesos int8-codable a ~4 bpw" não se aplica: o caminho com splat
não muda o formato dos pesos.

## 3. Bug que gerou o "dead-end"

`tile_row_d4` (`crates/inkwell-cuda/src/exl3_gemv_v2.cuh:181`) chama
`__dp4a((int)splat, (int)p0, isum)`, que é **signed×signed** (IDP.4A.S8.S8). Os
plane-bytes não têm sinal, então qualquer byte ≥128 entra negativo. Medido:
**15.604.998 de 16.711.680 produtos errados (93%)**. Esse é o bug de corretude
do `INKWELL_MOE_D4A`.

A entrada "DP4A/DP2A dead-end" do ledger e o "d2a REVERTIDO — saída garbage"
tiraram uma conclusão de arquitetura de um erro de sinal.

Correção:
```cuda
int d; asm("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(x_splat), "r"(p), "r"(acc));
```
E grave x já em splat (`uint4` com 4 splats) no `inkwell_exl3_quant_x_i8p`. Isso
elimina o PRMT que hoje roda por peso e por consumidor: é a diferença
entre V1 e V2.

## 4. O que fazer, por ordem de ms ganhos

1. Corrigir o sinal do d4 e pré-splatar x. Validar com um gold test (padrão
   `GOLD_Q80`) e depois ligar no serial.
2. Ocupação das GEMVs m=1: juntar os 8 experts e q/k/v/gate/up em um launch
   cada, e dar a cada thread 2–4 tiles independentes intercalados para ter ILP.
   A meta é sair de 86 G pesos/s para mais de 250.
3. Fundir o Hadamard (`had_in`/`had_out`) na GEMV. Ataca os ~20 ms de
   had/swarm.
4. Colocar o `step_block` inteiro num CUDA graph. Ataca os ~13 ms de host.
5. Fazer prefetch dos experts previstos para a camada L+1 enquanto a L roda, e
   usar a VRAM que sobra como cache LFU das camadas fora do loop. Ataca os
   8–15 ms de PCIe.
6. Opcional: escrever à mão em SASS o `w16*M` com 2 XMAD (+~10–15% no teto).

Medir cada passo com os apps ligados e registrar no ledger. O próximo veredito
sobre o teto deve citar G pesos/s medidos contra 251/347, não instruções
contadas num probe com LDS por peso.
