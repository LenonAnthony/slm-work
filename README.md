# slm-work

Trabalho de disciplina sobre fine-tuning de modelos de linguagem pequenos. Quatro modelos recebem a mesma receita QLoRA e são treinados no dataset C2 do PictoInstruct, que pede cinco cartões de Comunicação Aumentativa e Alternativa (CAA) em português para um conceito. A avaliação repete as métricas usadas com o Gemma 4 E2B no PictoInstruct, comparando cada modelo base com sua versão adaptada.

## Dados

O C2 é consolidado a partir dos candidatos gerados no PictoInstruct (`c2_c3_v2/generation/c2_full`), usando a mesma partição por componentes do dataset offline silver. Para cada conceito fica a primeira tentativa sem falha de métrica, de segurança, de duplicata ou de parse. A partição não separa todos os conceitos relacionados, e 732 dos 2.199 conceitos de teste (33%) compartilham um synset com algum conceito de otimização, como “dormir” no teste e “dormir bem” no treino.

| Partição | Registros |
| --- | ---: |
| otimização | 6.295 |
| desenvolvimento | 2.203 |
| teste | 2.199 |

Quatro conceitos ficaram em quarentena porque o gerador falhou nas duas tentativas. Os cartões foram gerados pelo `deepseek-v4-flash` com o prompt V2. Por isso o C2 é um dado silver, sem validação profissional.

## Modelos

| Chave | Modelo | Revisão |
| --- | --- | --- |
| `s5-nano` | `maritaca-ai/S5-nano` | `4aef892` |
| `gemma4-e2b` | `google/gemma-4-E2B-it` | `3e22461` |
| `minicpm-v4.7-1b` | `openbmb/MiniCPM-V-4.7-1B` | `ad9bf37` |
| `nemotron-bolt-e4b` | `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16` | `49843ee` |

## Receita

A receita é a mesma do Gemma 4 E2B no PictoInstruct. A quantização é NF4 em 4 bits com dupla quantização e computação em bf16. O LoRA usa r=16, α=32 e dropout de 0,05 nas projeções de atenção e de MLP do decodificador de texto. A lista de módulos muda com a arquitetura (`slm/models.py`), e por isso o adaptador tem de 10,8 M a 38,9 M de parâmetros. A receita foi definida para o Gemma e não foi ajustada para os outros modelos. O otimizador é o `paged_adamw_8bit`, com lr de 1e-4, warmup de 3% e agenda cosseno. O lote efetivo é de 8 exemplos e o treino dura uma época, com perda só sobre a resposta, seed 3407 e escolha do melhor checkpoint pela perda de validação. A geração é gulosa, com até 512 tokens novos e o modo de raciocínio desligado.

Há uma diferença em relação ao PictoInstruct. A preparação k-bit do `peft` converte as tabelas de embedding para fp32, e as tabelas por camada do Gemma 4 não cabem assim em 12 GB. Aqui as tabelas ficam em bf16. Como elas estão congeladas, os valores não mudam.

## Métricas

As métricas são as mesmas do PictoInstruct:

- validade estrutural, com teste exato de McNemar;
- NLL e perplexidade da referência, ponderadas por token;
- similaridade com o conceito, não redundância e Vendi score dos cinco cartões;
- alinhamento húngaro com a referência e sobreposição de emoji, calculados com os encoders Serafim e LaBSE;
- latência.

Os intervalos de confiança de 95% vêm de um bootstrap pareado e de um bootstrap por blocos de synset, ambos com 10.000 iterações. As métricas de embedding são diagnósticos offline, não julgamento profissional de CAA.

## Como rodar

```bash
uv sync
uv run python -m scripts.build_c2_dataset      # lê o PictoInstruct e grava data/c2
mkdir -p outputs/logs
setsid nohup scripts/run_all.sh > outputs/logs/run_all.log 2>&1 &
```

O build lê o PictoInstruct pelo caminho fixado em `scripts/build_c2_dataset.py`, e o Gemma exige aceitar a licença no Hugging Face. O dataset C2 não está no repositório. O `data/c2/manifest.json` guarda os hashes dos arquivos gerados.

O `run_all.sh` treina e avalia os quatro modelos em sequência e pode ser retomado depois de uma interrupção. Ao fim ele gera `outputs/<modelo>/comparison.json` e a tabela em `outputs/report/`. Esses arquivos e o `run_manifest.json` de cada treino estão versionados. Adaptadores e saídas por conceito ficam fora do repositório.

## Resultados

Teste com 2.199 conceitos, do modelo base para o adaptado. As médias semânticas usam as saídas válidas de cada versão, que no modelo base são poucas. Por isso a comparação entre as duas versões deve ser lida nas diferenças pareadas, e não na tabela. O MiniCPM base teve uma única saída válida e fica sem média.

| Métrica | S5-nano | Gemma 4 E2B | MiniCPM-V 4.7 1B | Nemotron 3.5 Bolt E4B |
| --- | ---: | ---: | ---: | ---: |
| Validade estrutural | 36,6% → 99,7% | 78,3% → 99,7% | 0,05% → 99,2% | 13,4% → 99,7% |
| Perplexidade | 4,50 → 1,99 | 11,37 → 2,21 | 19,15 → 4,02 | 9,08 → 2,00 |
| Similaridade com o conceito | 0,471 → 0,537 | 0,500 → 0,544 | — → 0,547 | 0,558 → 0,540 |
| Não redundância | 0,265 → 0,238 | 0,188 → 0,211 | — → 0,099 | 0,227 → 0,237 |
| Vendi @5 | 2,92 → 2,73 | 2,48 → 2,61 | — → 2,11 | 2,68 → 2,73 |
| Alinhamento com a referência | 0,524 → 0,644 | 0,520 → 0,629 | — → 0,571 | 0,564 → 0,646 |
| Sobreposição de emoji | 0,268 → 0,344 | 0,272 → 0,341 | — → 0,228 | 0,252 → 0,364 |
| Latência (s) | 1,76 → 4,44 | 1,80 → 4,82 | 0,84 → 3,37 | 2,26 → 6,44 |

O teste de McNemar dá p < 10⁻¹⁰⁰ na validade dos quatro modelos, e a diferença de NLL tem IC 95% longe de zero em todos. Nos conceitos válidos nas duas versões, o ganho pareado de alinhamento é de 0,119 no S5-nano (n = 802), 0,109 no Gemma (n = 1.715) e 0,107 no Nemotron (n = 292). Nesses mesmos conceitos a diversidade cai no S5-nano (Vendi −0,207) e no Nemotron (−0,070) e sobe no Gemma (+0,131). No Nemotron o sinal depende do encoder, com −0,144 no Serafim e +0,003 no LaBSE. A perplexidade só é comparável entre o Gemma e o Nemotron, que usam o mesmo tokenizador. A latência sobe porque os modelos adaptados geram respostas mais longas e porque o adaptador não foi fundido aos pesos. O tempo por token passa de 22,8 para 40,1 ms no S5-nano, de 24,6 para 45,5 ms no Gemma, de 15,5 para 29,6 ms no MiniCPM e de 33,5 para 60,8 ms no Nemotron.

Entre os modelos adaptados, o S5-nano e o Nemotron empatam em alinhamento, similaridade, não redundância e Vendi (alinhamento −0,002, IC de −0,005 a 0,002), e o Nemotron fica à frente na sobreposição de emoji (−0,020, IC de −0,032 a −0,008). Os dois superam o Gemma no alinhamento por 0,015 e 0,017 e no Vendi por cerca de 0,12, e o MiniCPM fica atrás dos três em alinhamento e diversidade. O bootstrap por blocos de synset leva às mesmas conclusões. Os modelos adaptados também repetem o texto curto de algum cartão bem mais que a referência, que faz isso em 3,7% dos conceitos. A taxa é de 21,9% no S5-nano, 25,9% no Nemotron, 30,0% no Gemma e 64,2% no MiniCPM. Os números completos estão em `outputs/report/summary.json` e `outputs/report/between_models.json`.

## Limitações

O parser segue o do PictoInstruct e separa cada linha nas duas últimas vírgulas. Quando a frase falada contém vírgula, o texto do cartão absorve parte dela. Isso acontece em 28% dos cartões de referência do teste, e 66% dos conjuntos têm ao menos um cartão assim. A frequência também muda entre os modelos. Ela vai de 1% a 35% dos cartões nas versões base do S5-nano, do Gemma e do Nemotron, e de 23% a 29% nas quatro versões adaptadas. A repetição de texto curto mantém a mesma ordem entre os modelos quando se usa só o texto antes da primeira vírgula. Os cartões de referência foram gerados por um LLM, então as métricas medem proximidade a uma referência silver e não qualidade clínica. Os modelos base rodaram sem exemplos no prompt, então boa parte do ganho de validade é aprendizado do formato. Cada modelo foi treinado uma única vez, então os intervalos não cobrem a variação entre seeds, e as diferenças entre modelos também refletem tamanho e arquitetura, já que a receita é a mesma para todos. A latência foi medida com o adaptador LoRA carregado sem fusão, e o S5-nano e o MiniCPM rodaram com a implementação de referência do `causal_conv1d`, que é mais lenta.
