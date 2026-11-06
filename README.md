Detecção de fraude em cartão de crédito

Projeto de classificação em uma base **extremamente desbalanceada**: só 0,17% das transações são fraude. O notebook carrega os dados, prepara as variáveis, compara modelos com foco no **recall da fraude**, ajusta o limiar de decisão e usa **SHAP** para explicar o que levou uma transação a ser marcada como fraude.

- Notebook: [`deteccao_fraude_cartao.ipynb`](deteccao_fraude_cartao.ipynb) (saídas salvas: tabelas e curvas)
- Tabela de resultados: [`resultados/comparacao_modelos.csv`](resultados/comparacao_modelos.csv)
- Figuras: pasta [`imagens/`](imagens/)

## O problema

A base tem 284.807 transações de cartão de crédito de clientes europeus (dois dias de 2013). As colunas `V1` a `V28` são componentes de uma PCA, aplicada para proteger a privacidade; `Time`, `Amount` e `Class` (1 = fraude) são as únicas colunas originais.

Depois de remover 1.081 linhas duplicadas, restam 283.726 transações, 473 delas fraudes (0,167%): cerca de **599 transações normais para cada fraude**.

**Por que o desbalanceamento muda a avaliação.** Um modelo que responde "não é fraude" para tudo acerta **99,83%** e detecta **zero** fraudes. A acurácia, portanto, não diz nada. A avaliação usa as métricas da classe de fraude:

- **recall**: das fraudes que existem, quantas o modelo pega (métrica principal);
- **precisão**: dos alertas emitidos, quantos são de fato fraude (evita "ganhar" recall marcando tudo);
- **F1 e F2**: resumem as duas; o F2 pesa mais o recall;
- **PR-AUC** (curva de precisão × recall) no lugar da ROC-AUC como medida geral, porque a ROC fica otimista quando há tantos exemplos normais.

![Distribuição das classes](imagens/01_distribuicao_classes.png)

## O que foi feito na preparação dos dados

- **Duplicadas:** 1.081 linhas idênticas removidas antes da separação, para não aparecerem ao mesmo tempo em treino e teste.
- **Variáveis novas:** `log_valor = log(1 + Amount)` (o valor tem cauda muito longa) e `hora = (Time / 3600) mod 24`. As colunas `Time` e `Amount` originais saem.
- **Padronização:** `StandardScaler` dentro de `Pipeline` para a regressão logística, ajustado só com o treino (sem vazamento). Modelos de árvore não dependem de escala e usam as variáveis originais, o que também deixa o SHAP legível.
- **Separação com `stratify`:** 80% treino / 20% teste. Do treino, 25% vai para **validação**, onde o limiar é escolhido. O teste só é usado no final.

| conjunto | transações | fraudes | % de fraude |
|---|---:|---:|---:|
| treino (fit) | 170.235 | 284 | 0,167 |
| validação | 56.745 | 94 | 0,166 |
| teste | 56.746 | 95 | 0,167 |

- **Desbalanceamento:** peso de classe nos modelos (`class_weight` e `scale_pos_weight`) como abordagem principal; **undersampling** (`RandomUnderSampler`) e **oversampling** (`SMOTE`), ambos levando as fraudes a 10% dos normais, como alternativas, aplicados só ao treino via `imblearn.Pipeline`.

## Comparação entre os modelos

Resultados no **conjunto de teste** (56.746 transações, 95 fraudes), classe de fraude, com o **limiar ajustado na validação** (máximo F2):

| modelo | limiar | recall | precisão | F1 | PR-AUC | fraudes detectadas | alertas falsos |
|---|---:|---:|---:|---:|---:|---:|---:|
| Regressão Logística (peso de classe) | 0,9998 | 0,768 | 0,777 | 0,773 | 0,680 | 73 | 21 |
| Random Forest (peso de classe) | 0,459 | 0,737 | 0,897 | 0,809 | 0,803 | 70 | 8 |
| XGBoost (peso de classe) | 0,479 | 0,789 | 0,926 | 0,852 | 0,821 | 75 | 6 |
| Regressão Logística + Undersampling | 0,951 | 0,737 | 0,614 | 0,670 | 0,583 | 70 | 44 |
| Regressão Logística + SMOTE | 0,959 | 0,768 | 0,802 | 0,785 | 0,691 | 73 | 18 |
| XGBoost + Undersampling | 0,990 | 0,716 | 0,944 | 0,814 | 0,812 | 68 | 4 |
| **XGBoost ajustado (GridSearchCV)** | **0,320** | **0,779** | **0,925** | **0,846** | **0,820** | **74** | **6** |

O modelo escolhido é o **XGBoost ajustado**, por ter o maior F2 na **validação** (a seleção não usa o teste). Ele detecta **74 das 95 fraudes** e gera 6 alertas falsos entre 56.651 transações normais.

![Curvas ROC e de precisão × recall](imagens/03_curvas_roc_pr.png)

Leituras principais:

- **Árvores superam a regressão logística** (PR-AUC 0,80 a 0,82 contra 0,68 a 0,69).
- **Undersampling e SMOTE não superaram o peso de classe.** Com limiar 0,5 eles aumentam o recall (até 0,83), mas com 99 a 171 alertas falsos.
- **Cuidado com diferenças pequenas:** com 95 fraudes no teste, uma fraude muda o recall em cerca de 1 ponto. O XGBoost sem ajuste de hiperparâmetros fica a uma fraude do escolhido (75 contra 74) e tem F2 de teste ligeiramente maior. Na prática, os dois XGBoost empatam.

## Limiar de decisão e o que o SHAP mostrou

**Limiar.** O modelo devolve uma probabilidade; o limiar a transforma em "fraude / não fraude". Foi escolhido na validação, maximizando o **F2** (recall com peso maior), porque deixar passar uma fraude custa mais do que investigar um alerta. O limiar escolhido é **0,32**.

O efeito do limiar depende do modelo: na regressão logística, o limiar padrão 0,5 gera 1.596 alertas falsos (precisão 0,049), e o ajuste leva a precisão a 0,777. No XGBoost, mover de 0,5 para 0,32 manteve as 74 fraudes detectadas e acrescentou 2 alertas falsos: os escores das árvores já são bem separados.

![Limiar de decisão](imagens/04_limiar.png)

**Relatório de classificação do modelo escolhido (teste):**

```
              precision    recall  f1-score   support

      normal     0.9996    0.9999    0.9998     56651
      fraude     0.9250    0.7789    0.8457        95
```

**SHAP.** As variáveis que mais pesam são **V14, V4, V12 e V10**: valores baixos de V14, V12 e V10 e valores altos de V4 empurram a transação para fraude. Como são componentes de PCA, elas não têm nome de negócio. A hora do dia está entre as 10 mais importantes; o valor pesa pouco no geral.

![SHAP: resumo](imagens/07_shap_resumo.png)

Três casos individuais do teste (gráficos em `imagens/`):

- **Fraude detectada:** V14 (−13,7) somou +4,8 log-odds sozinha, seguida de V12, V4 e V10.
- **Fraude não detectada:** V4, V14 e V11 tinham valores comuns de transações normais. É o limite do modelo: fraudes parecidas com transações legítimas nas componentes disponíveis não são separadas.
- **Alerta falso:** V14 (−7,1), V10 e V12 apontaram para fraude em uma transação normal.

![SHAP: fraude detectada](imagens/08_shap_fraude_detectada.png)

## O que mudei em relação ao que a Expert fez

> Preencha esta seção comparando com o vídeo da Expert: não tenho acesso ao que ela fez, então abaixo estão apenas as decisões **minhas** neste notebook, para você confirmar o que é diferente.

- Remoção de **linhas duplicadas** antes da separação.
- Separação em **três conjuntos** (treino, validação e teste): o limiar e o modelo são escolhidos na validação, e o teste só avalia no final.
- Limiar escolhido por **F2** (recall com peso 2), com a curva de precisão e recall da validação.
- **Undersampling e SMOTE** comparados com o peso de classe, dentro de `imblearn.Pipeline` (só no treino).
- **GridSearchCV** no XGBoost, com PR-AUC como métrica.
- Explicação com **importância por ganho, por permutação e SHAP**, incluindo um caso de fraude não detectada e um alerta falso.

## Como rodar

```bash
pip install -r requirements.txt
jupyter notebook deteccao_fraude_cartao.ipynb
```

A base é carregada **pelo link** (cópia pública do dataset `creditcard.csv`), e não fica no repositório. Se o link estiver indisponível, o notebook usa um arquivo local `creditcard.csv` (ignorado pelo Git), cujo caminho pode ser definido na variável de ambiente `CREDITCARD_CSV`.

As saídas salvas no notebook vêm de uma execução com a base completa. Nessa execução o link estava inacessível no ambiente e foi usado o arquivo local, com o mesmo conteúdo.

## Limitações

- Resultados de **uma única separação** treino/validação/teste; validação cruzada repetida daria um intervalo de confiança para o recall.
- O limiar vem do F2. Em uso real, deve vir do **custo de uma fraude não detectada contra o custo de investigar um alerta**.
- Variáveis de comportamento ao longo do tempo (transações em sequência) exigiriam identificar o cartão, o que a base anonimizada não traz.
