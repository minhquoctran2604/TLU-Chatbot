# RAG Benchmark Report

**Model**: xlm-roberta-large  
**Lang**: vi  
**Bootstrap resamples**: 1000  
**Queries**: 25 | **Total entries**: 125 | **After median collapse**: 124


## BERTScore F1 by Mode (with 95% CI)

| Mode | N | Mean F1 | 95% CI | Median |
|---|---|---|---|---|
| bm25 | 25 | 0.8947 | [0.8840, 0.9044] | 0.8943 |
| naive | 25 | 0.8947 | [0.8830, 0.9070] | 0.8886 |
| hybrid | 24 | 0.8946 | [0.8837, 0.9059] | 0.8930 |
| mix | 25 | 0.8972 | [0.8895, 0.9059] | 0.8941 |
| graph | 25 | 0.8842 | [0.8738, 0.8946] | 0.8837 |


## BERTScore F1 by Type × Mode (mean)

| Type | bm25 | naive | hybrid | mix | graph |
|---|---|---|---|---|---|
| factoid | 0.9106 | 0.9109 | 0.8948 | 0.9006 | 0.8908 |
| relational | 0.8766 | 0.8809 | 0.8769 | 0.8884 | 0.8659 |
| broad | 0.8894 | 0.8866 | 0.9054 | 0.9010 | 0.8896 |
| aggregate | 0.9018 | 0.8998 | 0.9028 | 0.8977 | 0.8894 |


## Latency by Mode

| Mode | N | Mean | Median | Max |
|---|---|---|---|---|
| bm25 | 25 | 5.62s | 5.07s | 16.48s |
| naive | 25 | 18.02s | 15.34s | 44.64s |
| hybrid | 24 | 42.80s | 40.30s | 75.31s |
| mix | 25 | 42.33s | 40.40s | 80.95s |
| graph | 25 | 29.39s | 28.01s | 45.91s |


## Errors by Mode

| Mode | Error count |
|---|---|
| bm25 | 0 |
| naive | 0 |
| hybrid | 1 |
| mix | 0 |
| graph | 0 |


## Expected vs Actual Best Mode (per query)

| Query | Type | Expected | Actual | F1 | Match |
|---|---|---|---|---|---|
| httt-agg-01 | aggregate | mix | mix | 0.9293 | ✓ |
| httt-agg-02 | aggregate | mix | naive | 0.9482 | ✗ |
| httt-broad-01 | broad | mix | graph | 0.9083 | ✗ |
| httt-broad-02 | broad | mix | naive | 0.9593 | ✗ |
| httt-fact-01 | factoid | naive | bm25 | 0.9514 | ✗ |
| httt-fact-02 | factoid | naive | graph | 0.9146 | ✗ |
| httt-rel-01 | relational | graph | mix | 0.9213 | ✗ |
| httt-rel-02 | relational | graph | naive | 0.9097 | ✗ |
| ktmt-agg-01 | aggregate | mix | mix | 0.8973 | ✓ |
| ktmt-agg-02 | aggregate | mix | bm25 | 0.9019 | ✗ |
| ktmt-broad-01 | broad | mix | bm25 | 0.8941 | ✗ |
| ktmt-broad-02 | broad | mix | hybrid | 0.8999 | ✗ |
| ktmt-broad-03 | broad | mix | mix | 0.8961 | ✓ |
| ktmt-fact-01 | factoid | naive | mix | 0.9118 | ✗ |
| ktmt-fact-02 | factoid | naive | naive | 0.9188 | ✓ |
| ktmt-fact-03 | factoid | naive | bm25 | 0.8971 | ✗ |
| ktmt-rel-01 | relational | graph | bm25 | 0.8932 | ✗ |
| ktmt-rel-02 | relational | graph | naive | 0.8956 | ✗ |
| ktmt-rel-03 | relational | graph | hybrid | 0.8975 | ✗ |
| web-agg-01 | aggregate | mix | hybrid | 0.9047 | ✗ |
| web-broad-01 | broad | mix | hybrid | 0.8859 | ✗ |
| web-broad-02 | broad | mix | mix | 0.9030 | ✓ |
| web-fact-01 | factoid | naive | naive | 0.9410 | ✓ |
| web-fact-02 | factoid | naive | bm25 | 0.8989 | ✗ |
| web-rel-01 | relational | graph | naive | 0.8778 | ✗ |

**Expected-mode hit rate**: 6/25 = 24.0%