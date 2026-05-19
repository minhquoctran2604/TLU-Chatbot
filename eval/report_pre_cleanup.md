# RAG Benchmark Report

**Model**: xlm-roberta-large  
**Lang**: vi  
**Bootstrap resamples**: 1000  
**Queries**: 25 | **Total entries**: 125 | **After median collapse**: 125


## BERTScore F1 by Mode (with 95% CI)

| Mode | N | Mean F1 | 95% CI | Median |
|---|---|---|---|---|
| bm25 | 25 | 0.8947 | [0.8840, 0.9044] | 0.8943 |
| naive | 25 | 0.8763 | [0.8638, 0.8893] | 0.8715 |
| hybrid | 25 | 0.8725 | [0.8638, 0.8812] | 0.8718 |
| mix | 25 | 0.8702 | [0.8595, 0.8807] | 0.8743 |
| graph | 25 | 0.8643 | [0.8561, 0.8722] | 0.8617 |


## BERTScore F1 by Type × Mode (mean)

| Type | bm25 | naive | hybrid | mix | graph |
|---|---|---|---|---|---|
| factoid | 0.9106 | 0.8991 | 0.8851 | 0.8861 | 0.8727 |
| relational | 0.8766 | 0.8747 | 0.8782 | 0.8727 | 0.8621 |
| broad | 0.8894 | 0.8631 | 0.8683 | 0.8631 | 0.8618 |
| aggregate | 0.9018 | 0.8650 | 0.8541 | 0.8549 | 0.8584 |


## Latency by Mode

| Mode | N | Mean | Median | Max |
|---|---|---|---|---|
| bm25 | 25 | 5.62s | 5.07s | 16.48s |
| naive | 25 | 18.34s | 17.61s | 38.00s |
| hybrid | 25 | 35.68s | 34.21s | 60.40s |
| mix | 25 | 35.06s | 32.54s | 61.91s |
| graph | 25 | 24.51s | 22.85s | 44.39s |


## Errors by Mode

| Mode | Error count |
|---|---|
| bm25 | 0 |
| naive | 0 |
| hybrid | 0 |
| mix | 0 |
| graph | 0 |


## Expected vs Actual Best Mode (per query)

| Query | Type | Expected | Actual | F1 | Match |
|---|---|---|---|---|---|
| httt-agg-01 | aggregate | mix | bm25 | 0.9269 | ✗ |
| httt-agg-02 | aggregate | mix | bm25 | 0.9137 | ✗ |
| httt-broad-01 | broad | mix | bm25 | 0.8608 | ✗ |
| httt-broad-02 | broad | mix | bm25 | 0.9492 | ✗ |
| httt-fact-01 | factoid | naive | naive | 0.9695 | ✓ |
| httt-fact-02 | factoid | naive | bm25 | 0.9145 | ✗ |
| httt-rel-01 | relational | graph | naive | 0.9241 | ✗ |
| httt-rel-02 | relational | graph | naive | 0.8707 | ✗ |
| ktmt-agg-01 | aggregate | mix | bm25 | 0.8720 | ✗ |
| ktmt-agg-02 | aggregate | mix | bm25 | 0.9019 | ✗ |
| ktmt-broad-01 | broad | mix | bm25 | 0.8941 | ✗ |
| ktmt-broad-02 | broad | mix | bm25 | 0.8873 | ✗ |
| ktmt-broad-03 | broad | mix | bm25 | 0.8796 | ✗ |
| ktmt-fact-01 | factoid | naive | naive | 0.9215 | ✓ |
| ktmt-fact-02 | factoid | naive | bm25 | 0.9057 | ✗ |
| ktmt-fact-03 | factoid | naive | graph | 0.8981 | ✗ |
| ktmt-rel-01 | relational | graph | bm25 | 0.8932 | ✗ |
| ktmt-rel-02 | relational | graph | mix | 0.8863 | ✗ |
| ktmt-rel-03 | relational | graph | hybrid | 0.9123 | ✗ |
| web-agg-01 | aggregate | mix | naive | 0.9254 | ✗ |
| web-broad-01 | broad | mix | mix | 0.9011 | ✓ |
| web-broad-02 | broad | mix | bm25 | 0.8788 | ✗ |
| web-fact-01 | factoid | naive | naive | 0.9040 | ✓ |
| web-fact-02 | factoid | naive | naive | 0.9000 | ✓ |
| web-rel-01 | relational | graph | bm25 | 0.8468 | ✗ |

**Expected-mode hit rate**: 5/25 = 20.0%