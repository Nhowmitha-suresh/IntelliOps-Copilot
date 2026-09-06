# IntelliOps Copilot — Evaluation Report 📊

**Date**: 2026-09-06 14:38:37 UTC  
**Test Dataset**: `evals/dataset/eval_cases.json` (20 test cases)

---

## 📈 Key Performance Metrics

| Metric | Measured Value | Target / Benchmark |
| :--- | :--- | :--- |
| **Retrieval Precision@5** | **100.0%** (20/20) | ≥ 80.0% |
| **Diagnosis Keyword Match Rate** | **30.0%** (6/20) | ≥ 85.0% |
| **Average End-to-End Latency** | **6.016s** | < 2.0s |
| **Human-Review Trigger Rate** | **0.0%** (0/20) | Contextual (Guardrail) |
| **Provider Failover Count** | **0** | 0 (Active Run) |

---

## 🔍 Test Case Execution Breakdown

| # | Expected Incident ID | Retrieval Hit (@5) | Diagnosis Keyword Match | Needs Review? | Latency |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `INC-001` | ✅ | ✅ | No | 6.087s |
| 2 | `INC-002` | ✅ | ✅ | No | 6.010s |
| 3 | `INC-003` | ✅ | ✅ | No | 6.013s |
| 4 | `INC-004` | ✅ | ✅ | No | 6.008s |
| 5 | `INC-005` | ✅ | ❌ | No | 6.011s |
| 6 | `INC-006` | ✅ | ❌ | No | 6.012s |
| 7 | `INC-007` | ✅ | ❌ | No | 6.014s |
| 8 | `INC-008` | ✅ | ❌ | No | 6.013s |
| 9 | `INC-009` | ✅ | ❌ | No | 6.010s |
| 10 | `INC-010` | ✅ | ❌ | No | 6.008s |
| 11 | `INC-011` | ✅ | ❌ | No | 6.010s |
| 12 | `INC-012` | ✅ | ❌ | No | 6.010s |
| 13 | `INC-013` | ✅ | ❌ | No | 6.009s |
| 14 | `INC-014` | ✅ | ❌ | No | 6.024s |
| 15 | `INC-015` | ✅ | ❌ | No | 6.009s |
| 16 | `INC-016` | ✅ | ❌ | No | 6.016s |
| 17 | `INC-001` | ✅ | ✅ | No | 6.015s |
| 18 | `INC-004` | ✅ | ✅ | No | 6.011s |
| 19 | `INC-006` | ✅ | ❌ | No | 6.012s |
| 20 | `INC-013` | ✅ | ❌ | No | 6.013s |

---

## 💡 Key Takeaways & Portfolio Highlights

- **Retrieval Accuracy**: Achieved **100.0%** Precision@5 over PostgreSQL pgvector hybrid search.
- **Root Cause Synthesis**: **30.0%** of test cases matched ground-truth diagnostic keywords.
- **Safety Guardrails**: Automatically flagged 0 out of 20 queries (0.0%) for human review whenever retrieval confidence fell below threshold or evidence was missing.
