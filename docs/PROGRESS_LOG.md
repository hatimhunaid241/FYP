# HKTVmall Semantic Search - Progress Log

## Week 1: March 6-12, 2026

### March 6 (Day 1)
**Status:** 🟡 In Progress
**Owner:** [Team]

**Completed:**
- [ ] Project structure created
- [ ] Dependencies installed
- [ ] API client template created
- [ ] Configuration files set up

**In Progress:**
- [ ] Testing API connection
- [ ] First product collection test

**Blockers:**
- None yet

**Notes:**
- Started late but making progress
- Need to verify API access ASAP

---

### March 7 (Day 2)
**Status:** ⚪ Not Started
**Owner:** 

**Planned:**
- [ ] Validate API client works
- [ ] Collect first 5,000 products ("apple")
- [ ] Inspect data structure
- [ ] Adjust schema if needed

---

### March 8-9 (Days 3-4)
**Status:** ⚪ Not Started
**Owner:**

**Planned:**
- [ ] Parallel collection for all 10 keywords
- [ ] Collect product reviews
- [ ] Handle errors and edge cases
- [ ] Validate data quality

**Target:** 50,000 products collected

---

### March 10-12 (Days 5-7)
**Status:** ⚪ Not Started
**Owner:**

**Planned:**
- [ ] EDA notebook creation
- [ ] Language distribution analysis
- [ ] Missing data analysis
- [ ] Data quality report

**Deliverable:** EDA Report

---

## Week 2: March 13-21, 2026

### March 13-15 (Days 8-10)
**Status:** ⚪ Not Started
**Owner:**

**Planned:**
- [ ] Text preprocessing implementation
- [ ] Embedding generation (all 50K products)
- [ ] Embedding visualization (UMAP)

**Deliverable:** Product embeddings

---

### March 16-18 (Days 11-13)
**Status:** ⚪ Not Started
**Owner:**

**Planned:**
- [ ] K-Means clustering implementation
- [ ] Hyperparameter tuning (K selection)
- [ ] Cluster evaluation metrics
- [ ] Cluster label generation

**Deliverable:** Cluster assignments + metrics

---

### March 19-21 (Days 14-15)
**Status:** ⚪ Not Started
**Owner:**

**Planned:**
- [ ] Semantic search implementation
- [ ] Keyword baseline implementation
- [ ] Initial comparison
- [ ] Checkpoint presentation prep

**Deliverable:** Working search prototype

---

## Key Metrics Tracker

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Products Collected | 50,000 | 0 | ⚪ |
| Unique Products | ~45,000 | 0 | ⚪ |
| Reviews Collected | ~100K | 0 | ⚪ |
| Embeddings Generated | 50,000 | 0 | ⚪ |
| Clustering Silhouette Score | > 0.3 | - | ⚪ |
| Search Precision@10 | > Baseline | - | ⚪ |

---

## Decision Log

### Decision 001 (March 6)
**Topic:** Embedding Model Selection
**Decision:** Use `paraphrase-multilingual-mpnet-base-v2`
**Rationale:** Supports both English and Chinese, good performance
**Status:** Confirmed

### Decision 002 (March 6)
**Topic:** Initial Clustering Algorithm
**Decision:** Start with K-Means (K=20)
**Rationale:** Simple, fast, good baseline
**Status:** Confirmed

---

## Risks & Issues

| ID | Risk | Probability | Impact | Mitigation | Status |
|----|------|-------------|--------|------------|--------|
| R01 | API rate limiting | Medium | High | Add delays, overnight collection | Active |
| R02 | Timeline too tight | High | High | Reduce scope if needed | Monitor |
| R03 | Chinese text complexity | Medium | Medium | Start English-only | Planned |

---

## Team Notes

[Add any team communication, decisions, or important notes here]
