# Immediate Action Plan: March 6-21

## 🚨 Current Status
**Date:** March 6, 2026
**Days Until Model Development Deadline:** 15 days
**Phase:** Behind schedule - need to accelerate

---

## Week 1: March 6-12 (Data Collection + EDA)

### Day 1-2 (Mar 6-7): Setup + Initial Data Collection
**Owner:** [Assign team member]

**Tasks:**
- [ ] Set up project structure
- [ ] Install all dependencies: `pip install -r requirements.txt`
- [ ] Test HKTVmall API connection
- [ ] Implement `api_client.py` basic wrapper
- [ ] Collect first 5,000 products (1 keyword: "apple")
- [ ] Validate data structure

**Deliverable:** Working API client + 5,000 products stored

**Code to write:**
```python
# src/data_collection/api_client.py
# src/data_collection/collect_products.py
```

---

### Day 3-4 (Mar 8-9): Full Data Collection
**Owner:** [Assign team member]

**Tasks:**
- [ ] Parallelize collection across 10 keywords
- [ ] Collect all 50,000 products
- [ ] Collect reviews for products
- [ ] Handle errors and API rate limits
- [ ] Save to structured format (Parquet)

**Deliverable:** 50,000 products + reviews in `data/processed/`

**Critical:** If collection is slow, reduce to 30,000 products (3,000 per keyword)

---

### Day 5-7 (Mar 10-12): Exploratory Data Analysis
**Owner:** [Assign team member]

**Tasks:**
- [ ] Load data into Jupyter notebook
- [ ] Analyze language distribution (English vs Chinese)
- [ ] Analyze category distribution
- [ ] Visualize missing data
- [ ] Text length statistics
- [ ] Identify data quality issues
- [ ] Create EDA summary report

**Deliverable:** `notebooks/01_data_exploration.ipynb` with findings

**Key questions to answer:**
1. How many products per keyword?
2. What percentage are Chinese vs English?
3. Average review count per product?
4. Category distribution?
5. Data quality issues?

---

## Week 2: March 13-21 (Embeddings + Clustering)

### Day 8-10 (Mar 13-15): Text Preprocessing + Embeddings
**Owner:** [Assign team member]

**Tasks:**
- [ ] Implement bilingual text preprocessing
- [ ] Test on sample (1,000 products)
- [ ] Generate embeddings for all 50,000 products
- [ ] Save embeddings to disk
- [ ] Visualize embeddings with UMAP/t-SNE (2D projection)

**Deliverable:** 
- `data/embeddings/product_embeddings.npy`
- Visualization showing semantic clusters emerging

**Code to write:**
```python
# src/preprocessing/text_cleaner.py
# src/preprocessing/chinese_processor.py
# src/preprocessing/english_processor.py
# src/embeddings/embedding_generator.py
```

**GPU Usage:** This is where GPU helps - batch process embeddings efficiently

---

### Day 11-13 (Mar 16-18): Clustering Experiments
**Owner:** [Assign team member]

**Tasks:**
- [ ] Implement K-Means clustering
- [ ] Test different K values (10, 15, 20, 25, 30)
- [ ] Compute clustering metrics (silhouette score)
- [ ] Generate cluster labels using TF-IDF
- [ ] Manual inspection of clusters
- [ ] Validate "apple" separates into meaningful groups

**Deliverable:** 
- `results/clusters/cluster_assignments.csv`
- Clustering evaluation report

**Code to write:**
```python
# src/clustering/kmeans_cluster.py
# src/clustering/cluster_labeling.py
# src/evaluation/clustering_metrics.py
```

---

### Day 14-15 (Mar 19-21): Initial Search + Baseline
**Owner:** [Assign team member]

**Tasks:**
- [ ] Implement semantic search (cosine similarity)
- [ ] Implement keyword baseline (for comparison)
- [ ] Test on sample queries
- [ ] Document initial results
- [ ] Prepare checkpoint presentation for team

**Deliverable:** 
- Working semantic search prototype
- Comparison results: semantic vs keyword search

**Code to write:**
```python
# src/search/semantic_search.py
# src/search/keyword_baseline.py
```

---

## Team Division Strategy

**Option 1: Pipeline Parallelization**
- **Person 1:** Data collection + preprocessing
- **Person 2:** Embeddings + clustering  
- **Person 3:** Evaluation + documentation

**Option 2: Feature Parallelization**
- **Person 1:** English pipeline (end-to-end)
- **Person 2:** Chinese pipeline (end-to-end)
- **Person 3:** Integration + evaluation

**Recommended:** Option 1 for faster progress

---

## Critical Success Factors

### Must Have by March 21:
✅ 50,000 products collected
✅ Embeddings generated
✅ Initial clustering with K-Means
✅ Cluster quality metrics computed
✅ Evidence that semantic ambiguity is resolved (e.g., "apple" clusters)

### Nice to Have by March 21:
- Sentiment analysis started
- Multiple clustering algorithms tested
- Quality scoring implemented

### Can Skip for Now:
- ❌ HDBSCAN (do K-Means first)
- ❌ Fine-tuned embeddings (use pre-trained)
- ❌ Advanced quality scoring (use simple rating-based)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| API rate limiting | Add delays, collect overnight |
| Slow embedding generation | Use GPU, batch process |
| Poor clustering results | Try different K values, check preprocessing |
| Bilingual complexity too high | Start English-only, add Chinese later |
| Timeline too tight | Cut scope: reduce to 30K products |

---

## Communication Protocol

**Daily Standup (async):**
- What did you complete yesterday?
- What will you do today?
- Any blockers?

**Checkpoints:**
- **March 9:** Have 50K products collected?
- **March 12:** EDA complete?
- **March 15:** Embeddings ready?
- **March 18:** Clustering working?
- **March 21:** Initial results ready?

---

## Next Immediate Steps (March 6, Today)

1. **Create project folders:** Run setup script
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Test API:** Write simple script to fetch 10 products
4. **Assign roles:** Decide who does what
5. **Start collection:** Begin collecting "apple" products overnight

**Tonight's goal:** Have 5,000 products in `data/raw/`
