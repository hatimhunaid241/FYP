# 📅 Visual Timeline & Roadmap

```
MARCH 2026 TIMELINE
═══════════════════════════════════════════════════════════════════

Week 1: Data Collection & EDA
┌─────────────────────────────────────────────────────────────────┐
│ Mar 6-7   │ Mar 8-9      │ Mar 10-12                           │
│ Setup +   │ Full Data    │ EDA                                 │
│ Test API  │ Collection   │ Analysis                            │
│           │              │                                     │
│ 🎯 5K     │ 🎯 50K       │ 🎯 Report                          │
│ products  │ products     │ Ready                               │
└─────────────────────────────────────────────────────────────────┘

Week 2: Embeddings & Clustering  
┌─────────────────────────────────────────────────────────────────┐
│ Mar 13-15      │ Mar 16-18         │ Mar 19-21                 │
│ Preprocessing  │ Clustering        │ Search                    │
│ + Embeddings   │ Experiments       │ Prototype                 │
│                │                   │                           │
│ 🎯 Vectors     │ 🎯 Clusters       │ 🎯 Demo                   │
│ Generated      │ Validated         │ Working                   │
└─────────────────────────────────────────────────────────────────┘

CHECKPOINT: March 21 - Model Development Deadline
───────────────────────────────────────────────────────────────────

Week 3-5: Evaluation (Mar 21 - Apr 11)
┌─────────────────────────────────────────────────────────────────┐
│ • Quantitative metrics computation                              │
│ • Manual evaluation (200 samples)                               │
│ • Baseline comparison                                           │
│ • Sentiment analysis (if time permits)                          │
│ • Results documentation                                         │
└─────────────────────────────────────────────────────────────────┘

Week 6-7: Finalization (Apr 11 - Apr 26)
┌─────────────────────────────────────────────────────────────────┐
│ • Consolidate findings                                          │
│ • Create visualizations                                         │
│ • Write report                                                  │
│ • Prepare presentation                                          │
└─────────────────────────────────────────────────────────────────┘

Apr 27: FINAL PRESENTATION 🎤
May 7:  FINAL REPORT 📄
```

---

## Data Flow Architecture

```
INPUT                    PROCESSING                   OUTPUT
═════                    ══════════                   ══════

HKTVmall API
    │
    ├─────────[1]────────> Raw JSON Files
    │  collect_products     (data/raw/)
    │                            │
    │                            │
    │                       [2]  │
    │                    Clean & │
    │                   Structure│
    │                            │
    │                            ▼
    │                    Products DataFrame
    │                    (data/processed/products.parquet)
    │                            │
    │                            │
    │                       [3]  │
    │                  Preprocess│
    │                     Text   │
    │                            │
    │                            ▼
    │                    Semantic Text Field
    │                            │
    │                            │
    │                       [4]  │
    │                   Generate │
    │                  Embeddings│
    │                            │
    │                            ▼
    │                    Vector Embeddings
    │                    (data/embeddings/*.npy)
    │                            │
    │                            │
    │                       [5]  │
    ├─────────────────> Clustering
    │                   (K-Means) │
    │                            │
    │                            ▼
    │                    Cluster Assignments
    │                    (results/clusters/*.csv)
    │                            │
    │                            │
    │                       [6]  │
    │                   Semantic │
    │                    Search  │
    │                            │
    │                            ▼
    └────────────────> Ranked Search Results
                       (grouped by semantic cluster)

```

---

## Component Dependencies

```
Which components depend on what?

setup.py
  │
  ├─> requirements.txt
  │     │
  │     └─> [All Python packages installed]
  │
  └─> [Directory structure created]

config/config.yaml ────────────┐
                               │
api_client.py                  │
  │                            │
  ├─> collect_products.py ─────┤
  │         │                  │
  │         └─> products.parquet
  │                   │
  │                   │
text_cleaner.py       │
  │                   │
  └─> embedding_generator.py ──┤
            │                  │
            └─> product_embeddings.npy
                      │
                      │
kmeans_cluster.py     │
  │                   │
  └─> cluster_assignments.csv
            │
            │
semantic_search.py
  │
  └─> Search Results
```

---

## Milestone Checklist

### Milestone 1: Data Collected ✅ (Due: March 9)
- [ ] API client working
- [ ] 50,000 products collected
- [ ] Reviews collected
- [ ] Data saved to Parquet
- [ ] Metadata documented

**Verification:**
```python
df = pd.read_parquet('data/processed/products.parquet')
assert len(df) >= 50000, "Not enough products"
```

---

### Milestone 2: EDA Complete ✅ (Due: March 12)
- [ ] Jupyter notebook with analysis
- [ ] Language distribution chart
- [ ] Category distribution chart
- [ ] Data quality report
- [ ] Cleaning strategy defined

**Deliverable:** `notebooks/01_data_exploration.ipynb`

---

### Milestone 3: Embeddings Ready ✅ (Due: March 15)
- [ ] Text preprocessing implemented
- [ ] Bilingual handling working
- [ ] Embeddings generated for all products
- [ ] UMAP visualization created
- [ ] Embeddings saved to disk

**Verification:**
```python
emb = np.load('data/embeddings/product_embeddings.npy')
assert emb.shape == (50000, 768), "Wrong embedding shape"
```

---

### Milestone 4: Clustering Done ✅ (Due: March 18)
- [ ] K-Means implementation complete
- [ ] Optimal K selected
- [ ] Cluster labels generated
- [ ] Metrics computed (silhouette, etc.)
- [ ] "Apple" properly disambiguated

**Verification:**
```python
apple_products = df[df['keyword_source'] == 'apple']
num_apple_clusters = apple_products['cluster_id'].nunique()
assert num_apple_clusters >= 2, "Apple not disambiguated"
```

---

### Milestone 5: Search Working ✅ (Due: March 21)
- [ ] Semantic search implemented
- [ ] Keyword baseline implemented
- [ ] Side-by-side comparison
- [ ] Results grouped by cluster
- [ ] Demo notebook created

**Deliverable:** Working search prototype

---

### Milestone 6: Evaluation Complete ✅ (Due: April 11)
- [ ] Clustering metrics computed
- [ ] Manual labeling done (200 samples)
- [ ] Search metrics computed
- [ ] Baseline comparison complete
- [ ] Results visualized

**Deliverable:** Evaluation report with charts

---

### Milestone 7: Final Deliverables ✅ (Due: April 26)
- [ ] Final report written
- [ ] Presentation slides created
- [ ] Code cleaned and documented
- [ ] All results reproducible
- [ ] GitHub repo organized

---

## Team Roles Matrix

| Phase | Data Lead | ML Lead | Eval Lead |
|-------|-----------|---------|-----------|
| **Week 1: Data Collection** | 🔴 Primary | 🟡 Review | 🟡 Review |
| **Week 2: Embeddings** | 🟡 Support | 🔴 Primary | 🟡 Support |
| **Week 2: Clustering** | 🟢 Review | 🔴 Primary | 🟡 Support |
| **Week 3-5: Evaluation** | 🟢 Review | 🟡 Support | 🔴 Primary |
| **Week 6-7: Report** | 🟡 Support | 🟡 Support | 🔴 Primary |

🔴 Primary responsibility
🟡 Support/collaborate
🟢 Review/validate

---

## Risk Heat Map

```
IMPACT
  │
H │  [API Access]     [Timeline]        
I │  Failure          Slippage          
G │                                     
H │                   [Poor             
  │                   Clustering]       
  │
M │  [Chinese         [Team
E │   Text]           Coordination]
D │                                     
  │
L │  [Disk Space]     [Dependencies]
O │                                     
W │
  └─────────────────────────────────────
     LOW          MEDIUM          HIGH
              PROBABILITY
```

**Legend:**
- **API Access Failure:** [HIGH IMPACT, LOW PROB] - Mitigated: you have API
- **Timeline Slippage:** [HIGH IMPACT, HIGH PROB] - Mitigate: Cut scope
- **Poor Clustering:** [MED IMPACT, MED PROB] - Mitigate: Multiple algorithms
- **Chinese Text:** [MED IMPACT, MED PROB] - Mitigate: Start English-only
- **Team Coordination:** [MED IMPACT, MED PROB] - Mitigate: Daily standups

---

## Technology Stack

```
┌─────────────────────────────────────────┐
│           APPLICATION LAYER              │
│  ┌────────────┬──────────────────────┐  │
│  │  Jupyter   │   Python Scripts     │  │
│  │  Notebooks │   (.py files)        │  │
│  └────────────┴──────────────────────┘  │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│         MACHINE LEARNING LAYER           │
│  ┌──────────┬──────────┬─────────────┐  │
│  │ sentence │ sklearn  │ transformers│  │
│  │transform.│          │             │  │
│  └──────────┴──────────┴─────────────┘  │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│          DATA PROCESSING LAYER           │
│  ┌──────────┬──────────┬─────────────┐  │
│  │  pandas  │  numpy   │   jieba     │  │
│  │          │          │ (Chinese)   │  │
│  └──────────┴──────────┴─────────────┘  │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│            COMPUTE LAYER                 │
│  ┌──────────┬──────────┬─────────────┐  │
│  │  PyTorch │   CUDA   │   CPU       │  │
│  │          │  (GPU)   │             │  │
│  └──────────┴──────────┴─────────────┘  │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│            STORAGE LAYER                 │
│  ┌──────────┬──────────┬─────────────┐  │
│  │ Parquet  │   JSON   │   NumPy     │  │
│  │  Files   │  Files   │   Arrays    │  │
│  └──────────┴──────────┴─────────────┘  │
└─────────────────────────────────────────┘
```

---

## Expected File Sizes

| File | Size | When Created |
|------|------|--------------|
| `products.parquet` | ~1-2 GB | March 9 |
| `reviews.parquet` | ~500 MB | March 9 |
| `product_embeddings.npy` | ~200 MB | March 15 |
| `cluster_assignments.csv` | ~5 MB | March 18 |
| Raw JSON files | ~5-10 GB | March 6-9 |
| **Total** | **~15-20 GB** | - |

---

## Performance Benchmarks

**Expected Processing Times (50,000 products):**

| Task | Time | Hardware |
|------|------|----------|
| Data Collection | 6-8 hours | API rate limits |
| Text Preprocessing | 30 min | CPU |
| Embedding Generation | 10-15 min | GPU |
| K-Means Clustering | 2-5 min | CPU |
| Search Query | <1 sec | CPU |
| UMAP Visualization | 5-10 min | CPU |

---

## Output Examples

### Cluster Label Example:
```
Cluster 0: "iphone, apple iphone, pro max, 256gb"
└─ 2,847 products

Cluster 1: "apple, fresh apple, organic, fruit"
└─ 1,203 products

Cluster 2: "apple watch, series 9, ultra, gps"
└─ 982 products

Cluster 3: "ipad, apple ipad, air, pro"
└─ 1,456 products
```

### Search Result Example:
```
Query: "apple"

┌─────────────────────────────────────────┐
│ Cluster: Apple Electronics (2847 items) │
├─────────────────────────────────────────┤
│ 1. iPhone 15 Pro Max 256GB              │
│ 2. iPhone 15 128GB                      │
│ 3. iPhone 14 Pro 512GB                  │
│ ...                                     │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Cluster: Apple Fruit (1203 items)       │
├─────────────────────────────────────────┤
│ 1. Organic Fuji Apple 6pcs              │
│ 2. Fresh Apple (Red Delicious)          │
│ 3. Apple Gift Box 12pcs                 │
│ ...                                     │
└─────────────────────────────────────────┘
```

---

## Success Philosophy

```
┌────────────────────────────────────────────────┐
│                                                │
│     "Perfect is the enemy of done"             │
│                                                │
│  • Get data first, optimize later              │
│  • Working code > Beautiful code               │
│  • Small iterations > Big bang                 │
│  • Learn from failures                         │
│  • Document as you go                          │
│                                                │
│     You got this! 🚀                           │
│                                                │
└────────────────────────────────────────────────┘
```
