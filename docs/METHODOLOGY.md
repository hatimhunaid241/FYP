# Detailed Methodology

## Overview
This document provides implementation details for each phase of the semantic search project.

---

## Phase 1: Data Collection

### API Integration
```python
# Key considerations for HKTVmall API:
1. Rate limiting (respect server limits)
2. Error handling (retry logic)
3. Data validation (check for missing fields)
4. Incremental saving (don't lose progress)
```

### Data Schema

**Products Table:**
```
product_id: str (primary key)
keyword_source: str (which search keyword found this)
title_en: str
title_zh: str
description_en: str
description_zh: str
brand: str
category: str
price: float
rating_avg: float
rating_count: int
review_count: int
url: str
collected_at: datetime
```

**Reviews Table:**
```
review_id: str (primary key)
product_id: str (foreign key)
review_text: str
language: str (en/zh)
rating: int (1-5)
date: datetime
helpful_count: int
```

### Collection Strategy

**Approach:** Breadth-first search
1. Start with 10 seed keywords
2. For each keyword:
   - Fetch top 5,000 products
   - Extract related keywords
   - Fetch reviews (limit 100 per product)
3. Deduplicate products (same product_id from different keywords)

**Expected outcome:** 
- 50,000 total products (with some duplicates)
- ~40,000-45,000 unique products after deduplication

---

## Phase 2: Text Preprocessing

### Bilingual Processing Pipeline

```python
def preprocess_product(product):
    """
    Unified preprocessing for bilingual products
    """
    # 1. Detect primary language
    lang = detect_language(product['title'])
    
    # 2. Apply language-specific cleaning
    if lang == 'zh':
        text = preprocess_chinese(product)
    elif lang == 'en':
        text = preprocess_english(product)
    else:  # Mixed
        text = preprocess_mixed(product)
    
    # 3. Create semantic field
    semantic_text = combine_fields(product, text)
    
    return semantic_text
```

### Chinese Text Processing
```python
# Key libraries:
import jieba  # Word segmentation
from opencc import OpenCC  # Traditional -> Simplified

# Processing steps:
1. Convert Traditional Chinese to Simplified
2. Segment text using jieba
3. Remove Chinese stopwords
4. Keep brand names intact (use custom dictionary)
```

### English Text Processing
```python
# Standard NLP pipeline:
1. Lowercase
2. Remove special characters (keep hyphens in product names)
3. Remove stopwords (but keep brand-relevant words)
4. Lemmatization (optional - may hurt brand names)
```

### Feature Engineering

**Semantic Text Field:**
```python
semantic_text = f"{title} {brand} {category} {description[:200]}"
```

**Rationale:**
- Title: Most important (highest weight)
- Brand: Disambiguates (Apple Inc. vs apple fruit)
- Category: Provides context
- Description: Limited to 200 chars (avoid noise)

---

## Phase 3: Embeddings

### Model Selection

**Chosen Model:** `paraphrase-multilingual-mpnet-base-v2`

**Justification:**
- ✅ Supports both English and Chinese
- ✅ Good semantic understanding
- ✅ Reasonable size (768 dimensions)
- ✅ Fast inference on GPU

**Alternative if too slow:** `distiluse-base-multilingual-cased-v2` (smaller, faster)

### Embedding Generation

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
model = model.to('cuda')  # Use GPU

# Batch processing
embeddings = model.encode(
    semantic_texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  # For cosine similarity
)

# Save to disk
np.save('data/embeddings/product_embeddings.npy', embeddings)
```

**Performance estimate:**
- 50,000 products × 0.01 sec/product = ~8-10 minutes on GPU

---

## Phase 4: Clustering

### K-Means Clustering

**Why K-Means first:**
- Simple to implement
- Fast convergence
- Easy to interpret
- Good baseline

**Choosing K:**
```python
# Method 1: Elbow method
inertias = []
K_range = range(5, 50, 5)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)
    inertias.append(kmeans.inertia_)

# Plot and look for "elbow"
plt.plot(K_range, inertias)
```

**Method 2: Silhouette analysis**
```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    silhouette_scores.append(score)

# Choose K with highest silhouette score
```

**Starting point:** K=20 (roughly 2-3 clusters per keyword)

### Cluster Labeling

**Approach:** TF-IDF on cluster members

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_cluster_labels(cluster_id, product_texts):
    """
    Extract top keywords for cluster
    """
    tfidf = TfidfVectorizer(max_features=10, ngram_range=(1,2))
    tfidf_matrix = tfidf.fit_transform(product_texts)
    
    # Get top terms
    feature_names = tfidf.get_feature_names_out()
    top_indices = tfidf_matrix.sum(axis=0).argsort()[::-1][:5]
    top_terms = [feature_names[i] for i in top_indices]
    
    return ", ".join(top_terms)
```

**Example output:**
- Cluster 5: "iphone, apple watch, ipad, macbook, airpods"
- Cluster 12: "apple fruit, fresh apple, organic apple"

### HDBSCAN (If Time Permits)

**Why HDBSCAN:**
- Automatic K selection
- Handles varying density
- Identifies noise points

**Caution:** Many hyperparameters to tune

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,  # Min products per cluster
    min_samples=10,
    metric='euclidean',
    cluster_selection_epsilon=0.5
)

labels = clusterer.fit_predict(embeddings)
```

---

## Phase 5: Semantic Search

### Implementation

```python
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearch:
    def __init__(self, products, embeddings, model):
        self.products = products
        self.embeddings = embeddings
        self.model = model
    
    def search(self, query, top_k=50):
        # 1. Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # 2. Compute similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # 3. Get top K
        top_indices = similarities.argsort()[::-1][:top_k]
        
        # 4. Return results with cluster info
        results = []
        for idx in top_indices:
            results.append({
                'product': self.products[idx],
                'score': similarities[idx],
                'cluster': self.products[idx]['cluster_id']
            })
        
        return results
```

### Cluster-Based Grouping

```python
def group_by_cluster(results):
    """
    Group search results by semantic cluster
    """
    clusters = {}
    for result in results:
        cluster_id = result['cluster']
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(result)
    
    return clusters
```

### Keyword Baseline

```python
from sklearn.feature_extraction.text import TfidfVectorizer

class KeywordSearch:
    def __init__(self, products):
        self.products = products
        texts = [p['semantic_text'] for p in products]
        
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
    
    def search(self, query, top_k=50):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = similarities.argsort()[::-1][:top_k]
        
        return [self.products[i] for i in top_indices]
```

---

## Phase 6: Evaluation

### Clustering Evaluation

**Quantitative Metrics:**

1. **Silhouette Score** (higher is better)
```python
from sklearn.metrics import silhouette_score
score = silhouette_score(embeddings, cluster_labels)
# Range: [-1, 1], target: > 0.5
```

2. **Calinski-Harabasz Index** (higher is better)
```python
from sklearn.metrics import calinski_harabasz_score
score = calinski_harabasz_score(embeddings, cluster_labels)
```

3. **Davies-Bouldin Index** (lower is better)
```python
from sklearn.metrics import davies_bouldin_score
score = davies_bouldin_score(embeddings, cluster_labels)
# Lower means better separation
```

**Qualitative Evaluation:**

1. Manual labeling (200 products sample)
2. Compute cluster purity
3. Check if ambiguous keywords are properly separated

### Search Evaluation

**Metrics:**

1. **Precision@K**
```python
def precision_at_k(results, relevant_items, k):
    top_k = results[:k]
    relevant_retrieved = len(set(top_k) & set(relevant_items))
    return relevant_retrieved / k
```

2. **NDCG (Normalized Discounted Cumulative Gain)**
```python
from sklearn.metrics import ndcg_score
# Requires relevance scores (0-5) for each result
```

3. **Mean Reciprocal Rank (MRR)**
```python
def mrr(results, relevant_items):
    for i, item in enumerate(results):
        if item in relevant_items:
            return 1 / (i + 1)
    return 0
```

**Comparison Setup:**
- Same 50 test queries
- Compare semantic search vs keyword baseline
- Manually judge top 10 results

---

## Phase 7: Sentiment Analysis (Optional Enhancement)

**Model:** `nlptown/bert-base-multilingual-uncased-sentiment`

**Pipeline:**
```python
from transformers import pipeline

sentiment_analyzer = pipeline('sentiment-analysis', 
                             model='nlptown/bert-base-multilingual-uncased-sentiment')

def analyze_product_sentiment(reviews):
    sentiments = sentiment_analyzer(reviews)
    # Average sentiment score
    avg_sentiment = np.mean([s['score'] for s in sentiments])
    return avg_sentiment
```

**Quality Score:**
```python
def calculate_quality_score(product):
    score = (
        0.40 * normalized_rating +
        0.30 * sentiment_score +
        0.20 * log_review_count +
        0.10 * sentiment_consistency
    )
    return score
```

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Data Collection | 4 days | 50K products dataset |
| EDA | 3 days | Analysis report |
| Preprocessing + Embeddings | 3 days | Embedding vectors |
| Clustering | 3 days | Cluster assignments |
| Search + Baseline | 2 days | Working prototype |
| **Checkpoint (Mar 21)** | - | Initial results |
| Evaluation | 2 weeks | Metrics report |
| Sentiment (optional) | 1 week | Quality scores |
| Final polish | 1 week | Complete system |

---

## Success Criteria

**Minimum Viable Product (MVP):**
- ✅ 50,000 products collected
- ✅ Embeddings generated
- ✅ K-Means clustering with K=20
- ✅ Silhouette score > 0.3
- ✅ "Apple" query separates into 2+ distinct clusters
- ✅ Semantic search outperforms keyword baseline on test queries

**Stretch Goals:**
- Try HDBSCAN
- Implement sentiment analysis
- Support Chinese query search
- Create interactive visualization
