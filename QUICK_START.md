# 🚀 Quick Start Guide

## TODAY (March 6, 2026) - Immediate Actions

### Step 1: Setup Environment (30 minutes)

```powershell
# Navigate to project directory
cd "c:\Users\Hatim\Documents\Hatim\FYP"

# Create directory structure
python setup.py

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Step 2: Configure API (1 hour)

1. **Test your HKTVmall API:**
   - Open `src/data_collection/api_client.py`
   - Replace placeholder URLs with actual endpoints
   - Add authentication if needed
   
2. **Run test collection:**
```powershell
python src/data_collection/api_client.py
```

3. **Check output:**
   - Look for `data/raw/sample_response.json`
   - Inspect structure to understand data format

### Step 3: First Data Collection (Tonight - runs overnight)

```powershell
# Collect products for "apple" keyword
python src/data_collection/collect_products.py
```

**Goal:** Wake up tomorrow with 5,000 products collected

---

## TOMORROW (March 7) - Scale Up

### Step 4: Full Collection

Once single keyword works:
1. Verify data quality
2. Run full collection (all 10 keywords)
3. Monitor progress throughout the day

**Expected time:** 6-8 hours (run in background)

---

## Common Issues & Solutions

### Issue: "ModuleNotFoundError"
```powershell
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Issue: "API returns empty results"
```python
# Check your API client configuration
# Verify endpoints and authentication
# Add print statements to debug
```

### Issue: "Out of memory"
```python
# Reduce batch size in config.yaml
# embeddings -> batch_size: 16 (instead of 32)
```

### Issue: "Slow embedding generation"
```python
# Verify GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# If False, install GPU version of PyTorch:
# Visit: https://pytorch.org/get-started/locally/
```

---

## Project File Overview

### Critical Files to Edit First:

1. **`src/data_collection/api_client.py`**
   - Replace placeholder URLs
   - Add actual API endpoints
   - Test authentication

2. **`config/config.yaml`**
   - Verify keywords list
   - Adjust target counts if needed
   - Set API delay (avoid rate limiting)

### Files You'll Edit Later:

- `src/preprocessing/*.py` - Week 2
- `src/embeddings/*.py` - Week 2
- `src/clustering/*.py` - Week 2

---

## Team Collaboration Tips

### Divide the Work:

**Person 1 (Data Lead):**
- API client implementation
- Data collection
- Data quality checks

**Person 2 (ML Lead):**
- Preprocessing pipeline
- Embedding generation
- Clustering implementation

**Person 3 (Evaluation Lead):**
- EDA notebooks
- Visualization
- Metrics computation

### Communication:

- **Daily updates** in `docs/PROGRESS_LOG.md`
- **Use Git** for version control
- **Test on small samples** before full runs

---

## Testing Checklist

Before running full collection:

- [ ] API client returns valid data
- [ ] Data is saved correctly to disk
- [ ] Can load data back into pandas
- [ ] Checkpoint saving works
- [ ] Error handling works (try invalid keyword)

---

## Resource Management

### GPU Usage:
```python
# Check GPU memory
nvidia-smi

# If running out of memory:
# - Reduce batch size
# - Process in chunks
# - Use CPU for some tasks
```

### Disk Space:
- Raw JSON: ~5-10 GB
- Processed Parquet: ~1-2 GB
- Embeddings: ~200 MB
- **Total needed:** ~15-20 GB

---

## When Things Go Wrong

### Collection Fails Halfway:
- Checkpoint files are saved after each keyword
- Resume from last checkpoint
- Don't start from scratch

### Data Looks Wrong:
- Inspect `data/raw/sample_response.json`
- Check field mappings in `collect_products.py`
- Adjust schema parsing

### Can't Install Dependencies:
- Try installing one at a time
- Check Python version (need 3.8+)
- Use virtual environment

---

## Validation Steps

After each milestone, verify:

### After Collection:
```python
import pandas as pd
df = pd.read_parquet('data/processed/products.parquet')
print(f"Total products: {len(df)}")
print(f"Unique products: {df['id'].nunique()}")
print(f"Keywords: {df['keyword_source'].value_counts()}")
```

### After Embeddings:
```python
import numpy as np
embeddings = np.load('data/embeddings/product_embeddings.npy')
print(f"Shape: {embeddings.shape}")  # Should be (50000, 768)
print(f"Mean: {embeddings.mean():.3f}")  # Should be close to 0
```

### After Clustering:
```python
df = pd.read_csv('results/clusters/cluster_assignments.csv')
print(f"Number of clusters: {df['cluster_id'].nunique()}")
print(f"Cluster sizes:\n{df['cluster_id'].value_counts()}")
```

---

## Emergency Contacts

**If truly stuck:**
- Check documentation in `docs/`
- Review `notebooks/` for examples
- Read error logs in `logs/`

**Remember:**
- Progress > Perfection
- Test small before scaling
- Save checkpoints frequently
- Document what you learn

---

## Success Criteria for Week 1

By March 12, you should have:
- ✅ 50,000 products collected
- ✅ Data cleaned and saved
- ✅ EDA completed
- ✅ Understanding of data quality

**If behind schedule:** Reduce to 30,000 products minimum

---

## Next Week Preview (March 13-21)

You'll be working on:
- Bilingual text preprocessing
- Embedding generation (exciting part!)
- K-Means clustering
- Cluster visualization
- Initial search prototype

**Keep this week simple** - focus on getting clean data.

Good luck! 🚀
