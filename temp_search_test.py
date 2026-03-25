import sys
sys.path.insert(0, '.')
from src.search.semantic_search import load_search_data, build_search_model, query_semantic_search

data = load_search_data()
model = build_search_model()
results = query_semantic_search('usb charger', data, model, top_k=5)
print(results[['cluster', 'cluster_label', 'name_en', 'score']].to_string(index=False))
