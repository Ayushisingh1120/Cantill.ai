import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sentence_transformers import SentenceTransformer

def clean_text_for_classification(text: str) -> str:
    if pd.isna(text) or not text:
        return ""
    s = str(text).lower()
    s = re.sub(r'[\r\n\t]+', ' ', s)
    s = re.sub(r'[^a-z0-9\s/_\-]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

class SpendClassifier:
    """
    Dual-mode spend taxonomy classifier:
    1. Zero-shot semantic embedding + TF-IDF hybrid matching on unseen taxonomies.
    2. Supervised few-shot / transfer learning when seed/training labels exist.
    """
    def __init__(self, taxonomy_df: pd.DataFrame, use_zero_shot: bool = False):
        self.taxonomy_df = taxonomy_df.copy().fillna('')
        self.levels = [c for c in taxonomy_df.columns if c.startswith('category_l') or c.lower() in ('l1', 'l2', 'l3', 'l4')]
        self.levels.sort()
        self.lowest_level = self.levels[-1] if self.levels else 'category_l3'
        
        # Build path string for each leaf
        self.tax_paths = []
        for _, r in self.taxonomy_df.iterrows():
            parts = [str(r[lvl]).strip() for lvl in self.levels if str(r[lvl]).strip()]
            self.tax_paths.append(" > ".join(parts))
        
        self.use_zero_shot = use_zero_shot
        self.supervised_model = None
        self.supervised_vec = None
        self.classes_ = []
        
        # Load local embedding model
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tax_embeddings = self.embed_model.encode(self.tax_paths, normalize_embeddings=True, show_progress_bar=False)
        
        # TF-IDF on taxonomy paths
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.tax_tfidf = self.tfidf.fit_transform(self.tax_paths)

    def train_supervised(self, texts: List[str], labels: List[str]):
        """Train fast, calibrated linear classifier on available labels."""
        if self.use_zero_shot or len(texts) == 0:
            return
        clean_texts = [clean_text_for_classification(t) for t in texts]
        self.supervised_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=15000)
        X = self.supervised_vec.fit_transform(clean_texts)
        self.supervised_model = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
        self.supervised_model.fit(X, labels)
        self.classes_ = list(self.supervised_model.classes_)

    def predict_batch(self, texts: List[str]) -> List[Tuple[Dict[str, str], float]]:
        """
        Predict hierarchical category path and calibrated confidence for each text.
        Returns: list of (path_dict, confidence)
        """
        results = []
        clean_texts = [clean_text_for_classification(t) for t in texts]
        
        # Supervised prediction if available
        supervised_preds = None
        supervised_probs = None
        if self.supervised_model is not None and not self.use_zero_shot:
            try:
                X_sup = self.supervised_vec.transform(clean_texts)
                probs = self.supervised_model.predict_proba(X_sup)
                max_indices = np.argmax(probs, axis=1)
                supervised_preds = [self.classes_[idx] for idx in max_indices]
                supervised_probs = np.max(probs, axis=1)
            except Exception as e:
                print(f"Supervised prediction fallback to zero-shot: {e}")
                supervised_preds = None
        
        # Batch zero-shot prediction
        # Deduplicate texts for fast embedding computation
        unique_texts = list(set(clean_texts))
        if not unique_texts:
            return [({}, 0.0)] * len(texts)
            
        unique_embs = self.embed_model.encode(unique_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        unique_tfidf = self.tfidf.transform(unique_texts)
        
        sim_dense = np.dot(unique_embs, self.tax_embeddings.T) # shape: (len(unique), len(tax))
        sim_sparse = (unique_tfidf * self.tax_tfidf.T).toarray()
        
        sim_hybrid = 0.6 * sim_dense + 0.4 * sim_sparse
        best_indices = np.argmax(sim_hybrid, axis=1)
        best_scores = np.max(sim_hybrid, axis=1)
        
        # Map text to best index and score
        text_to_idx = {t: idx for t, idx in zip(unique_texts, best_indices)}
        text_to_score = {t: float(sc) for t, sc in zip(unique_texts, best_scores)}
        
        # Build path lookup from taxonomy_df
        path_lookup = {}
        for idx, row in self.taxonomy_df.iterrows():
            leaf = str(row[self.lowest_level]).strip()
            path_dict = {}
            for i, lvl in enumerate(self.levels):
                lvl_key = f"l{i+1}"
                val = str(row[lvl]).strip() if pd.notna(row[lvl]) else ""
                path_dict[lvl_key] = val
            path_lookup[idx] = path_dict
            path_lookup[leaf] = path_dict

        for i, t in enumerate(clean_texts):
            # Check supervised first
            if supervised_preds is not None and supervised_preds[i] in path_lookup:
                prob = float(supervised_probs[i])
                if prob >= 0.45:
                    results.append((path_lookup[supervised_preds[i]], min(0.99, round(prob, 2))))
                    continue
            
            # Zero-shot prediction
            idx = text_to_idx[t]
            raw_score = text_to_score[t]
            # Calibrate zero-shot cosine to [0.30, 0.95]
            conf = min(0.95, max(0.30, round(float(raw_score), 2)))
            results.append((path_lookup[idx], conf))
            
        return results
