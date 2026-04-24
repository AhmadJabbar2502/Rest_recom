# 🍽️ RestaurantIQ — Streamlit Dashboard

A production-grade dashboard for the **Restaurant Recommendation Engine** built on SVD + KNN hybrid collaborative filtering + Pinecone semantic search.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | KPI cards, rating distribution, model performance, pipeline summary |
| 🔍 Get Recommendations | Existing user (hybrid) and new user (semantic) recommendation UI |
| 📊 Model Analytics | Error distributions, RMSE by bucket, weight breakdown, dataset split |
| 🏗️ Architecture | Hyperparams, inference routing table, technology stack |

## Integration with Notebook 5

Replace the `simulate_recommendations()` function in `app.py` with the real `recommend()` function from Notebook 5:

```python
from notebook_5_hybrid_inference import recommend

def get_recommendations(user_id, preference_text, city, state, top_k):
    return recommend(
        user_id=user_id or None,
        preference_text=preference_text or None,
        city=city,
        state=state,
        top_k=top_k
    )
```

Make sure `models/`, `data/` folders from Notebooks 3 & 4 are present.
