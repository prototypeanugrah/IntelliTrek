from math import asin, cos, radians, sin, sqrt

import gspread
import numpy as np
import pandas as pd
from google.oauth2.service_account import Credentials
from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = OllamaEmbeddings(model="mxbai-embed-large:latest")


def get_feature_vector(
    feature: str, feature_vector_cache: dict[str, np.ndarray]
) -> np.ndarray:
    if feature not in feature_vector_cache:
        feature_vector_cache[feature] = np.array(embedding_model.embed_query(feature))
    return feature_vector_cache[feature]


def haversine_miles(lat1, lon1, lat2, lon2):
    # distance between two lat/lon pairs in miles
    R = 3958.8
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    lat1, lat2 = map(radians, (lat1, lat2))
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def create_google_client(service_account_file: str):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(service_account_file, scopes=scopes)
    client = gspread.authorize(creds)
    return client


def trail_matches(
    row: pd.Series,
    desired_activities: list[str],
    desired_feats: list[str],
    SIM_THRESHOLD: float,
    feature_vector_cache: dict[str, np.ndarray],
) -> bool:
    # Lower‑case lists of features and activities for this trail
    trail_feats = [f.lower() for f in row.features]
    trail_acts = [a.lower() for a in row.activities]

    # Embed every token once
    token_vectors = np.vstack(
        [get_feature_vector(t, feature_vector_cache) for t in trail_feats + trail_acts]
    )

    # Require *all* requested activities to be present (cosine ≥ threshold)
    for act in desired_activities:
        if (
            cosine_similarity(
                get_feature_vector(act, feature_vector_cache).reshape(1, -1),
                token_vectors,
            ).max()
            < SIM_THRESHOLD
        ):
            return False

    # Require *all* requested features to be present
    for feat in desired_feats:
        if (
            cosine_similarity(
                get_feature_vector(feat, feature_vector_cache).reshape(1, -1),
                token_vectors,
            ).max()
            < SIM_THRESHOLD
        ):
            return False

    return True
