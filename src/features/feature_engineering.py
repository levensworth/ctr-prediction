"""Feature engineering for CTR prediction.

Implements feature transformations including:
- Historical CTR statistics (rolling window) for campaigns
- Publication audience statistics (rolling window)
- TF-IDF features from publication tags
- One-hot encoding for categorical features
- Temporal features (day of week, hour bucket, month)
- Publication cluster features (from embeddings)
"""

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer


# Publisher-specific feature columns (keyed by publication_id)
PUBLISHER_NUMERICAL_FEATURES = [
    "pub_avg_opens",
    "pub_std_opens",
    "pub_ctr_mean",
    "pub_placement_count",
]

PUBLISHER_CATEGORICAL_FEATURES: list[str] = []  # cluster_* columns are added dynamically

# Campaign-specific feature columns (keyed by campaign_id)
CAMPAIGN_NUMERICAL_FEATURES = [
    "campaign_ctr_mean",
    "campaign_ctr_std",
    "campaign_ctr_count",
    "campaign_weighted_ctr",
    "num_income_targets",
    "num_age_targets",
]

CAMPAIGN_CATEGORICAL_FEATURES = [
    "gender_no_pref",
    "gender_balanced",
    "gender_male",
    "gender_female",
    "gender_unknown",
    "item_product",
    "item_service",
    "item_newsletter",
    "item_knowledge",
    "item_event",
    "item_other",
    "item_unknown",
    "income_range_1",
    "income_range_2",
    "income_range_3",
    "income_range_4",
    "income_range_5",
]

# Temporal features (per-placement, not per entity)
TEMPORAL_FEATURES = [
    "month",
    "hour_morning",
    "hour_midday",
    "hour_night",
    "dow_mon",
    "dow_tue",
    "dow_wed",
    "dow_thu",
    "dow_fri",
    "dow_sat",
    "dow_sun",
]


@dataclass
class CTRFeatureEngineer:
    """Feature engineering for CTR prediction models.
    
    This class encapsulates all feature transformations needed for the
    XGBoost ensemble model, including stateful transformers like TF-IDF.
    """
    
    tfidf_max_features: int = 50
    tfidf_min_df: int = 5
    rolling_window_days: int = 90
    
    _tfidf_vectorizer: TfidfVectorizer | None = field(default=None, init=False)
    _campaign_features: pl.DataFrame | None = field(default=None, init=False)
    _campaign_ctr_stats: pl.DataFrame | None = field(default=None, init=False)
    _publication_audience: pl.DataFrame | None = field(default=None, init=False)
    _tfidf_features: pl.DataFrame | None = field(default=None, init=False)
    _cluster_features: pl.DataFrame | None = field(default=None, init=False)
    _feature_columns: list[str] = field(default_factory=list, init=False)
    _is_fitted: bool = field(default=False, init=False)

    def fit(
        self,
        placements_df: pl.DataFrame,
        campaigns_df: pl.DataFrame,
        tags_df: pl.DataFrame,
        clusters_df: pl.DataFrame | None = None,
    ) -> "CTRFeatureEngineer":
        """Fit the feature engineer on training data.
        
        Computes historical statistics and fits TF-IDF vectorizer.
        """
        placements_with_ctr = self._compute_ctr_and_datetime(placements_df)
        
        self._campaign_features = self._encode_campaign_features(campaigns_df)
        self._campaign_ctr_stats = self._compute_rolling_campaign_stats(
            placements_with_ctr, self.rolling_window_days
        )
        self._publication_audience = self._compute_rolling_publication_stats(
            placements_with_ctr, self.rolling_window_days
        )
        self._tfidf_features, self._tfidf_vectorizer = self._build_tfidf_features(
            tags_df, self.tfidf_max_features, self.tfidf_min_df
        )
        
        if clusters_df is not None:
            self._cluster_features = self._build_cluster_features(clusters_df)
        
        self._feature_columns = self._determine_feature_columns()
        self._is_fitted = True
        
        return self

    def transform(self, placements_df: pl.DataFrame) -> pl.DataFrame:
        """Transform placements data into feature matrix."""
        if not self._is_fitted:
            raise RuntimeError("Feature engineer must be fitted before transform")
        
        placements_with_ctr = self._compute_ctr_and_datetime(placements_df)
        return self._build_feature_matrix(placements_with_ctr)

    def fit_transform(
        self,
        placements_df: pl.DataFrame,
        campaigns_df: pl.DataFrame,
        tags_df: pl.DataFrame,
        clusters_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Fit and transform in one step."""
        self.fit(placements_df, campaigns_df, tags_df, clusters_df)
        return self.transform(placements_df)

    def get_feature_columns(self) -> list[str]:
        """Return the list of feature column names."""
        return self._feature_columns.copy()

    def get_publisher_feature_columns(self) -> tuple[list[str], list[str]]:
        """Return publisher feature columns split into numerical and categorical.
        
        Returns:
            Tuple of (numerical_columns, categorical_columns)
        """
        numerical = PUBLISHER_NUMERICAL_FEATURES.copy()
        categorical = PUBLISHER_CATEGORICAL_FEATURES.copy()
        
        # Add TF-IDF columns as numerical
        if self._tfidf_features is not None:
            tfidf_cols = [c for c in self._tfidf_features.columns if c.startswith("tfidf_")]
            numerical.extend(tfidf_cols)
        
        # Add cluster columns as categorical
        if self._cluster_features is not None:
            cluster_cols = [c for c in self._cluster_features.columns if c.startswith("cluster_")]
            categorical.extend(cluster_cols)
        
        return numerical, categorical

    def get_campaign_feature_columns(self) -> tuple[list[str], list[str]]:
        """Return campaign feature columns split into numerical and categorical.
        
        Returns:
            Tuple of (numerical_columns, categorical_columns)
        """
        return CAMPAIGN_NUMERICAL_FEATURES.copy(), CAMPAIGN_CATEGORICAL_FEATURES.copy()

    def get_all_publisher_feature_columns(self) -> list[str]:
        """Return all publisher feature column names."""
        numerical, categorical = self.get_publisher_feature_columns()
        return numerical + categorical

    def get_all_campaign_feature_columns(self) -> list[str]:
        """Return all campaign feature column names."""
        numerical, categorical = self.get_campaign_feature_columns()
        return numerical + categorical

    def save(self, path: Path) -> None:
        """Save fitted state to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted feature engineer")
        
        state = {
            "tfidf_max_features": self.tfidf_max_features,
            "tfidf_min_df": self.tfidf_min_df,
            "rolling_window_days": self.rolling_window_days,
            "tfidf_vectorizer": self._tfidf_vectorizer,
            "campaign_features": self._campaign_features,
            "campaign_ctr_stats": self._campaign_ctr_stats,
            "publication_audience": self._publication_audience,
            "tfidf_features": self._tfidf_features,
            "cluster_features": self._cluster_features,
            "feature_columns": self._feature_columns,
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> "CTRFeatureEngineer":
        """Load fitted state from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.tfidf_max_features = state["tfidf_max_features"]
        self.tfidf_min_df = state["tfidf_min_df"]
        self.rolling_window_days = state["rolling_window_days"]
        self._tfidf_vectorizer = state["tfidf_vectorizer"]
        self._campaign_features = state["campaign_features"]
        self._campaign_ctr_stats = state["campaign_ctr_stats"]
        self._publication_audience = state["publication_audience"]
        self._tfidf_features = state["tfidf_features"]
        self._cluster_features = state["cluster_features"]
        self._feature_columns = state["feature_columns"]
        self._is_fitted = True
        
        return self

    def _compute_ctr_and_datetime(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate CTR and parse datetime from placements DataFrame."""
        ctr_expr = pl.col("approved_clicks") / pl.col("approved_opens")
        return df.with_columns([
            pl.when(pl.col("approved_opens") > 0)
              .then(pl.min_horizontal(ctr_expr, pl.lit(1.0)))
              .otherwise(0)
              .alias("ctr"),
            pl.col("post_send_at").str.to_datetime().alias("send_datetime")
        ])

    def _compute_rolling_campaign_stats(
        self,
        placements_df: pl.DataFrame,
        window_days: int,
    ) -> pl.DataFrame:
        """Compute historical CTR statistics per campaign using rolling window."""
        max_train_date = placements_df["send_datetime"].max()
        window_start = max_train_date - pl.duration(days=window_days)
        
        recent_data = placements_df.filter(pl.col("send_datetime") >= window_start)
        
        return recent_data.group_by("campaign_id").agg([
            pl.col("ctr").mean().alias("campaign_ctr_mean"),
            pl.col("ctr").std().alias("campaign_ctr_std"),
            pl.col("ctr").count().alias("campaign_ctr_count"),
            pl.col("approved_clicks").sum().alias("campaign_total_clicks"),
            pl.col("approved_opens").sum().alias("campaign_total_opens"),
        ]).with_columns([
            (pl.col("campaign_total_clicks") / pl.col("campaign_total_opens")).alias("campaign_weighted_ctr"),
            pl.col("campaign_ctr_std").fill_null(0.0)
        ])

    def _compute_rolling_publication_stats(
        self,
        placements_df: pl.DataFrame,
        window_days: int,
    ) -> pl.DataFrame:
        """Compute historical audience metrics per publication using rolling window."""
        max_train_date = placements_df["send_datetime"].max()
        window_start = max_train_date - pl.duration(days=window_days)
        
        recent_data = placements_df.filter(pl.col("send_datetime") >= window_start)
        
        return recent_data.group_by("publication_id").agg([
            pl.col("approved_opens").mean().alias("pub_avg_opens"),
            pl.col("approved_opens").std().alias("pub_std_opens"),
            pl.col("approved_opens").sum().alias("pub_total_opens"),
            pl.col("ctr").mean().alias("pub_ctr_mean"),
            pl.count().alias("pub_placement_count"),
        ]).with_columns([
            pl.col("pub_std_opens").fill_null(0.0)
        ])

    def _encode_campaign_features(self, campaigns_df: pl.DataFrame) -> pl.DataFrame:
        """Create one-hot encoded features for campaign categorical variables."""
        return campaigns_df.select([
            "campaign_id",
            "advertiser_id",
            "target_gender",
            "promoted_item",
            "target_incomes",
            "target_ages"
        ]).with_columns([
            # One-hot for target_gender
            (pl.col("target_gender") == "no_preference").cast(pl.Int8).alias("gender_no_pref"),
            (pl.col("target_gender") == "balanced").cast(pl.Int8).alias("gender_balanced"),
            (pl.col("target_gender") == "predominantly_male").cast(pl.Int8).alias("gender_male"),
            (pl.col("target_gender") == "predominantly_female").cast(pl.Int8).alias("gender_female"),
            pl.col("target_gender").is_null().cast(pl.Int8).alias("gender_unknown"),
            
            # One-hot for promoted_item
            (pl.col("promoted_item") == "product").cast(pl.Int8).alias("item_product"),
            (pl.col("promoted_item") == "service").cast(pl.Int8).alias("item_service"),
            (pl.col("promoted_item") == "newsletter").cast(pl.Int8).alias("item_newsletter"),
            (pl.col("promoted_item") == "knowledge_product").cast(pl.Int8).alias("item_knowledge"),
            (pl.col("promoted_item") == "event").cast(pl.Int8).alias("item_event"),
            (pl.col("promoted_item") == "other").cast(pl.Int8).alias("item_other"),
            pl.col("promoted_item").is_null().cast(pl.Int8).alias("item_unknown"),
            
            # Target incomes
            pl.col("target_incomes").str.contains("range_1").fill_null(False).cast(pl.Int8).alias("income_range_1"),
            pl.col("target_incomes").str.contains("range_2").fill_null(False).cast(pl.Int8).alias("income_range_2"),
            pl.col("target_incomes").str.contains("range_3").fill_null(False).cast(pl.Int8).alias("income_range_3"),
            pl.col("target_incomes").str.contains("range_4").fill_null(False).cast(pl.Int8).alias("income_range_4"),
            pl.col("target_incomes").str.contains("range_5").fill_null(False).cast(pl.Int8).alias("income_range_5"),
            
            # Count of targeting options
            pl.col("target_incomes").str.count_matches(r"range_").fill_null(0).alias("num_income_targets"),
            pl.col("target_ages").str.count_matches(r"range_").fill_null(0).alias("num_age_targets"),
        ])

    def _build_tfidf_features(
        self,
        tags_df: pl.DataFrame,
        max_features: int,
        min_df: int,
    ) -> tuple[pl.DataFrame, TfidfVectorizer]:
        """Build TF-IDF features from publication tags."""
        cleaned = tags_df.with_columns([
            pl.col("tags")
            .str.replace_all(r"[\{\}'\"]", "")
            .str.replace_all(r",", " ")
            .str.to_lowercase()
            .alias("tags_cleaned")
        ]).filter(
            pl.col("tags_cleaned").is_not_null() & (pl.col("tags_cleaned") != "")
        )
        
        pub_ids = cleaned["publication_id"].to_list()
        texts = [t if t is not None else "" for t in cleaned["tags_cleaned"].to_list()]
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            stop_words='english'
        )
        matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        data: dict[str, Any] = {"publication_id": pub_ids}
        for idx, name in enumerate(feature_names):
            data[f"tfidf_{name}"] = matrix[:, idx].toarray().flatten().tolist()
        
        return pl.DataFrame(data), vectorizer

    def _build_cluster_features(self, clusters_df: pl.DataFrame) -> pl.DataFrame:
        """Build one-hot encoded cluster features from publication clusters."""
        unique_clusters = sorted(clusters_df["cluster"].unique().to_list())
        
        cluster_cols = []
        for cluster_id in unique_clusters:
            col_name = f"cluster_{cluster_id}"
            cluster_cols.append(
                (pl.col("cluster") == cluster_id).cast(pl.Int8).alias(col_name)
            )
        
        return clusters_df.select([
            "publication_id",
            "cluster",
        ]).with_columns(cluster_cols)

    def _add_temporal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add temporal features to placements DataFrame."""
        return df.with_columns([
            pl.col("send_datetime").dt.weekday().alias("day_of_week"),
            pl.col("send_datetime").dt.hour().alias("hour"),
            pl.col("send_datetime").dt.month().alias("month"),
        ]).with_columns([
            pl.when(pl.col("hour").is_between(6, 11))
            .then(pl.lit("morning"))
            .when(pl.col("hour").is_between(12, 17))
            .then(pl.lit("midday"))
            .otherwise(pl.lit("night"))
            .alias("hour_bucket"),
        ]).with_columns([
            (pl.col("hour_bucket") == "morning").cast(pl.Int8).alias("hour_morning"),
            (pl.col("hour_bucket") == "midday").cast(pl.Int8).alias("hour_midday"),
            (pl.col("hour_bucket") == "night").cast(pl.Int8).alias("hour_night"),
            
            (pl.col("day_of_week") == 0).cast(pl.Int8).alias("dow_mon"),
            (pl.col("day_of_week") == 1).cast(pl.Int8).alias("dow_tue"),
            (pl.col("day_of_week") == 2).cast(pl.Int8).alias("dow_wed"),
            (pl.col("day_of_week") == 3).cast(pl.Int8).alias("dow_thu"),
            (pl.col("day_of_week") == 4).cast(pl.Int8).alias("dow_fri"),
            (pl.col("day_of_week") == 5).cast(pl.Int8).alias("dow_sat"),
            (pl.col("day_of_week") == 6).cast(pl.Int8).alias("dow_sun"),
        ])

    def _build_feature_matrix(self, placements_df: pl.DataFrame) -> pl.DataFrame:
        """Build complete feature matrix by joining all features."""
        df = self._add_temporal_features(placements_df)
        
        # Join campaign features
        df = df.join(
            self._campaign_features.select([
                "campaign_id", "advertiser_id",
                "gender_no_pref", "gender_balanced", "gender_male", "gender_female", "gender_unknown",
                "item_product", "item_service", "item_newsletter", "item_knowledge",
                "item_event", "item_other", "item_unknown",
                "income_range_1", "income_range_2", "income_range_3", "income_range_4", "income_range_5",
                "num_income_targets", "num_age_targets"
            ]),
            on="campaign_id",
            how="left"
        )
        
        # Join campaign CTR stats
        df = df.join(
            self._campaign_ctr_stats.select([
                "campaign_id",
                "campaign_ctr_mean", "campaign_ctr_std", "campaign_ctr_count",
                "campaign_weighted_ctr"
            ]),
            on="campaign_id",
            how="left"
        )
        
        # Join publication audience stats
        df = df.join(
            self._publication_audience.select([
                "publication_id",
                "pub_avg_opens", "pub_std_opens", "pub_ctr_mean", "pub_placement_count"
            ]),
            on="publication_id",
            how="left"
        )
        
        # Join TF-IDF features
        df = df.join(self._tfidf_features, on="publication_id", how="left")
        
        # Join cluster features if available
        if self._cluster_features is not None:
            df = df.join(self._cluster_features, on="publication_id", how="left")
            for col in [c for c in df.columns if c.startswith("cluster_")]:
                df = df.with_columns(pl.col(col).fill_null(0))
            if "cluster" in df.columns:
                df = df.with_columns(pl.col("cluster").fill_null(-1))
        
        # Fill nulls for numeric columns
        numeric_fill_cols = [
            "campaign_ctr_mean", "campaign_ctr_std", "campaign_ctr_count", "campaign_weighted_ctr",
            "pub_avg_opens", "pub_std_opens", "pub_ctr_mean", "pub_placement_count"
        ]
        for col in numeric_fill_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(0.0))
        
        # Fill nulls for one-hot columns
        onehot_fill_cols = [
            "gender_no_pref", "gender_balanced", "gender_male", "gender_female", "gender_unknown",
            "item_product", "item_service", "item_newsletter", "item_knowledge",
            "item_event", "item_other", "item_unknown",
            "income_range_1", "income_range_2", "income_range_3", "income_range_4", "income_range_5",
            "hour_morning", "hour_midday", "hour_night",
            "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun"
        ]
        for col in onehot_fill_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(0))
        
        # Fill TF-IDF columns
        for col in [c for c in df.columns if c.startswith("tfidf_")]:
            df = df.with_columns(pl.col(col).fill_null(0.0))
        
        return df

    def _determine_feature_columns(self) -> list[str]:
        """Determine the list of feature columns for model training."""
        base_features = [
            "campaign_ctr_mean", "campaign_ctr_std", "campaign_ctr_count", "campaign_weighted_ctr",
            "pub_avg_opens", "pub_std_opens", "pub_ctr_mean", "pub_placement_count",
            "gender_no_pref", "gender_balanced", "gender_male", "gender_female", "gender_unknown",
            "item_product", "item_service", "item_newsletter", "item_knowledge",
            "item_event", "item_other", "item_unknown",
            "income_range_1", "income_range_2", "income_range_3", "income_range_4", "income_range_5",
            "num_income_targets", "num_age_targets",
            "month",
            "hour_morning", "hour_midday", "hour_night",
            "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun",
        ]
        
        # Add TF-IDF columns
        if self._tfidf_features is not None:
            tfidf_cols = [c for c in self._tfidf_features.columns if c.startswith("tfidf_")]
            base_features.extend(tfidf_cols)
        
        # Add cluster columns
        if self._cluster_features is not None:
            cluster_cols = sorted([c for c in self._cluster_features.columns if c.startswith("cluster_")])
            base_features.extend(cluster_cols)
        
        return base_features


def prepare_model_data(
    features_df: pl.DataFrame,
    feature_columns: list[str],
    target_col: str = "ctr",
    weight_col: str = "approved_opens",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare numpy arrays for model training/evaluation.
    
    Returns:
        X: Feature matrix
        y: Target values
        weights: Sample weights (normalized by mean)
        opens: Raw approved_opens values
    """
    X = features_df.select(feature_columns).to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    y = features_df.select(target_col).to_numpy().flatten()
    
    weights = features_df.select(weight_col).to_numpy().flatten().astype(np.float32)
    if weights.mean() > 0:
        weights = weights / weights.mean()
    
    opens = features_df.select(weight_col).to_numpy().flatten()
    
    return X, y, weights, opens


def split_by_audience(
    df: pl.DataFrame,
    threshold: int = 1000,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split DataFrame by audience size (approved_opens)."""
    small = df.filter(pl.col("approved_opens") < threshold)
    large = df.filter(pl.col("approved_opens") >= threshold)
    return small, large


def create_temporal_split(
    df: pl.DataFrame,
    test_days: int = 90,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split data chronologically: train on older data, test on recent."""
    max_date = df["send_datetime"].max()
    split_date = max_date - pl.duration(days=test_days)
    
    train_data = df.filter(pl.col("send_datetime") < split_date)
    test_data = df.filter(pl.col("send_datetime") >= split_date)
    
    return train_data, test_data
