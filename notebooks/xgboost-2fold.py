# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "polars>=1.37.1",
#     "matplotlib>=3.10.8",
#     "seaborn>=0.13.0",
#     "scikit-learn>=1.5.0",
#     "xgboost>=2.0.0",
#     "numpy>=1.24.0",
# ]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from datetime import datetime, timedelta
    from typing import Tuple
    from dataclasses import dataclass
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import mean_absolute_error, r2_score
    import xgboost as xgb

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    mo.md("""
    # XGBoost 2-Fold Ensemble CTR Prediction Model

    This notebook builds an **ensemble of 2 XGBoost models** to predict Click-Through Rate (CTR):
    - **Model 1**: For small audiences (opens < 1,000)
    - **Model 2**: For large audiences (opens >= 1,000)

    ## Features
    1. **Historical CTR Statistics** - mean, std, count for campaign_id (rolling 3-month window)
    2. **TF-IDF Vectors** - based on publication tags
    3. **Audience Size** - historical mean of previous 3 months opens by publication_id
    4. **Target Gender** - one-hot encoded
    5. **Promoted Item Type** - one-hot encoded
    6. **Temporal Features** - day of week, hour bucket (morning/mid-day/night), month
    7. **Target Audience** - one-hot encoded for target_gender and target_incomes
    8. **Publication Clusters** - one-hot encoded clusters from mean post embeddings (Qwen3-Embedding-0.6B)

    ## Training Strategy
    - **Historical train/test split**: Train on older data, test on last year of data
    - **2-Fold Ensemble**: Separate models for small vs large audiences

    ## Evaluation Metrics
    - **R² Score**: Coefficient of determination
    - **MAE**: Mean Absolute Error
    """)
    return (
        Path,
        TfidfVectorizer,
        Tuple,
        mean_absolute_error,
        mo,
        np,
        pl,
        plt,
        r2_score,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1. Data Loading
    """)
    return


@app.cell
def _(Path, Tuple, pl):
    def load_datasets(data_dir: Path) -> Tuple:
        """Load all required datasets including publication clusters."""
        placements = pl.read_csv(data_dir / "placements.csv")
        campaigns = pl.read_csv(data_dir / "campaigns.csv")
        publication_tags = pl.read_csv(data_dir / "publication_tags.csv")
        
        # Load publication clusters from clustering analysis
        clusters_path = data_dir / "publication_clusters.csv"
        if clusters_path.exists():
            publication_clusters = pl.read_csv(clusters_path)
            print(f"Publication clusters loaded: {len(publication_clusters):,} publications")
        else:
            publication_clusters = None
            print("Warning: publication_clusters.csv not found. Run publication_clustering.py first.")
        
        return placements, campaigns, publication_tags, publication_clusters

    DATA_DIR = Path("../data")
    raw_placements, raw_campaigns, raw_publication_tags, raw_publication_clusters = load_datasets(DATA_DIR)

    print("=== Data Shapes ===")
    print(f"Placements: {raw_placements.shape}")
    print(f"Campaigns: {raw_campaigns.shape}")
    print(f"Publication tags: {raw_publication_tags.shape}")
    if raw_publication_clusters is not None:
        print(f"Publication clusters: {raw_publication_clusters.shape}")
        print(f"Unique clusters: {raw_publication_clusters['cluster'].n_unique()}")
    return DATA_DIR, raw_campaigns, raw_placements, raw_publication_clusters, raw_publication_tags


@app.cell
def _(pl, raw_placements):
    def compute_ctr_and_datetime(df: pl.DataFrame) -> pl.DataFrame:
        """Calculate CTR and parse datetime from placements DataFrame."""
        ctr_expr = pl.col("approved_clicks") / pl.col("approved_opens")
        return df.with_columns([
            pl.when(pl.col("approved_opens") > 0)
              .then(pl.min_horizontal(ctr_expr, pl.lit(1.0)))
              .otherwise(0)
              .alias("ctr"),
            pl.col("post_send_at").str.to_datetime().alias("send_datetime")
        ])

    placements_with_ctr = compute_ctr_and_datetime(raw_placements)

    print(f"Placements with CTR: {len(placements_with_ctr):,}")
    print(f"Date range: {placements_with_ctr['send_datetime'].min()} to {placements_with_ctr['send_datetime'].max()}")
    return (placements_with_ctr,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. Historical Train/Test Split

    We split data chronologically:
    - **Training**: All data except the last year
    - **Testing**: Last year of data (to simulate real-world prediction)
    """)
    return


@app.cell
def _(Tuple, pl, placements_with_ctr):
    def create_temporal_split(df: pl.DataFrame) -> Tuple:
        """Split data: train on older data, test on last year."""
        max_date = df["send_datetime"].max()
        # Test on last 90 days of data
        split_date = max_date - pl.duration(days=90)

        train_data = df.filter(pl.col("send_datetime") < split_date)
        test_data = df.filter(pl.col("send_datetime") >= split_date)

        return train_data, test_data, split_date

    train_placements, test_placements, split_date = create_temporal_split(placements_with_ctr)

    print(f"Split date: {split_date}")
    print(f"Training data: before {split_date}")
    print(f"Test data: from {split_date} onwards")
    print(f"\nTrain samples: {len(train_placements):,}")
    print(f"Test samples: {len(test_placements):,}")
    return test_placements, train_placements


@app.cell
def _(mo):
    mo.md("""
    ## 3. Feature Engineering

    ### 3.1 Historical CTR Statistics for Campaigns (Rolling 3-Month Window)
    """)
    return


@app.cell
def _(pl, train_placements):
    def compute_rolling_campaign_stats(
        placements_df: pl.DataFrame,
        window_days: int = 90
    ) -> pl.DataFrame:
        """
        Compute historical CTR statistics per campaign using a rolling 3-month window.
        For simplicity, we compute the aggregate stats over the training period.
        """
        max_train_date = placements_df["send_datetime"].max()
        window_start = max_train_date - pl.duration(days=window_days)

        # Filter to last 3 months of training data
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

    campaign_ctr_stats = compute_rolling_campaign_stats(train_placements, window_days=90)
    print(f"Campaigns with 3-month rolling stats: {len(campaign_ctr_stats):,}")
    return (campaign_ctr_stats,)


@app.cell
def _(mo):
    mo.md("""
    ### 3.2 Publication Audience Size (Rolling 3-Month Window)
    """)
    return


@app.cell
def _(pl, train_placements):
    def compute_rolling_publication_stats(
        placements_df: pl.DataFrame,
        window_days: int = 90
    ) -> pl.DataFrame:
        """
        Compute historical audience size metrics per publication using a rolling 3-month window.
        """
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

    publication_audience = compute_rolling_publication_stats(train_placements, window_days=90)
    print(f"Publications with 3-month rolling stats: {len(publication_audience):,}")
    return (publication_audience,)


@app.cell
def _(mo):
    mo.md("""
    ### 3.3 Publication Cluster Features
    
    Use cluster assignments from the publication clustering analysis (based on mean post embeddings).
    Clusters are one-hot encoded to capture different publication segments.
    """)
    return


@app.cell
def _(pl, raw_publication_clusters):
    def build_cluster_features(clusters_df: pl.DataFrame | None) -> pl.DataFrame | None:
        """Build one-hot encoded cluster features from publication clusters."""
        if clusters_df is None:
            return None
        
        # Get unique cluster values
        unique_clusters = sorted(clusters_df["cluster"].unique().to_list())
        n_clusters = len(unique_clusters)
        print(f"Building one-hot encoding for {n_clusters} clusters")
        
        # Create one-hot encoded columns for each cluster
        cluster_cols = []
        for cluster_id in unique_clusters:
            col_name = f"cluster_{cluster_id}"
            cluster_cols.append(
                (pl.col("cluster") == cluster_id).cast(pl.Int8).alias(col_name)
            )
        
        # Select publication_id and add one-hot columns
        result = clusters_df.select([
            "publication_id",
            "cluster",  # Keep original cluster for potential ordinal use
        ]).with_columns(cluster_cols)
        
        return result

    cluster_features = build_cluster_features(raw_publication_clusters)
    
    if cluster_features is not None:
        print(f"Cluster features shape: {cluster_features.shape}")
        print(f"Cluster columns: {[c for c in cluster_features.columns if c.startswith('cluster_')]}")
        print(cluster_features.head(5))
    else:
        print("No cluster features available")
    return (cluster_features,)


@app.cell
def _(mo):
    mo.md("""
    ### 3.5 TF-IDF Features from Publication Tags
    """)
    return


@app.cell
def _(TfidfVectorizer, Tuple, pl, raw_publication_tags):
    def build_tfidf_features(
        tags_df: pl.DataFrame,
        max_features: int = 50,
        min_df: int = 5
    ) -> Tuple:
        """Build TF-IDF features from publication tags."""
        # Clean tags
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

        # Fit TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            stop_words='english'
        )
        matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # Build DataFrame
        data = {"publication_id": pub_ids}
        for idx, name in enumerate(feature_names):
            data[f"tfidf_{name}"] = matrix[:, idx].toarray().flatten().tolist()

        return pl.DataFrame(data), feature_names

    tfidf_features, tfidf_feature_names = build_tfidf_features(raw_publication_tags)
    print(f"TF-IDF features created: {len(tfidf_feature_names)}")
    print(f"Sample features: {list(tfidf_feature_names[:10])}")
    return (tfidf_features,)


@app.cell
def _(mo):
    mo.md("""
    ### 3.6 One-Hot Encoding for Categorical Features

    - Target Gender
    - Promoted Item Type
    - Target Incomes
    """)
    return


@app.cell
def _(pl, raw_campaigns):
    def encode_campaign_features(campaigns_df: pl.DataFrame) -> pl.DataFrame:
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
            # Handle null/empty as separate category
            pl.col("target_gender").is_null().cast(pl.Int8).alias("gender_unknown"),

            # One-hot for promoted_item
            (pl.col("promoted_item") == "product").cast(pl.Int8).alias("item_product"),
            (pl.col("promoted_item") == "service").cast(pl.Int8).alias("item_service"),
            (pl.col("promoted_item") == "newsletter").cast(pl.Int8).alias("item_newsletter"),
            (pl.col("promoted_item") == "knowledge_product").cast(pl.Int8).alias("item_knowledge"),
            (pl.col("promoted_item") == "event").cast(pl.Int8).alias("item_event"),
            (pl.col("promoted_item") == "other").cast(pl.Int8).alias("item_other"),
            pl.col("promoted_item").is_null().cast(pl.Int8).alias("item_unknown"),

            # Target incomes - parse income ranges
            pl.col("target_incomes").str.contains("range_1").fill_null(False).cast(pl.Int8).alias("income_range_1"),
            pl.col("target_incomes").str.contains("range_2").fill_null(False).cast(pl.Int8).alias("income_range_2"),
            pl.col("target_incomes").str.contains("range_3").fill_null(False).cast(pl.Int8).alias("income_range_3"),
            pl.col("target_incomes").str.contains("range_4").fill_null(False).cast(pl.Int8).alias("income_range_4"),
            pl.col("target_incomes").str.contains("range_5").fill_null(False).cast(pl.Int8).alias("income_range_5"),

            # Count of income/age ranges targeted
            pl.col("target_incomes").str.count_matches(r"range_").fill_null(0).alias("num_income_targets"),
            pl.col("target_ages").str.count_matches(r"range_").fill_null(0).alias("num_age_targets"),
        ])

    campaign_features = encode_campaign_features(raw_campaigns)

    print("=== Unique Values in Campaigns ===")
    print(f"Target Gender: {raw_campaigns['target_gender'].unique().to_list()}")
    print(f"Promoted Item: {raw_campaigns['promoted_item'].unique().to_list()}")
    return (campaign_features,)


@app.cell
def _(mo):
    mo.md("""
    ### 3.7 Temporal Features

    - Day of week (one-hot)
    - Hour bucket: morning (6-11), mid-day (12-17), night (18-5)
    - Month
    """)
    return


@app.cell
def _(pl):
    def add_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
        """Add temporal features to placements DataFrame."""
        return df.with_columns([
            pl.col("send_datetime").dt.weekday().alias("day_of_week"),
            pl.col("send_datetime").dt.hour().alias("hour"),
            pl.col("send_datetime").dt.month().alias("month"),
        ]).with_columns([
            # Hour bucket
            pl.when(pl.col("hour").is_between(6, 11))
            .then(pl.lit("morning"))
            .when(pl.col("hour").is_between(12, 17))
            .then(pl.lit("midday"))
            .otherwise(pl.lit("night"))
            .alias("hour_bucket"),
        ]).with_columns([
            # One-hot encode hour bucket
            (pl.col("hour_bucket") == "morning").cast(pl.Int8).alias("hour_morning"),
            (pl.col("hour_bucket") == "midday").cast(pl.Int8).alias("hour_midday"),
            (pl.col("hour_bucket") == "night").cast(pl.Int8).alias("hour_night"),

            # One-hot encode day of week
            (pl.col("day_of_week") == 0).cast(pl.Int8).alias("dow_mon"),
            (pl.col("day_of_week") == 1).cast(pl.Int8).alias("dow_tue"),
            (pl.col("day_of_week") == 2).cast(pl.Int8).alias("dow_wed"),
            (pl.col("day_of_week") == 3).cast(pl.Int8).alias("dow_thu"),
            (pl.col("day_of_week") == 4).cast(pl.Int8).alias("dow_fri"),
            (pl.col("day_of_week") == 5).cast(pl.Int8).alias("dow_sat"),
            (pl.col("day_of_week") == 6).cast(pl.Int8).alias("dow_sun"),
        ])
    return (add_temporal_features,)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Feature Assembly

    Combine all features into a single feature matrix.
    """)
    return


@app.cell
def _(
    add_temporal_features,
    campaign_ctr_stats,
    campaign_features,
    cluster_features,
    pl,
    publication_audience,
    tfidf_features,
):
    def build_feature_matrix(placements_df: pl.DataFrame) -> pl.DataFrame:
        """Build complete feature matrix by joining all features including publication clusters."""
        # Add temporal features
        df = add_temporal_features(placements_df)

        # Join campaign features
        df = df.join(
            campaign_features.select([
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

        # Join campaign CTR stats (rolling 3-month)
        df = df.join(
            campaign_ctr_stats.select([
                "campaign_id", 
                "campaign_ctr_mean", "campaign_ctr_std", "campaign_ctr_count",
                "campaign_weighted_ctr"
            ]),
            on="campaign_id",
            how="left"
        )

        # Join publication audience (rolling 3-month)
        df = df.join(
            publication_audience.select([
                "publication_id",
                "pub_avg_opens", "pub_std_opens", "pub_ctr_mean", "pub_placement_count"
            ]),
            on="publication_id",
            how="left"
        )

        # Join TF-IDF features
        df = df.join(tfidf_features, on="publication_id", how="left")

        # Join publication cluster features (if available)
        if cluster_features is not None:
            df = df.join(cluster_features, on="publication_id", how="left")
            # Fill nulls for cluster columns (publications without cluster assignment)
            for col in [c for c in df.columns if c.startswith("cluster_")]:
                df = df.with_columns(pl.col(col).fill_null(0))
            # Fill null for the raw cluster column with -1 (unknown cluster)
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
    return (build_feature_matrix,)


@app.cell
def _(build_feature_matrix, test_placements, train_placements):
    # Build feature matrices
    print("Building feature matrix for training data...")
    train_features = build_feature_matrix(train_placements)

    print("Building feature matrix for test data...")
    test_features = build_feature_matrix(test_placements)

    print(f"\nTrain feature matrix shape: {train_features.shape}")
    print(f"Test feature matrix shape: {test_features.shape}")
    return test_features, train_features


@app.cell
def _(mo):
    mo.md("""
    ## 5. Define Feature Columns for XGBoost
    """)
    return


@app.cell
def _(train_features):
    def get_feature_columns(df_columns: list) -> list:
        """Get the list of feature columns for model training including cluster features."""
        base_features = [
            # Campaign CTR stats (rolling 3-month)
            "campaign_ctr_mean", "campaign_ctr_std", "campaign_ctr_count", "campaign_weighted_ctr",

            # Publication stats (rolling 3-month audience size)
            "pub_avg_opens", "pub_std_opens", "pub_ctr_mean", "pub_placement_count",

            # Target gender one-hot
            "gender_no_pref", "gender_balanced", "gender_male", "gender_female", "gender_unknown",

            # Promoted item one-hot
            "item_product", "item_service", "item_newsletter", "item_knowledge", 
            "item_event", "item_other", "item_unknown",

            # Target incomes one-hot
            "income_range_1", "income_range_2", "income_range_3", "income_range_4", "income_range_5",

            # Targeting counts
            "num_income_targets", "num_age_targets",

            # Temporal features
            "month",
            "hour_morning", "hour_midday", "hour_night",
            "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun",
        ]

        # Add TF-IDF columns
        tfidf_cols = [c for c in df_columns if c.startswith("tfidf_")]
        
        # Add publication cluster one-hot columns
        cluster_cols = sorted([c for c in df_columns if c.startswith("cluster_")])

        # Filter to only existing columns
        all_features = base_features + tfidf_cols + cluster_cols
        return [c for c in all_features if c in df_columns]

    feature_cols = get_feature_columns(train_features.columns)
    
    # Print feature breakdown
    tfidf_count = len([c for c in feature_cols if c.startswith("tfidf_")])
    cluster_count = len([c for c in feature_cols if c.startswith("cluster_")])
    base_count = len(feature_cols) - tfidf_count - cluster_count
    
    print(f"Total features: {len(feature_cols)}")
    print(f"  - Base features: {base_count}")
    print(f"  - TF-IDF features: {tfidf_count}")
    print(f"  - Cluster features: {cluster_count}")
    print(f"\nCluster features: {[c for c in feature_cols if c.startswith('cluster_')]}")
    return (feature_cols,)


@app.cell
def _(mo):
    mo.md("""
    ## 6. Split Data by Audience Size for Ensemble

    We create two models:
    - **Small Audience Model**: For placements with `approved_opens < 1,000`
    - **Large Audience Model**: For placements with `approved_opens >= 1,000`
    """)
    return


@app.cell
def _(Tuple, pl):
    AUDIENCE_THRESHOLD = 1000  # Split point for ensemble

    def split_by_audience(df: pl.DataFrame, threshold: int = AUDIENCE_THRESHOLD) -> Tuple:
        """Split DataFrame by audience size."""
        small = df.filter(pl.col("approved_opens") < threshold)
        large = df.filter(pl.col("approved_opens") >= threshold)
        return small, large
    return AUDIENCE_THRESHOLD, split_by_audience


@app.cell
def _(split_by_audience, test_features, train_features):
    # Split training data
    train_small, train_large = split_by_audience(train_features)
    print(f"Training - Small audience (<1k): {len(train_small):,}")
    print(f"Training - Large audience (>=1k): {len(train_large):,}")

    # Split test data
    test_small, test_large = split_by_audience(test_features)
    print(f"\nTest - Small audience (<1k): {len(test_small):,}")
    print(f"Test - Large audience (>=1k): {len(test_large):,}")
    return test_large, test_small, train_large, train_small


@app.cell
def _(mo):
    mo.md("""
    ## 7. Prepare Data for XGBoost Training
    """)
    return


@app.cell
def _(
    Tuple,
    feature_cols,
    np,
    pl,
    test_large,
    test_small,
    train_large,
    train_small,
):
    def prepare_xgb_data(df: pl.DataFrame, feature_columns: list) -> Tuple:
        """Prepare numpy arrays for XGBoost from DataFrame."""
        X = df.select(feature_columns).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        y = df.select("ctr").to_numpy().flatten()

        # Sample weights based on approved_opens
        weights = df.select("approved_opens").to_numpy().flatten().astype(np.float32)
        weights = weights / weights.mean() if weights.mean() > 0 else weights

        opens = df.select("approved_opens").to_numpy().flatten()

        return X, y, weights, opens

    # Prepare small audience data
    X_train_small, y_train_small, w_train_small, _ = prepare_xgb_data(train_small, feature_cols)
    X_test_small, y_test_small, _, opens_test_small = prepare_xgb_data(test_small, feature_cols)

    # Prepare large audience data
    X_train_large, y_train_large, w_train_large, _ = prepare_xgb_data(train_large, feature_cols)
    X_test_large, y_test_large, _, opens_test_large = prepare_xgb_data(test_large, feature_cols)

    print("=== Small Audience Data ===")
    print(f"X_train_small: {X_train_small.shape}")
    print(f"X_test_small: {X_test_small.shape}")
    print(f"y_train mean CTR: {y_train_small.mean():.6f}")

    print("\n=== Large Audience Data ===")
    print(f"X_train_large: {X_train_large.shape}")
    print(f"X_test_large: {X_test_large.shape}")
    print(f"y_train mean CTR: {y_train_large.mean():.6f}")
    return (
        X_test_large,
        X_test_small,
        X_train_large,
        X_train_small,
        opens_test_large,
        opens_test_small,
        w_train_large,
        w_train_small,
        y_test_large,
        y_test_small,
        y_train_large,
        y_train_small,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 8. Train XGBoost Ensemble

    ### 8.1 Model for Small Audiences (< 1,000 opens)
    """)
    return


@app.cell
def _(X_train_small, w_train_small, y_train_small):
    import xgboost as xgb_train

    def train_xgb_model(X, y, weights, model_name: str):
        """Train an XGBoost model with given data."""
        dtrain = xgb_train.DMatrix(X, label=y, weight=weights)

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': ['mae', 'rmse'],
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'seed': 42,
            'n_jobs': -1,
        }

        print(f"\nTraining {model_name}...")
        model = xgb_train.train(
            params,
            dtrain,
            num_boost_round=200,
            verbose_eval=50
        )
        print(f"{model_name} training complete!")
        return model

    model_small = train_xgb_model(
        X_train_small, y_train_small, w_train_small,
        "Small Audience Model (<1k opens)"
    )
    return model_small, train_xgb_model


@app.cell
def _(mo):
    mo.md("""
    ### 8.2 Model for Large Audiences (>= 1,000 opens)
    """)
    return


@app.cell
def _(X_train_large, train_xgb_model, w_train_large, y_train_large):
    model_large = train_xgb_model(
        X_train_large, y_train_large, w_train_large,
        "Large Audience Model (>=1k opens)"
    )
    return (model_large,)


@app.cell
def _(mo):
    mo.md("""
    ## 9. Model Evaluation

    Evaluate both models using:
    - **R² Score**: Coefficient of determination
    - **MAE**: Mean Absolute Error
    """)
    return


@app.cell
def _(mean_absolute_error, np, r2_score):
    import xgboost as xgb_eval

    def evaluate_model(
        model,
        X_test,
        y_test,
        model_name: str
    ):
        """Evaluate a model and return predictions with metrics."""
        dtest = xgb_eval.DMatrix(X_test)
        y_pred = model.predict(dtest)
        y_pred = np.clip(y_pred, 0.0, 1.0)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n=== {model_name} ===")
        print(f"R² Score: {r2:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"Predictions range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")

        return y_pred, mae, r2
    return (evaluate_model,)


@app.cell
def _(
    X_test_large,
    X_test_small,
    evaluate_model,
    model_large,
    model_small,
    y_test_large,
    y_test_small,
):
    # Evaluate small audience model
    pred_small, mae_small, r2_small = evaluate_model(
        model_small, X_test_small, y_test_small,
        "Small Audience Model (<1k opens)"
    )

    # Evaluate large audience model
    pred_large, mae_large, r2_large = evaluate_model(
        model_large, X_test_large, y_test_large,
        "Large Audience Model (>=1k opens)"
    )
    return mae_large, mae_small, pred_large, pred_small, r2_large, r2_small


@app.cell
def _(
    Tuple,
    mae_large,
    mae_small,
    mean_absolute_error,
    np,
    pred_large,
    pred_small,
    r2_large,
    r2_score,
    r2_small,
    y_test_large,
    y_test_small,
):
    def compute_ensemble_metrics(
        y_test_s: np.ndarray, pred_s: np.ndarray,
        y_test_l: np.ndarray, pred_l: np.ndarray
    ) -> Tuple:
        """Compute combined ensemble metrics."""
        # Combine predictions and actuals
        y_combined = np.concatenate([y_test_s, y_test_l])
        pred_combined = np.concatenate([pred_s, pred_l])

        overall_mae = mean_absolute_error(y_combined, pred_combined)
        overall_r2 = r2_score(y_combined, pred_combined)

        return overall_mae, overall_r2, y_combined, pred_combined

    overall_mae, overall_r2, y_test_combined, pred_combined = compute_ensemble_metrics(
        y_test_small, pred_small, y_test_large, pred_large
    )

    print("\n" + "=" * 70)
    print("ENSEMBLE MODEL EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n{'Model':<35} {'R² Score':>15} {'MAE':>15}")
    print("-" * 70)
    print(f"{'Small Audience (<1k opens)':<35} {r2_small:>15.6f} {mae_small:>15.6f}")
    print(f"{'Large Audience (>=1k opens)':<35} {r2_large:>15.6f} {mae_large:>15.6f}")
    print("-" * 70)
    print(f"{'ENSEMBLE (Combined)':<35} {overall_r2:>15.6f} {overall_mae:>15.6f}")
    print("=" * 70)
    return overall_mae, overall_r2, pred_combined, y_test_combined


@app.cell
def _(mo):
    mo.md("""
    ## 10. Baseline Comparison
    """)
    return


@app.cell
def _(
    Tuple,
    mean_absolute_error,
    np,
    overall_mae,
    overall_r2,
    r2_score,
    y_test_combined,
    y_train_large,
    y_train_small,
):
    def compute_baseline_metrics(y_test: np.ndarray, y_train_s: np.ndarray, y_train_l: np.ndarray) -> Tuple:
        """Compute baseline metrics (predict mean)."""
        # Combined training mean
        train_mean = np.concatenate([y_train_s, y_train_l]).mean()
        baseline_pred = np.full_like(y_test, train_mean)

        baseline_mae = mean_absolute_error(y_test, baseline_pred)
        baseline_r2 = r2_score(y_test, baseline_pred)

        return baseline_mae, baseline_r2, train_mean

    baseline_mae, baseline_r2, train_mean_ctr = compute_baseline_metrics(
        y_test_combined, y_train_small, y_train_large
    )

    print("=== Baseline Comparison (Predict Mean) ===")
    print(f"Training mean CTR: {train_mean_ctr:.6f}")
    print(f"Baseline MAE: {baseline_mae:.6f}")
    print(f"Baseline R²: {baseline_r2:.6f}")
    print(f"\nEnsemble improvement over baseline:")
    print(f"  MAE improvement: {(baseline_mae - overall_mae) / baseline_mae * 100:.2f}%")
    print(f"  R² improvement: {overall_r2 - baseline_r2:.6f}")
    return (baseline_mae,)


@app.cell
def _(mo):
    mo.md("""
    ## 11. Feature Importance Analysis
    """)
    return


@app.cell
def _(feature_cols, model_large, model_small, np, plt):
    def plot_feature_importance(model, feature_names: list, title: str, ax) -> list:
        """Plot feature importance for a model."""
        importance_dict = model.get_score(importance_type='gain')

        feature_importance = []
        for idx, name in enumerate(feature_names):
            feat_key = f'f{idx}'
            imp = importance_dict.get(feat_key, 0)
            feature_importance.append((name, imp))

        feature_importance.sort(key=lambda x: x[1], reverse=True)
        top_n = min(15, len(feature_importance))
        top_features = feature_importance[:top_n]

        names = [f[0] for f in top_features]
        values = [f[1] for f in top_features]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color='steelblue', edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Feature Importance (Gain)')
        ax.set_title(title)
        ax.invert_yaxis()

        return feature_importance

    fig_imp, axes_imp = plt.subplots(1, 2, figsize=(16, 8))

    importance_small = plot_feature_importance(
        model_small, feature_cols,
        "Small Audience Model (<1k opens)",
        axes_imp[0]
    )

    importance_large = plot_feature_importance(
        model_large, feature_cols,
        "Large Audience Model (>=1k opens)",
        axes_imp[1]
    )

    plt.tight_layout()
    plt.show()
    return importance_large, importance_small


@app.cell
def _(mo):
    mo.md("""
    ## 12. Prediction Analysis
    """)
    return


@app.cell
def _(np, plt, pred_combined, y_test_combined):
    def plot_prediction_analysis(y_test: np.ndarray, y_pred: np.ndarray):
        """Plot prediction analysis charts."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Scatter plot: Predicted vs Actual
        ax1 = axes[0, 0]
        sample_size = min(5000, len(y_test))
        sample_idx = np.random.choice(len(y_test), sample_size, replace=False)
        ax1.scatter(y_test[sample_idx], y_pred[sample_idx], alpha=0.3, s=5)
        ax1.plot([0, 0.2], [0, 0.2], 'r--', linewidth=2, label='Perfect prediction')
        ax1.set_xlabel('Actual CTR')
        ax1.set_ylabel('Predicted CTR')
        ax1.set_title('Predicted vs Actual CTR (Sample)')
        ax1.set_xlim(0, 0.2)
        ax1.set_ylim(0, 0.2)
        ax1.legend()

        # 2. Residual distribution
        ax2 = axes[0, 1]
        residuals = y_pred - y_test
        ax2.hist(residuals, bins=100, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residual (Predicted - Actual)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Residual Distribution\nMean: {np.mean(residuals):.6f}, Std: {np.std(residuals):.6f}')

        # 3. Prediction distribution comparison
        ax3 = axes[1, 0]
        ax3.hist(y_pred, bins=100, alpha=0.7, label='Predicted', edgecolor='black')
        ax3.hist(y_test, bins=100, alpha=0.5, label='Actual', edgecolor='black')
        ax3.set_xlabel('CTR')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Predicted vs Actual CTR')
        ax3.legend()
        ax3.set_xlim(0, 0.15)

        # 4. Error by actual CTR bin
        ax4 = axes[1, 1]
        ctr_bins = np.linspace(0, 0.1, 11)
        bin_errors = []
        bin_centers = []

        for i in range(len(ctr_bins) - 1):
            mask = (y_test >= ctr_bins[i]) & (y_test < ctr_bins[i + 1])
            if np.sum(mask) > 0:
                bin_mae = np.mean(np.abs(y_pred[mask] - y_test[mask]))
                bin_errors.append(bin_mae)
                bin_centers.append((ctr_bins[i] + ctr_bins[i + 1]) / 2)

        ax4.bar(bin_centers, bin_errors, width=0.008, edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Actual CTR Bin')
        ax4.set_ylabel('MAE')
        ax4.set_title('MAE by Actual CTR Range')

        plt.tight_layout()
        plt.show()

    plot_prediction_analysis(y_test_combined, pred_combined)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 13. MAE by Opens Bins Analysis
    """)
    return


@app.cell
def _(
    np,
    opens_test_large,
    opens_test_small,
    pred_large,
    pred_small,
    y_test_large,
    y_test_small,
):
    def analyze_mae_by_opens_bins(
        y_small: np.ndarray, pred_s: np.ndarray, opens_s: np.ndarray,
        y_large: np.ndarray, pred_l: np.ndarray, opens_l: np.ndarray
    ) -> list:
        """Analyze MAE across different opens bins."""
        # Combine all data
        y_all = np.concatenate([y_small, y_large])
        pred_all = np.concatenate([pred_s, pred_l])
        opens_all = np.concatenate([opens_s, opens_l])

        bins_def = [
            (0, 100, "0-100"),
            (101, 500, "101-500"),
            (501, 1000, "501-1,000"),
            (1001, 10000, "1,001-10,000"),
            (10001, 100000, "10,001-100,000"),
            (100001, float('inf'), "100,001+"),
        ]

        results = []
        for bin_min, bin_max, label in bins_def:
            if bin_max == float('inf'):
                mask = opens_all >= bin_min
            else:
                mask = (opens_all >= bin_min) & (opens_all <= bin_max)

            count = np.sum(mask)
            if count > 0:
                errors = np.abs(y_all[mask] - pred_all[mask])
                mae = np.mean(errors)
                mean_ctr = np.mean(y_all[mask])
                mean_pred = np.mean(pred_all[mask])
            else:
                mae = mean_ctr = mean_pred = 0.0

            results.append({
                "label": label,
                "count": int(count),
                "mae": mae,
                "mean_ctr": mean_ctr,
                "mean_pred": mean_pred
            })

        return results

    mae_by_bins = analyze_mae_by_opens_bins(
        y_test_small, pred_small, opens_test_small,
        y_test_large, pred_large, opens_test_large
    )

    print("\n=== MAE by Opens Bins ===")
    print(f"{'Opens Bin':<20} {'Count':>12} {'MAE':>12} {'Mean CTR':>12} {'Mean Pred':>12}")
    print("-" * 70)
    for row in mae_by_bins:
        print(f"{row['label']:<20} {row['count']:>12,} {row['mae']:>12.6f} {row['mean_ctr']:>12.6f} {row['mean_pred']:>12.6f}")
    return (mae_by_bins,)


@app.cell
def _(mae_by_bins, np, plt):
    def plot_mae_by_bins(bins_data: list):
        """Plot MAE by opens bins."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        labels = [r["label"] for r in bins_data]
        maes = [r["mae"] for r in bins_data]
        counts = [r["count"] for r in bins_data]

        x_pos = np.arange(len(labels))

        # Plot 1: MAE by bin
        ax1 = axes[0]
        bars1 = ax1.bar(x_pos, maes, color='steelblue', edgecolor='black', alpha=0.8)
        ax1.set_xlabel('Opens Bin', fontsize=12)
        ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
        ax1.set_title('MAE by Approved Opens Bin', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=45, ha='right')

        for bar, mae in zip(bars1, maes):
            height = bar.get_height()
            ax1.annotate(f'{mae:.5f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Sample count by bin
        ax2 = axes[1]
        bars2 = ax2.bar(x_pos, counts, color='coral', edgecolor='black', alpha=0.8)
        ax2.set_xlabel('Opens Bin', fontsize=12)
        ax2.set_ylabel('Sample Count', fontsize=12)
        ax2.set_title('Sample Distribution by Opens Bin', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right')

        for bar, count in zip(bars2, counts):
            height = bar.get_height()
            ax2.annotate(f'{count:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()

    plot_mae_by_bins(mae_by_bins)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 14. Summary and Conclusions
    """)
    return


@app.cell
def _(
    AUDIENCE_THRESHOLD,
    baseline_mae,
    importance_large,
    importance_small,
    mae_large,
    mae_small,
    mo,
    overall_mae,
    overall_r2,
    r2_large,
    r2_small,
):
    def format_summary() -> str:
        """Format the model summary."""
        top_small = [f[0] for f in importance_small[:5]]
        top_large = [f[0] for f in importance_large[:5]]

        improvement_pct = (baseline_mae - overall_mae) / baseline_mae * 100

        return f"""
    ## Model Performance Summary

    ### Ensemble Architecture
    - **Threshold**: {AUDIENCE_THRESHOLD:,} opens
    - **Small Audience Model**: For placements with < {AUDIENCE_THRESHOLD:,} opens
    - **Large Audience Model**: For placements with >= {AUDIENCE_THRESHOLD:,} opens

    ### Evaluation Metrics

    | Model | R² Score | MAE |
    |-------|----------|-----|
    | Small Audience (<1k opens) | {r2_small:.6f} | {mae_small:.6f} |
    | Large Audience (>=1k opens) | {r2_large:.6f} | {mae_large:.6f} |
    | **ENSEMBLE (Combined)** | **{overall_r2:.6f}** | **{overall_mae:.6f}** |

    ### Baseline Comparison
    - Baseline MAE (predict mean): {baseline_mae:.6f}
    - **Improvement over baseline: {improvement_pct:.2f}%**

    ### Top 5 Features by Model

    **Small Audience Model:**
    1. {top_small[0]}
    2. {top_small[1]}
    3. {top_small[2]}
    4. {top_small[3]}
    5. {top_small[4]}

    **Large Audience Model:**
    1. {top_large[0]}
    2. {top_large[1]}
    3. {top_large[2]}
    4. {top_large[3]}
    5. {top_large[4]}

    ### Key Observations

    1. **Two-Model Ensemble**: Using separate models for small vs large audiences captures different CTR dynamics
    2. **Historical Features**: Rolling 3-month window for campaign and publication statistics
    3. **Rich Feature Set**: Includes TF-IDF content features, temporal features, targeting features, and publication clusters
    4. **Publication Clusters**: Embedding-based clusters capture content similarity patterns across publications
    5. **Audience-Specific Patterns**: Different features may be important for different audience sizes
    """

    mo.md(format_summary())
    return


@app.cell
def _(DATA_DIR, model_large, model_small):
    # Save models
    def save_models(data_dir, m_small, m_large):
        """Save both ensemble models."""
        m_small.save_model(str(data_dir / "xgboost_ctr_small_audience.json"))
        m_large.save_model(str(data_dir / "xgboost_ctr_large_audience.json"))
        print(f"Models saved to: {data_dir}")
        print("  - xgboost_ctr_small_audience.json")
        print("  - xgboost_ctr_large_audience.json")

    save_models(DATA_DIR, model_small, model_large)
    return


if __name__ == "__main__":
    app.run()
