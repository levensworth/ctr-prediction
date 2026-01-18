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
    from typing import Dict, List, Optional, Tuple
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import mean_absolute_error
    import xgboost as xgb

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    mo.md("""
    # CTR Ensemble Model with Smoothed Targets

    This notebook creates a model to predict CTR for each (newsletter, campaign) tuple.

    ## Features
    - **Historical CTR Statistics** (last 3 months):
      - For `campaign_id`: mean, std, count
      - For `publication_id`: mean, std, count
    - **Publication Tags**: TF-IDF vectors from `publication_tags.csv`
    - **Audience Size**: Historical mean of previous 3 months opens by `publication_id`
    - **Target Gender**: One-hot encoded
    - **Promoted Item Type**: One-hot encoded
    - **Target Audience**: One-hot encoded for `target_gender` and `target_incomes`

    ## Sample Weighting
    Each training record is weighted by its `approved_opens` value:
    - Records with more opens have more influence on the model
    - This improves accuracy on high-traffic records where precision matters most
    - The XGBoost `weight` parameter is used to apply these sample weights

    ## Ensemble Strategy
    - **Model 1**: For audience size < 1,000 opens
    - **Model 2**: For audience size >= 1,000 opens

    ## Target Smoothing
    Use smoothed CTR instead of raw CTR:
    ```
    ctr_smooth = (clicks + m*p0) / (opens + m)
    ```
    where:
    - `p0` = global prior CTR
    - `m` = prior strength (50-200 opens)

    ## Prediction Strategy
    Use historical data as prior:
    ```
    predicted_ctr = prior_ctr + r
    ```
    where `r` is the residual prediction from the model.

    ## Data Split
    - Train on historical data
    - Test on last 90 days

    ## Evaluation
    - **MAE** (Mean Absolute Error)
    """)
    return (
        Dict,
        List,
        Optional,
        Path,
        TfidfVectorizer,
        Tuple,
        datetime,
        mean_absolute_error,
        mo,
        np,
        pl,
        plt,
        timedelta,
        xgb,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1. Data Loading
    """)
    return


@app.cell
def _(Path, Tuple, pl):
    def load_datasets(data_dir: Path) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Load all required datasets."""
        placements_data = pl.read_csv(data_dir / "placements.csv")
        campaigns_data = pl.read_csv(data_dir / "campaigns.csv")
        publication_tags_data = pl.read_csv(data_dir / "publication_tags.csv")
        return placements_data, campaigns_data, publication_tags_data

    DATA_DIR = Path("../data")
    raw_placements, raw_campaigns, raw_publication_tags = load_datasets(DATA_DIR)

    print("=== Data Shapes ===")
    print(f"Placements: {raw_placements.shape}")
    print(f"Campaigns: {raw_campaigns.shape}")
    print(f"Publication tags: {raw_publication_tags.shape}")

    raw_placements.head(5)
    return raw_campaigns, raw_placements, raw_publication_tags


@app.cell
def _(pl, raw_placements):
    def parse_datetime_and_calculate_ctr(placements_data: pl.DataFrame) -> pl.DataFrame:
        """Parse datetime and calculate raw CTR."""
        ctr_raw_expr = pl.col("approved_clicks") / pl.col("approved_opens")

        placements_with_ctr_data = placements_data.with_columns([
            pl.when(pl.col("approved_opens") > 0)
              .then(pl.min_horizontal(ctr_raw_expr, pl.lit(1.0)))
              .otherwise(0)
              .alias("ctr_raw"),
            pl.col("post_send_at").str.to_datetime().alias("send_datetime")
        ]).filter(pl.col("approved_opens") > 0)

        return placements_with_ctr_data

    parsed_placements = parse_datetime_and_calculate_ctr(raw_placements)

    print(f"Placements with CTR: {len(parsed_placements):,}")
    print(f"Date range: {parsed_placements['send_datetime'].min()} to {parsed_placements['send_datetime'].max()}")

    parsed_placements.head(5)
    return (parsed_placements,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. Target Smoothing

    Apply CTR smoothing using the formula:
    ```
    ctr_smooth = (clicks + m*p0) / (opens + m)
    ```
    where `p0` is the global prior CTR and `m` is the prior strength.
    """)
    return


@app.cell
def _(Tuple, parsed_placements, pl):
    def calculate_smoothed_ctr(
        placements_data: pl.DataFrame, 
        prior_strength: int = 100
    ) -> Tuple[pl.DataFrame, float, int]:
        """
        Calculate smoothed CTR using the formula: (clicks + m*p0) / (opens + m).
        Returns the dataframe with smoothed CTR, global prior (p0), and prior strength (m).
        """
        total_clicks_val = placements_data["approved_clicks"].sum()
        total_opens_val = placements_data["approved_opens"].sum()
        global_prior_p0 = total_clicks_val / total_opens_val if total_opens_val > 0 else 0.0

        placements_with_smoothed_data = placements_data.with_columns([
            ((pl.col("approved_clicks") + prior_strength * global_prior_p0) / 
             (pl.col("approved_opens") + prior_strength)).alias("ctr_smooth")
        ])

        return placements_with_smoothed_data, global_prior_p0, prior_strength

    PRIOR_STRENGTH_M = 25
    smoothed_placements, global_prior_p0, m_value = calculate_smoothed_ctr(parsed_placements, PRIOR_STRENGTH_M)

    print(f"Global prior CTR (p0): {global_prior_p0:.6f}")
    print(f"Prior strength (m): {m_value}")

    print(f"\nRaw CTR stats:")
    print(f"  Mean: {parsed_placements['ctr_raw'].mean():.6f}")
    print(f"  Std: {parsed_placements['ctr_raw'].std():.6f}")

    print(f"\nSmoothed CTR stats:")
    print(f"  Mean: {smoothed_placements['ctr_smooth'].mean():.6f}")
    print(f"  Std: {smoothed_placements['ctr_smooth'].std():.6f}")

    smoothed_placements.head(5)
    return (smoothed_placements,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Historical Train/Test Split

    Split data chronologically:
    - Train on historical data (before last 90 days)
    - Test on last 90 days
    """)
    return


@app.cell
def _(Tuple, datetime, pl, smoothed_placements, timedelta):
    def split_train_test_historical(
        placements_data: pl.DataFrame, 
        test_days: int = 90
    ) -> Tuple[pl.DataFrame, pl.DataFrame, datetime]:
        """
        Split data chronologically into train and test sets.
        Test set is the last N days.
        """
        max_date_val = placements_data["send_datetime"].max()
        test_start_date_val = max_date_val - timedelta(days=test_days)

        train_mask_expr = pl.col("send_datetime") < test_start_date_val
        test_mask_expr = pl.col("send_datetime") >= test_start_date_val

        train_split = placements_data.filter(train_mask_expr)
        test_split = placements_data.filter(test_mask_expr)

        return train_split, test_split, test_start_date_val

    train_split_data, test_split_data, test_start_date_value = split_train_test_historical(
        smoothed_placements, test_days=90
    )

    print(f"Maximum date: {smoothed_placements['send_datetime'].max()}")
    print(f"Test start date: {test_start_date_value}")
    print(f"Test period: last 90 days")

    print(f"\nTrain samples: {len(train_split_data):,}")
    print(f"Test samples: {len(test_split_data):,}")
    print(f"Train date range: {train_split_data['send_datetime'].min()} to {train_split_data['send_datetime'].max()}")
    print(f"Test date range: {test_split_data['send_datetime'].min()} to {test_split_data['send_datetime'].max()}")

    test_split_data.head(5)
    return test_split_data, train_split_data


@app.cell
def _(mo):
    mo.md("""
    ## 4. Feature Engineering

    ### 4.1 Historical CTR Statistics (last 90 days)

    Memory-efficient approach: compute aggregate stats once from the last 90 days
    of training data, instead of computing rolling stats for every unique date.
    """)
    return


@app.cell
def _(pl, timedelta, train_split_data):
    def compute_historical_ctr_statistics(
        df: pl.DataFrame, 
        group_col: str, 
        date_col: str, 
        ctr_col: str,
        window_days: int = 90
    ) -> pl.DataFrame:
        """
        Calculate historical CTR statistics for each group using the last N days.

        Memory-efficient: computes stats once from the filtered window,
        instead of iterating over every unique date.
        """
        # Get max date and compute window start
        max_date_val = df[date_col].max()
        window_start_date = max_date_val - timedelta(days=window_days)

        # Filter to window period only
        window_data = df.filter(pl.col(date_col) > window_start_date)

        # Compute stats per group (single aggregation, memory-efficient)
        group_stats_df = window_data.group_by(group_col).agg([
            pl.col(ctr_col).mean().alias(f"{group_col}_ctr_mean"),
            pl.col(ctr_col).std().fill_null(0.0).alias(f"{group_col}_ctr_std"),
            pl.col(ctr_col).count().alias(f"{group_col}_ctr_count"),
        ])

        return group_stats_df

    # Calculate campaign CTR stats from last 90 days of training data
    print("Calculating historical CTR statistics for campaigns...")
    campaign_historical_stats = compute_historical_ctr_statistics(
        train_split_data, 
        "campaign_id", 
        "send_datetime", 
        "ctr_smooth",
        window_days=90
    )

    print(f"Campaign historical stats shape: {campaign_historical_stats.shape}")
    campaign_historical_stats.head(10)
    return campaign_historical_stats, compute_historical_ctr_statistics


@app.cell
def _(compute_historical_ctr_statistics, train_split_data):
    # Calculate publication CTR stats from last 90 days of training data
    print("Calculating historical CTR statistics for publications...")
    publication_historical_stats = compute_historical_ctr_statistics(
        train_split_data,
        "publication_id",
        "send_datetime",
        "ctr_smooth",
        window_days=90
    )

    print(f"Publication historical stats shape: {publication_historical_stats.shape}")
    publication_historical_stats.head(10)
    return (publication_historical_stats,)


@app.cell
def _(pl, timedelta, train_split_data):
    def compute_historical_audience_size(
        df: pl.DataFrame,
        group_col: str,
        date_col: str,
        opens_col: str,
        window_days: int = 90
    ) -> pl.DataFrame:
        """
        Calculate historical mean opens (audience size) for each group.

        Memory-efficient: computes stats once from the filtered window.
        """
        # Get max date and compute window start
        max_date_val = df[date_col].max()
        window_start_date = max_date_val - timedelta(days=window_days)

        # Filter to window period only
        window_data = df.filter(pl.col(date_col) > window_start_date)

        # Compute stats per group (single aggregation)
        group_stats_df = window_data.group_by(group_col).agg([
            pl.col(opens_col).mean().alias(f"{group_col}_avg_opens"),
        ])

        return group_stats_df

    print("Calculating historical audience size for publications...")
    publication_audience_stats = compute_historical_audience_size(
        train_split_data,
        "publication_id",
        "send_datetime",
        "approved_opens",
        window_days=90
    )

    print(f"Publication audience stats shape: {publication_audience_stats.shape}")
    publication_audience_stats.head(10)
    return (publication_audience_stats,)


@app.cell
def _(mo):
    mo.md("""
    ### 4.2 TF-IDF Features from Publication Tags
    """)
    return


@app.cell
def _(TfidfVectorizer, Tuple, pl, raw_publication_tags):
    def create_tfidf_features_from_tags(
        publication_tags_data: pl.DataFrame,
        max_features: int = 50,
        min_df: int = 5
    ) -> Tuple[pl.DataFrame, TfidfVectorizer]:
        """Create TF-IDF features from publication tags."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Clean and prepare tags for TF-IDF
        tags_cleaned_data = publication_tags_data.with_columns([
            pl.col("tags")
            .str.replace_all(r"[\{\}'\"]", "")
            .str.replace_all(r",", " ")
            .str.to_lowercase()
            .alias("tags_cleaned")
        ]).filter(
            pl.col("tags_cleaned").is_not_null() & (pl.col("tags_cleaned") != "")
        )

        # Get publication IDs and tags as lists
        publication_ids_list = tags_cleaned_data["publication_id"].to_list()
        tags_texts_list = tags_cleaned_data["tags_cleaned"].to_list()
        tags_texts_list = [t if t is not None else "" for t in tags_texts_list]

        # Create TF-IDF vectorizer
        vectorizer_instance = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            stop_words='english'
        )

        # Fit and transform
        tfidf_matrix_result = vectorizer_instance.fit_transform(tags_texts_list)
        tfidf_feature_names_list = vectorizer_instance.get_feature_names_out()

        # Create DataFrame with TF-IDF features
        tfidf_data_dict = {"publication_id": publication_ids_list}

        for idx, feat_name in enumerate(tfidf_feature_names_list):
            tfidf_data_dict[f"tfidf_{feat_name}"] = tfidf_matrix_result[:, idx].toarray().flatten().tolist()

        tfidf_result_df = pl.DataFrame(tfidf_data_dict)

        return tfidf_result_df, vectorizer_instance

    tfidf_features_df, tfidf_vectorizer_obj = create_tfidf_features_from_tags(raw_publication_tags)

    print(f"TF-IDF features: {len(tfidf_vectorizer_obj.get_feature_names_out())}")
    print(f"Feature names: {tfidf_vectorizer_obj.get_feature_names_out()[:20]}...")
    print(f"\nTF-IDF DataFrame shape: {tfidf_features_df.shape}")
    tfidf_features_df.head(5)
    return (tfidf_features_df,)


@app.cell
def _(mo):
    mo.md("""
    ### 4.3 One-Hot Encoding for Categorical Features
    """)
    return


@app.cell
def _(pl, raw_campaigns):
    def create_campaign_onehot_features(campaigns_data: pl.DataFrame) -> pl.DataFrame:
        """Create one-hot encoded features for campaigns."""
        campaign_features_df = campaigns_data.select([
            "campaign_id",
            "target_gender",
            "promoted_item",
            "target_incomes",
        ]).with_columns([
            # One-hot for target_gender
            (pl.col("target_gender") == "no_preference").cast(pl.Int8).alias("gender_no_pref"),
            (pl.col("target_gender") == "balanced").cast(pl.Int8).alias("gender_balanced"),
            (pl.col("target_gender") == "predominantly_male").cast(pl.Int8).alias("gender_male"),
            (pl.col("target_gender") == "predominantly_female").cast(pl.Int8).alias("gender_female"),

            # One-hot for promoted_item
            (pl.col("promoted_item") == "product").cast(pl.Int8).alias("item_product"),
            (pl.col("promoted_item") == "service").cast(pl.Int8).alias("item_service"),
            (pl.col("promoted_item") == "newsletter").cast(pl.Int8).alias("item_newsletter"),
            (pl.col("promoted_item") == "knowledge_product").cast(pl.Int8).alias("item_knowledge"),
            (pl.col("promoted_item") == "event").cast(pl.Int8).alias("item_event"),
            (pl.col("promoted_item") == "other").cast(pl.Int8).alias("item_other"),

            # Income targeting
            pl.col("target_incomes").str.count_matches(r"range_").fill_null(0).alias("num_income_targets"),
            (pl.col("target_incomes").str.contains("range_1")).cast(pl.Int8).alias("income_range_1"),
            (pl.col("target_incomes").str.contains("range_2")).cast(pl.Int8).alias("income_range_2"),
            (pl.col("target_incomes").str.contains("range_3")).cast(pl.Int8).alias("income_range_3"),
            (pl.col("target_incomes").str.contains("range_4")).cast(pl.Int8).alias("income_range_4"),
            (pl.col("target_incomes").str.contains("range_5")).cast(pl.Int8).alias("income_range_5"),
        ])

        return campaign_features_df

    campaign_onehot_features = create_campaign_onehot_features(raw_campaigns)

    print("=== Unique Values ===")
    print(f"Target Gender: {raw_campaigns['target_gender'].unique().to_list()}")
    print(f"Promoted Item: {raw_campaigns['promoted_item'].unique().to_list()}")

    campaign_onehot_features.head(10)
    return (campaign_onehot_features,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Build Feature Matrix

    Join all features together using pre-computed historical statistics (last 90 days).
    Memory-efficient approach: stats computed once per group, not per date.
    """)
    return


@app.cell
def _(
    campaign_historical_stats,
    campaign_onehot_features,
    pl,
    publication_audience_stats,
    publication_historical_stats,
    test_split_data,
    tfidf_features_df,
    train_split_data,
):
    def build_complete_feature_matrix(
        placements_data: pl.DataFrame,
        campaign_features_df: pl.DataFrame,
        tfidf_features_df: pl.DataFrame,
        campaign_stats_df: pl.DataFrame,
        publication_stats_df: pl.DataFrame,
        publication_audience_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Build feature matrix by joining all feature sources.
        Uses pre-computed historical statistics (memory-efficient).
        """
        feature_df = placements_data.clone()

        # Join with campaign features (static)
        feature_df = feature_df.join(
            campaign_features_df.select([
                "campaign_id",
                "gender_no_pref", "gender_balanced", "gender_male", "gender_female",
                "item_product", "item_service", "item_newsletter", "item_knowledge",
                "item_event", "item_other",
                "num_income_targets",
                "income_range_1", "income_range_2", "income_range_3", "income_range_4", "income_range_5",
            ]),
            on="campaign_id",
            how="left"
        )

        # Join with TF-IDF features (static)
        feature_df = feature_df.join(
            tfidf_features_df,
            on="publication_id",
            how="left"
        )

        # Join campaign CTR stats (already aggregated, no sorting needed)
        feature_df = feature_df.join(
            campaign_stats_df,
            on="campaign_id",
            how="left"
        )

        # Join publication CTR stats
        feature_df = feature_df.join(
            publication_stats_df,
            on="publication_id",
            how="left"
        )

        # Join publication audience size
        feature_df = feature_df.join(
            publication_audience_df,
            on="publication_id",
            how="left"
        )

        # Fill nulls for numeric columns
        numeric_cols_list = [
            "campaign_id_ctr_mean", "campaign_id_ctr_std", "campaign_id_ctr_count",
            "publication_id_ctr_mean", "publication_id_ctr_std", "publication_id_ctr_count",
            "publication_id_avg_opens"
        ]

        for col_name in numeric_cols_list:
            if col_name in feature_df.columns:
                feature_df = feature_df.with_columns(pl.col(col_name).fill_null(0.0))

        # Fill nulls for one-hot encoded columns
        onehot_cols_list = [
            "gender_no_pref", "gender_balanced", "gender_male", "gender_female",
            "item_product", "item_service", "item_newsletter", "item_knowledge",
            "item_event", "item_other",
            "num_income_targets",
            "income_range_1", "income_range_2", "income_range_3", "income_range_4", "income_range_5",
        ]

        for col_name in onehot_cols_list:
            if col_name in feature_df.columns:
                feature_df = feature_df.with_columns(pl.col(col_name).fill_null(0))

        # Fill TF-IDF columns with 0
        tfidf_cols_list = [c for c in feature_df.columns if c.startswith("tfidf_")]
        for col_name in tfidf_cols_list:
            feature_df = feature_df.with_columns(pl.col(col_name).fill_null(0.0))

        return feature_df

    # Build feature matrices
    print("Building feature matrix for training data...")
    train_feature_matrix = build_complete_feature_matrix(
        train_split_data,
        campaign_onehot_features,
        tfidf_features_df,
        campaign_historical_stats,
        publication_historical_stats,
        publication_audience_stats,
    )

    print("Building feature matrix for test data...")
    test_feature_matrix = build_complete_feature_matrix(
        test_split_data,
        campaign_onehot_features,
        tfidf_features_df,
        campaign_historical_stats,
        publication_historical_stats,
        publication_audience_stats,
    )

    print(f"\nTrain feature matrix shape: {train_feature_matrix.shape}")
    print(f"Test feature matrix shape: {test_feature_matrix.shape}")

    train_feature_matrix.head(5)
    return test_feature_matrix, train_feature_matrix


@app.cell
def _(mo):
    mo.md("""
    ## 6. Calculate Prior CTR

    Calculate prior CTR for each (publication_id, campaign_id) pair from historical data.
    This will be used in the prediction formula: `predicted_ctr = prior_ctr + r`
    """)
    return


@app.cell
def _(pl, test_feature_matrix, train_feature_matrix):
    def calculate_prior_ctr_per_pair(train_data: pl.DataFrame) -> pl.DataFrame:
        """Calculate prior CTR for each (publication_id, campaign_id) pair."""
        prior_ctr_df = train_data.group_by(["publication_id", "campaign_id"]).agg([
            pl.col("ctr_smooth").mean().alias("prior_ctr"),
            pl.col("ctr_smooth").count().alias("prior_count"),
        ])
        return prior_ctr_df

    def add_prior_ctr_and_residual(
        feature_matrix: pl.DataFrame,
        prior_ctr_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Add prior CTR to feature matrix and calculate target residual."""
        matrix_with_prior = feature_matrix.join(
            prior_ctr_df.select(["publication_id", "campaign_id", "prior_ctr"]),
            on=["publication_id", "campaign_id"],
            how="left"
        ).with_columns([
            pl.col("prior_ctr").fill_null(0.0)
        ]).with_columns([
            (pl.col("ctr_smooth") - pl.col("prior_ctr")).alias("target_residual")
        ])

        return matrix_with_prior

    prior_ctr_matrix = calculate_prior_ctr_per_pair(train_feature_matrix)

    print(f"Unique (publication_id, campaign_id) pairs in training: {len(prior_ctr_matrix):,}")

    train_matrix_with_prior = add_prior_ctr_and_residual(train_feature_matrix, prior_ctr_matrix)
    test_matrix_with_prior = add_prior_ctr_and_residual(test_feature_matrix, prior_ctr_matrix)

    print(f"\nPrior CTR statistics:")
    print(f"  Mean: {prior_ctr_matrix['prior_ctr'].mean():.6f}")
    print(f"  Std: {prior_ctr_matrix['prior_ctr'].std():.6f}")
    print(f"  Min: {prior_ctr_matrix['prior_ctr'].min():.6f}")
    print(f"  Max: {prior_ctr_matrix['prior_ctr'].max():.6f}")

    test_matrix_with_prior.head(5)
    return test_matrix_with_prior, train_matrix_with_prior


@app.cell
def _(mo):
    mo.md("""
    ## 7. Prepare Data for Ensemble Models

    Split into two groups:
    - Small audience: < 1,000 opens
    - Large audience: >= 1,000 opens
    """)
    return


@app.cell
def _(List, Tuple, np, pl, train_matrix_with_prior):
    def extract_feature_columns(df: pl.DataFrame) -> List[str]:
        """Extract feature column names (excluding metadata and target columns)."""
        feature_cols_list = [
            # Campaign CTR stats
            "campaign_id_ctr_mean", "campaign_id_ctr_std", "campaign_id_ctr_count",
            # Publication CTR stats
            "publication_id_ctr_mean", "publication_id_ctr_std", "publication_id_ctr_count",
            # Publication audience size
            "publication_id_avg_opens",
            # Target gender one-hot
            "gender_no_pref", "gender_balanced", "gender_male", "gender_female",
            # Promoted item one-hot
            "item_product", "item_service", "item_newsletter", "item_knowledge",
            "item_event", "item_other",
            # Income targeting
            "num_income_targets",
            "income_range_1", "income_range_2", "income_range_3", "income_range_4", "income_range_5",
        ]

        # Add TF-IDF columns
        tfidf_cols_list = [c for c in df.columns if c.startswith("tfidf_")]
        feature_cols_list.extend(tfidf_cols_list)

        # Filter to only existing columns
        feature_cols_list = [c for c in feature_cols_list if c in df.columns]

        return feature_cols_list

    def prepare_ensemble_data(
        feature_matrix_with_prior: pl.DataFrame,
        feature_cols_list: List[str],
        threshold_opens: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data by audience size and prepare numpy arrays for XGBoost.
        Returns: X_small, y_small, prior_small, weights_small, X_large, y_large, prior_large, weights_large

        Note: weights are based on approved_opens to give more importance to high-traffic records.
        """
        small_audience_df = feature_matrix_with_prior.filter(pl.col("approved_opens") < threshold_opens)
        large_audience_df = feature_matrix_with_prior.filter(pl.col("approved_opens") >= threshold_opens)

        # Small audience arrays
        X_small_arr = small_audience_df.select(feature_cols_list).to_numpy()
        X_small_arr = np.nan_to_num(X_small_arr, nan=0.0, posinf=0.0, neginf=0.0)
        y_small_arr = small_audience_df.select("target_residual").to_numpy().flatten()
        prior_small_arr = small_audience_df.select("prior_ctr").to_numpy().flatten()
        weights_small_arr = small_audience_df.select("approved_opens").to_numpy().flatten().astype(np.float64)

        # Large audience arrays
        X_large_arr = large_audience_df.select(feature_cols_list).to_numpy()
        X_large_arr = np.nan_to_num(X_large_arr, nan=0.0, posinf=0.0, neginf=0.0)
        y_large_arr = large_audience_df.select("target_residual").to_numpy().flatten()
        prior_large_arr = large_audience_df.select("prior_ctr").to_numpy().flatten()
        weights_large_arr = large_audience_df.select("approved_opens").to_numpy().flatten().astype(np.float64)

        return X_small_arr, y_small_arr, prior_small_arr, weights_small_arr, X_large_arr, y_large_arr, prior_large_arr, weights_large_arr

    feature_columns_list = extract_feature_columns(train_matrix_with_prior)

    print(f"Total features: {len(feature_columns_list)}")
    print(f"Features: {feature_columns_list[:10]}...")

    (
        X_train_small_data,
        y_train_small_data,
        prior_train_small_data,
        weights_train_small_data,
        X_train_large_data,
        y_train_large_data,
        prior_train_large_data,
        weights_train_large_data,
    ) = prepare_ensemble_data(train_matrix_with_prior, feature_columns_list, threshold_opens=1000)

    print(f"\nTraining data split:")
    small_train_count = len(X_train_small_data)
    large_train_count = len(X_train_large_data)
    print(f"  Small audience (< 1k opens): {small_train_count:,} samples")
    print(f"  Large audience (>= 1k opens): {large_train_count:,} samples")

    print(f"\nX_train_small shape: {X_train_small_data.shape}")
    print(f"X_train_large shape: {X_train_large_data.shape}")

    print(f"\nSample weights (approved_opens) statistics:")
    print(f"  Small audience - Mean: {weights_train_small_data.mean():.2f}, Max: {weights_train_small_data.max():.2f}")
    print(f"  Large audience - Mean: {weights_train_large_data.mean():.2f}, Max: {weights_train_large_data.max():.2f}")

    feature_columns_list
    return (
        X_train_large_data,
        X_train_small_data,
        feature_columns_list,
        weights_train_large_data,
        weights_train_small_data,
        y_train_large_data,
        y_train_small_data,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 8. Train Ensemble Models

    Train separate XGBoost models for small and large audiences.
    """)
    return


@app.cell
def _(
    Dict,
    Optional,
    X_train_large_data,
    X_train_small_data,
    np,
    weights_train_large_data,
    weights_train_small_data,
    xgb,
    y_train_large_data,
    y_train_small_data,
):
    def create_xgboost_params() -> Dict:
        """Create XGBoost hyperparameters."""
        return {
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

    def train_xgboost_model(
        X_train: np.ndarray,
        y_train: np.ndarray,
        params_dict: Dict,
        sample_weights: Optional[np.ndarray] = None,
        num_rounds: int = 200
    ):
        """
        Train an XGBoost model with optional sample weights.

        Args:
            X_train: Feature matrix
            y_train: Target values
            params_dict: XGBoost hyperparameters
            sample_weights: Optional weights for each sample (e.g., approved_opens).
                           Records with higher weights have more influence on training.
            num_rounds: Number of boosting rounds

        Returns:
            Trained XGBoost model
        """
        dtrain_matrix = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        trained_model = xgb.train(
            params_dict,
            dtrain_matrix,
            num_boost_round=num_rounds,
            verbose_eval=50
        )
        return trained_model

    xgb_params_dict = create_xgboost_params()

    # Train model for small audience with sample weights
    print("Training model for small audience (< 1k opens) with sample weights (approved_opens)...")
    model_small_audience = train_xgboost_model(
        X_train_small_data,
        y_train_small_data,
        xgb_params_dict,
        sample_weights=weights_train_small_data,
        num_rounds=200
    )

    print("\nTraining model for large audience (>= 1k opens) with sample weights (approved_opens)...")
    model_large_audience = train_xgboost_model(
        X_train_large_data,
        y_train_large_data,
        xgb_params_dict,
        sample_weights=weights_train_large_data,
        num_rounds=200
    )

    print("\nBoth models trained successfully with sample weighting by approved_opens!")
    return model_large_audience, model_small_audience


@app.cell
def _(mo):
    mo.md("""
    ## 9. Evaluation on Test Set

    Evaluate the ensemble models on the test set (last 90 days).
    """)
    return


@app.cell
def _(
    List,
    Tuple,
    feature_columns_list,
    model_large_audience,
    model_small_audience,
    np,
    pl,
    test_matrix_with_prior,
    xgb,
):
    def predict_with_ensemble(
        test_feature_matrix_with_prior: pl.DataFrame,
        feature_cols_list: List[str],
        model_small_trained,
        model_large_trained,
        threshold_opens: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using ensemble models.
        Returns: y_test_true, y_pred_final
        """
        # Split test data by audience size
        test_small_df = test_feature_matrix_with_prior.filter(pl.col("approved_opens") < threshold_opens)
        test_large_df = test_feature_matrix_with_prior.filter(pl.col("approved_opens") >= threshold_opens)

        print(f"Test data split:")
        print(f"  Small audience (< 1k opens): {len(test_small_df):,} samples")
        print(f"  Large audience (>= 1k opens): {len(test_large_df):,} samples")

        # Prepare test arrays
        X_test_small_arr = test_small_df.select(feature_cols_list).to_numpy()
        X_test_small_arr = np.nan_to_num(X_test_small_arr, nan=0.0, posinf=0.0, neginf=0.0)
        y_test_small_true = test_small_df.select("ctr_smooth").to_numpy().flatten()
        prior_test_small_arr = test_small_df.select("prior_ctr").to_numpy().flatten()

        X_test_large_arr = test_large_df.select(feature_cols_list).to_numpy()
        X_test_large_arr = np.nan_to_num(X_test_large_arr, nan=0.0, posinf=0.0, neginf=0.0)
        y_test_large_true = test_large_df.select("ctr_smooth").to_numpy().flatten()
        prior_test_large_arr = test_large_df.select("prior_ctr").to_numpy().flatten()

        # Make predictions (residuals)
        dtest_small_matrix = xgb.DMatrix(X_test_small_arr)
        residual_pred_small_arr = model_small_trained.predict(dtest_small_matrix)

        dtest_large_matrix = xgb.DMatrix(X_test_large_arr)
        residual_pred_large_arr = model_large_trained.predict(dtest_large_matrix)

        # Final prediction: prior_ctr + residual
        y_pred_small_final = np.clip(prior_test_small_arr + residual_pred_small_arr, 0.0, 1.0)
        y_pred_large_final = np.clip(prior_test_large_arr + residual_pred_large_arr, 0.0, 1.0)

        # Combine results
        y_test_combined = np.concatenate([y_test_small_true, y_test_large_true])
        y_pred_combined = np.concatenate([y_pred_small_final, y_pred_large_final])

        print(f"\nTotal test samples: {len(y_test_combined):,}")

        return y_test_combined, y_pred_combined

    (
        y_test_final,
        y_pred_final,
    ) = predict_with_ensemble(
        test_matrix_with_prior,
        feature_columns_list,
        model_small_audience,
        model_large_audience,
        threshold_opens=1000
    )
    return y_pred_final, y_test_final


@app.cell
def _(
    Tuple,
    mean_absolute_error,
    np,
    pl,
    test_matrix_with_prior,
    y_pred_final,
    y_test_final,
):
    def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate model performance using MAE.
        Returns: overall_mae, baseline_mae
        """
        overall_mae_value = mean_absolute_error(y_true, y_pred)

        # Baseline comparison (predict mean)
        baseline_prediction = np.full_like(y_true, y_true.mean())
        baseline_mae_value = mean_absolute_error(y_true, baseline_prediction)

        return overall_mae_value, baseline_mae_value

    def calculate_split_mae_from_arrays(
        test_feature_matrix: pl.DataFrame,
        y_test_all_arr: np.ndarray,
        y_pred_all_arr: np.ndarray,
        threshold: int = 1000
    ) -> Tuple[float, float]:
        """Calculate MAE separately for small and large audiences."""
        opens_array = test_feature_matrix.select("approved_opens").to_numpy().flatten()
        small_indices_arr = np.where(opens_array < threshold)[0]
        large_indices_arr = np.where(opens_array >= threshold)[0]

        mae_small_val = mean_absolute_error(
            y_test_all_arr[small_indices_arr], 
            y_pred_all_arr[small_indices_arr]
        ) if len(small_indices_arr) > 0 else 0.0

        mae_large_val = mean_absolute_error(
            y_test_all_arr[large_indices_arr], 
            y_pred_all_arr[large_indices_arr]
        ) if len(large_indices_arr) > 0 else 0.0

        return mae_small_val, mae_large_val

    overall_mae, baseline_mae_result = evaluate_model_performance(y_test_final, y_pred_final)

    # Calculate split MAE using proper indexing
    mae_small_split, mae_large_split = calculate_split_mae_from_arrays(
        test_matrix_with_prior,
        y_test_final,
        y_pred_final,
        threshold=1000
    )

    print("=" * 70)
    print("MODEL EVALUATION RESULTS (MAE)")
    print("=" * 70)
    print(f"\n{'Dataset':<30} {'MAE':>15}")
    print("-" * 70)
    print(f"{'Small audience (< 1k opens)':<30} {mae_small_split:>15.6f}")
    print(f"{'Large audience (>= 1k opens)':<30} {mae_large_split:>15.6f}")
    print(f"{'All test data':<30} {overall_mae:>15.6f}")
    print("-" * 70)

    print(f"\nBaseline (predict mean) MAE: {baseline_mae_result:.6f}")
    print(f"Improvement over baseline: {(baseline_mae_result - overall_mae) / baseline_mae_result * 100:.2f}%")

    print(f"\nPrediction statistics:")
    print(f"  Predicted CTR - Mean: {y_pred_final.mean():.6f}, Std: {y_pred_final.std():.6f}")
    print(f"  Actual CTR - Mean: {y_test_final.mean():.6f}, Std: {y_test_final.std():.6f}")
    print(f"  Residual stats - Mean: {np.mean(y_pred_final - y_test_final):.6f}, Std: {np.std(y_pred_final - y_test_final):.6f}")

    baseline_mae_result, mae_large_split, mae_small_split, overall_mae
    return mae_large_split, mae_small_split, overall_mae


@app.cell
def _(mo):
    mo.md("""
    ## 10. Visualization and Analysis
    """)
    return


@app.cell
def _(np, plt, y_pred_final, y_test_final):
    def create_evaluation_plots(y_true_arr: np.ndarray, y_pred_arr: np.ndarray):
        """Create visualization plots for model evaluation."""
        fig_obj, axes_arr = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Scatter plot: Predicted vs Actual
        ax1 = axes_arr[0, 0]
        sample_indices = np.random.choice(len(y_true_arr), min(5000, len(y_true_arr)), replace=False)
        ax1.scatter(y_true_arr[sample_indices], y_pred_arr[sample_indices], alpha=0.3, s=5)
        ax1.plot([0, 0.2], [0, 0.2], 'r--', linewidth=2, label='Perfect prediction')
        ax1.set_xlabel('Actual CTR (smoothed)')
        ax1.set_ylabel('Predicted CTR')
        ax1.set_title('Predicted vs Actual CTR (5K sample)')
        ax1.set_xlim(0, 0.2)
        ax1.set_ylim(0, 0.2)
        ax1.legend()

        # 2. Residual distribution
        ax2 = axes_arr[0, 1]
        residuals_arr = y_pred_arr - y_true_arr
        ax2.hist(residuals_arr, bins=100, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residual (Predicted - Actual)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Residual Distribution\nMean: {np.mean(residuals_arr):.6f}, Std: {np.std(residuals_arr):.6f}')

        # 3. Prediction distribution
        ax3 = axes_arr[1, 0]
        ax3.hist(y_pred_arr, bins=100, alpha=0.7, label='Predicted', edgecolor='black')
        ax3.hist(y_true_arr, bins=100, alpha=0.5, label='Actual', edgecolor='black')
        ax3.set_xlabel('CTR (smoothed)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Predicted vs Actual CTR')
        ax3.legend()
        ax3.set_xlim(0, 0.15)

        # 4. Error by actual CTR bin
        ax4 = axes_arr[1, 1]
        ctr_bins_arr = np.linspace(0, 0.1, 11)
        bin_errors_list = []
        bin_centers_list = []

        for i in range(len(ctr_bins_arr) - 1):
            mask_bin = (y_true_arr >= ctr_bins_arr[i]) & (y_true_arr < ctr_bins_arr[i + 1])
            if np.sum(mask_bin) > 0:
                bin_mae_val = np.mean(np.abs(y_pred_arr[mask_bin] - y_true_arr[mask_bin]))
                bin_errors_list.append(bin_mae_val)
                bin_centers_list.append((ctr_bins_arr[i] + ctr_bins_arr[i + 1]) / 2)

        ax4.bar(bin_centers_list, bin_errors_list, width=0.008, edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Actual CTR Bin')
        ax4.set_ylabel('MAE')
        ax4.set_title('MAE by Actual CTR Range')

        plt.tight_layout()
        plt.show()

    create_evaluation_plots(y_test_final, y_pred_final)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 11. Summary
    """)
    return


@app.cell
def _(mae_large_split, mae_small_split, mo, overall_mae):
    summary_text = f"""
    ## Model Performance Summary

    ### Ensemble Model Architecture

    The model uses an ensemble approach with two specialized models:
    - **Model for Small Audiences** (< 1,000 opens): MAE = {mae_small_split:.6f}
    - **Model for Large Audiences** (>= 1,000 opens): MAE = {mae_large_split:.6f}
    - **Overall Test MAE**: {overall_mae:.6f}

    ### Key Features

    1. **Sample Weighting by Approved Opens**: Each training record is weighted by its `approved_opens` value.
       This gives more importance to high-traffic records during training, improving accuracy where it matters most.

    2. **Target Smoothing**: Used smoothed CTR formula `(clicks + m*p0) / (opens + m)` to reduce variance
       in low-opens scenarios while preserving signal in high-opens scenarios.

    3. **Prior-Based Prediction**: Model predicts residual `r` where `predicted_ctr = prior_ctr + r`,
       reducing the learning burden on the model.

    4. **Historical Statistics (Memory-Efficient)**: Stats computed once from last 90 days of training data,
       avoiding expensive per-date iterations that cause OOM errors.

    5. **Ensemble Strategy**: Separate models for small vs large audiences capture different patterns
       in these distinct regimes.

    ### Feature Set

    - Historical CTR statistics (last 90 days) for campaigns and publications
    - Publication tag TF-IDF vectors
    - Audience size (mean opens from last 90 days)
    - Target gender and promoted item type (one-hot encoded)
    - Target audience (income ranges, one-hot encoded)

    ### Training/Test Split

    - Training: All historical data before last 90 days
    - Testing: Last 90 days of data
    """
    mo.md(summary_text)
    return


if __name__ == "__main__":
    app.run()
