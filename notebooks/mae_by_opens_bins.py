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

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    mo.md("""
    # MAE Analysis by Opens Bins

    This notebook analyzes the Mean Absolute Error (MAE) of the trained XGBoost CTR model
    across different bins of `approved_opens` (audience size).

    ## Opens Bins
    - 0-100
    - 101-500
    - 501-1,000
    - 1,001-10,000
    - 10,001-100,000
    - 100,001+
    """)
    return Path, mo, np, pl, plt


@app.cell
def _(mo):
    mo.md("""
    ## 1. Data Loading
    """)
    return


@app.cell
def _(Path, pl):
    # Define data paths
    DATA_DIR = Path("../data")

    # Load all datasets
    placements_df = pl.read_csv(DATA_DIR / "placements.csv")
    campaigns_df = pl.read_csv(DATA_DIR / "campaigns.csv")
    publication_tags_df = pl.read_csv(DATA_DIR / "publication_tags.csv")

    print("=== Data Shapes ===")
    print(f"Placements: {placements_df.shape}")
    print(f"Campaigns: {campaigns_df.shape}")
    print(f"Publication tags: {publication_tags_df.shape}")

    placements_df.head(5)
    return campaigns_df, placements_df, publication_tags_df


@app.cell
def _(pl, placements_df):
    # Calculate CTR and parse datetime
    ctr_raw = pl.col("approved_clicks") / pl.col("approved_opens")

    placements_with_ctr = placements_df.with_columns([
        pl.when(pl.col("approved_opens") > 0)
          .then(pl.min_horizontal(ctr_raw, pl.lit(1.0)))
          .otherwise(0)
          .alias("ctr"),

        pl.col("post_send_at").str.to_datetime().alias("send_datetime")
    ])

    print(f"Placements with CTR: {len(placements_with_ctr):,}")
    print(f"Date range: {placements_with_ctr['send_datetime'].min()} to {placements_with_ctr['send_datetime'].max()}")

    placements_with_ctr.head(5)
    return (placements_with_ctr,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. Train/Test Split & Feature Engineering

    We use the same chronological split and feature engineering as the training script.
    """)
    return


@app.cell
def _(pl, placements_with_ctr):
    # Determine split point (80% train, 20% test by time)
    sorted_dates = placements_with_ctr.select("send_datetime").sort("send_datetime")
    split_idx = int(len(sorted_dates) * 0.8)
    split_date = sorted_dates.row(split_idx)[0]

    print(f"Split date: {split_date}")

    train_placements = placements_with_ctr.filter(pl.col("send_datetime") < split_date)
    test_placements = placements_with_ctr.filter(pl.col("send_datetime") >= split_date)

    print(f"Train samples: {len(train_placements):,}")
    print(f"Test samples: {len(test_placements):,}")
    return split_date, train_placements


@app.cell
def _(pl, train_placements):
    # Calculate historical CTR statistics per campaign (from training data only)
    campaign_ctr_stats = train_placements.group_by("campaign_id").agg([
        pl.col("ctr").mean().alias("campaign_ctr_mean"),
        pl.col("ctr").std().alias("campaign_ctr_std"),
        pl.col("ctr").count().alias("campaign_ctr_count"),
        pl.col("approved_clicks").sum().alias("campaign_total_clicks"),
        pl.col("approved_opens").sum().alias("campaign_total_opens"),
    ]).with_columns([
        (pl.col("campaign_total_clicks") / pl.col("campaign_total_opens")).alias("campaign_weighted_ctr"),
        pl.col("campaign_ctr_std").fill_null(0.0)
    ])

    # Calculate publication audience size metrics from training data
    publication_audience = train_placements.group_by("publication_id").agg([
        pl.col("approved_opens").mean().alias("pub_avg_opens"),
        pl.col("approved_opens").std().alias("pub_std_opens"),
        pl.col("approved_opens").sum().alias("pub_total_opens"),
        pl.col("ctr").mean().alias("pub_ctr_mean"),
        pl.count().alias("pub_placement_count"),
    ]).with_columns([
        pl.col("pub_std_opens").fill_null(0.0)
    ])

    print(f"Campaigns with historical stats: {len(campaign_ctr_stats):,}")
    print(f"Publications with audience data: {len(publication_audience):,}")
    return campaign_ctr_stats, publication_audience


@app.cell
def _(pl, publication_tags_df):
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Clean and prepare tags for TF-IDF
    tags_cleaned = publication_tags_df.with_columns([
        pl.col("tags")
        .str.replace_all(r"[\{\}'\"]", "")
        .str.replace_all(r",", " ")
        .str.to_lowercase()
        .alias("tags_cleaned")
    ]).filter(
        pl.col("tags_cleaned").is_not_null() & (pl.col("tags_cleaned") != "")
    )

    publication_ids = tags_cleaned["publication_id"].to_list()
    tags_texts = tags_cleaned["tags_cleaned"].to_list()
    tags_texts = [t if t is not None else "" for t in tags_texts]

    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=50,
        min_df=5,
        stop_words='english'
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(tags_texts)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    # Create DataFrame with TF-IDF features
    tfidf_data = {"publication_id": publication_ids}
    for i, name in enumerate(tfidf_feature_names):
        tfidf_data[f"tfidf_{name}"] = tfidf_matrix[:, i].toarray().flatten().tolist()

    tfidf_df = pl.DataFrame(tfidf_data)
    print(f"TF-IDF features: {len(tfidf_feature_names)}")
    return (tfidf_df,)


@app.cell
def _(campaigns_df, pl):
    # Create one-hot encoded features for campaigns
    campaign_features = campaigns_df.select([
        "campaign_id",
        "advertiser_id",
        "target_gender",
        "promoted_item",
        "target_incomes",
        "target_ages"
    ]).with_columns([
        (pl.col("target_gender") == "no_preference").cast(pl.Int8).alias("gender_no_pref"),
        (pl.col("target_gender") == "balanced").cast(pl.Int8).alias("gender_balanced"),
        (pl.col("target_gender") == "predominantly_male").cast(pl.Int8).alias("gender_male"),
        (pl.col("target_gender") == "predominantly_female").cast(pl.Int8).alias("gender_female"),
        (pl.col("promoted_item") == "product").cast(pl.Int8).alias("item_product"),
        (pl.col("promoted_item") == "service").cast(pl.Int8).alias("item_service"),
        (pl.col("promoted_item") == "newsletter").cast(pl.Int8).alias("item_newsletter"),
        (pl.col("promoted_item") == "knowledge_product").cast(pl.Int8).alias("item_knowledge"),
        (pl.col("promoted_item") == "event").cast(pl.Int8).alias("item_event"),
        (pl.col("promoted_item") == "other").cast(pl.Int8).alias("item_other"),
        pl.col("target_incomes").str.count_matches(r"range_").fill_null(0).alias("num_income_targets"),
        pl.col("target_ages").str.count_matches(r"range_").fill_null(0).alias("num_age_targets"),
    ])
    return (campaign_features,)


@app.cell
def _(pl):
    # Add temporal features function
    def add_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
        """Add temporal features to placements DataFrame."""
        return df.with_columns([
            pl.col("send_datetime").dt.weekday().alias("day_of_week"),
            pl.col("send_datetime").dt.hour().alias("hour"),
            pl.col("send_datetime").dt.month().alias("month"),
            pl.col("send_datetime").dt.year().alias("year"),
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
    return (add_temporal_features,)


@app.cell
def _(
    add_temporal_features,
    campaign_ctr_stats,
    campaign_features,
    pl,
    placements_with_ctr,
    publication_audience,
    split_date,
    tfidf_df,
):
    def build_feature_matrix(placements_df: pl.DataFrame) -> pl.DataFrame:
        """Build the complete feature matrix for model prediction."""
        df = add_temporal_features(placements_df)

        df = df.join(
            campaign_features.select([
                "campaign_id", "advertiser_id",
                "gender_no_pref", "gender_balanced", "gender_male", "gender_female",
                "item_product", "item_service", "item_newsletter", "item_knowledge", 
                "item_event", "item_other",
                "num_income_targets", "num_age_targets"
            ]),
            on="campaign_id",
            how="left"
        )

        df = df.join(
            campaign_ctr_stats.select([
                "campaign_id", 
                "campaign_ctr_mean", "campaign_ctr_std", "campaign_ctr_count",
                "campaign_weighted_ctr"
            ]),
            on="campaign_id",
            how="left"
        )

        df = df.join(
            publication_audience.select([
                "publication_id",
                "pub_avg_opens", "pub_std_opens", "pub_ctr_mean", "pub_placement_count"
            ]),
            on="publication_id",
            how="left"
        )

        df = df.join(tfidf_df, on="publication_id", how="left")

        # Fill nulls
        numeric_cols = [
            "campaign_ctr_mean", "campaign_ctr_std", "campaign_ctr_count", "campaign_weighted_ctr",
            "pub_avg_opens", "pub_std_opens", "pub_ctr_mean", "pub_placement_count"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(0.0))

        onehot_cols = [
            "gender_no_pref", "gender_balanced", "gender_male", "gender_female",
            "item_product", "item_service", "item_newsletter", "item_knowledge", 
            "item_event", "item_other",
            "hour_morning", "hour_midday", "hour_night",
            "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun"
        ]
        for col in onehot_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(0))

        tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
        for col in tfidf_cols:
            df = df.with_columns(pl.col(col).fill_null(0.0))

        return df

    # Build test feature matrix
    print("Building feature matrix for test data...")
    test_df = build_feature_matrix(
        placements_with_ctr.filter(pl.col("send_datetime") >= split_date)
    )
    print(f"Test feature matrix shape: {test_df.shape}")
    return (test_df,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Train Model & Make Predictions

    We train the XGBoost model (or load if saved) and generate predictions on the test set.
    """)
    return


@app.cell
def _(
    add_temporal_features,
    campaign_ctr_stats,
    campaign_features,
    pl,
    placements_with_ctr,
    publication_audience,
    split_date,
    tfidf_df,
):
    # Build training feature matrix
    def build_train_feature_matrix(placements_df: pl.DataFrame) -> pl.DataFrame:
        """Build the complete feature matrix for model training."""
        df = add_temporal_features(placements_df)

        df = df.join(
            campaign_features.select([
                "campaign_id", "advertiser_id",
                "gender_no_pref", "gender_balanced", "gender_male", "gender_female",
                "item_product", "item_service", "item_newsletter", "item_knowledge", 
                "item_event", "item_other",
                "num_income_targets", "num_age_targets"
            ]),
            on="campaign_id",
            how="left"
        )

        df = df.join(
            campaign_ctr_stats.select([
                "campaign_id", 
                "campaign_ctr_mean", "campaign_ctr_std", "campaign_ctr_count",
                "campaign_weighted_ctr"
            ]),
            on="campaign_id",
            how="left"
        )

        df = df.join(
            publication_audience.select([
                "publication_id",
                "pub_avg_opens", "pub_std_opens", "pub_ctr_mean", "pub_placement_count"
            ]),
            on="publication_id",
            how="left"
        )

        df = df.join(tfidf_df, on="publication_id", how="left")

        # Fill nulls
        numeric_cols = [
            "campaign_ctr_mean", "campaign_ctr_std", "campaign_ctr_count", "campaign_weighted_ctr",
            "pub_avg_opens", "pub_std_opens", "pub_ctr_mean", "pub_placement_count"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(0.0))

        onehot_cols = [
            "gender_no_pref", "gender_balanced", "gender_male", "gender_female",
            "item_product", "item_service", "item_newsletter", "item_knowledge", 
            "item_event", "item_other",
            "hour_morning", "hour_midday", "hour_night",
            "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun"
        ]
        for col in onehot_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(0))

        tfidf_cols_fill = [c for c in df.columns if c.startswith("tfidf_")]
        for col in tfidf_cols_fill:
            df = df.with_columns(pl.col(col).fill_null(0.0))

        return df

    # Build training feature matrix
    print("Building feature matrix for training data...")
    train_df = build_train_feature_matrix(
        placements_with_ctr.filter(pl.col("send_datetime") < split_date)
    )
    print(f"Train feature matrix shape: {train_df.shape}")
    return (train_df,)


@app.cell
def _(np, test_df, train_df):
    import xgboost as xgb

    # Define feature columns
    feature_columns = [
        "campaign_ctr_mean", "campaign_ctr_std", "campaign_ctr_count", "campaign_weighted_ctr",
        "pub_avg_opens", "pub_std_opens", "pub_ctr_mean", "pub_placement_count",
        "gender_no_pref", "gender_balanced", "gender_male", "gender_female",
        "item_product", "item_service", "item_newsletter", "item_knowledge", 
        "item_event", "item_other",
        "num_income_targets", "num_age_targets",
        "month",
        "hour_morning", "hour_midday", "hour_night",
        "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun",
    ]

    # Add TF-IDF columns
    tfidf_cols = [c for c in train_df.columns if c.startswith("tfidf_")]
    feature_columns.extend(tfidf_cols)
    feature_columns = [c for c in feature_columns if c in train_df.columns]

    print(f"Total features: {len(feature_columns)}")

    # Prepare training data
    X_train = train_df.select(feature_columns).to_numpy()
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = train_df.select("ctr").to_numpy().flatten()
    w_train = train_df.select("approved_opens").to_numpy().flatten().astype(np.float32)
    w_train = w_train / w_train.mean()

    # Prepare test data
    X_test = test_df.select(feature_columns).to_numpy()
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = test_df.select("ctr").to_numpy().flatten()
    opens_test = test_df.select("approved_opens").to_numpy().flatten()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Train XGBoost model
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)

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

    print("\nTraining XGBoost model...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        verbose_eval=50
    )
    print("Model training complete!")

    # Make predictions
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    y_pred = np.clip(y_pred, 0.001, 0.999)

    print(f"\nPredictions made: {len(y_pred):,}")
    print(f"Prediction range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")

    # Overall MAE
    overall_mae = np.mean(np.abs(y_test - y_pred))
    print(f"Overall Test MAE: {overall_mae:.6f}")
    return opens_test, y_pred, y_test


@app.cell
def _(mo):
    mo.md("""
    ## 4. MAE by Opens Bins

    We calculate the Mean Absolute Error for different bins of `approved_opens`:
    - **0-100**: Very small audience
    - **101-500**: Small audience
    - **501-1,000**: Medium-small audience
    - **1,001-10,000**: Medium audience
    - **10,001-100,000**: Large audience
    - **100,001+**: Very large audience
    """)
    return


@app.cell
def _(np, opens_test, pl, y_pred, y_test):
    from dataclasses import dataclass
    from typing import List, Tuple

    @dataclass
    class BinStats:
        """Statistics for a single opens bin."""
        bin_label: str
        bin_min: int
        bin_max: int
        count: int
        mae: float
        median_ae: float
        std_ae: float
        mean_ctr: float
        mean_pred: float

    def calculate_mae_by_bins(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        opens: np.ndarray,
        bins: List[Tuple[int, int, str]]
    ) -> List[BinStats]:
        """Calculate MAE for each opens bin."""
        results = []

        for bin_min, bin_max, label in bins:
            if bin_max == float('inf'):
                mask = opens >= bin_min
            else:
                mask = (opens >= bin_min) & (opens <= bin_max)

            count = np.sum(mask)
            if count > 0:
                errors = np.abs(y_true[mask] - y_pred[mask])
                mae = np.mean(errors)
                median_ae = np.median(errors)
                std_ae = np.std(errors)
                mean_ctr = np.mean(y_true[mask])
                mean_pred_val = np.mean(y_pred[mask])
            else:
                mae = median_ae = std_ae = mean_ctr = mean_pred_val = 0.0

            results.append(BinStats(
                bin_label=label,
                bin_min=bin_min,
                bin_max=bin_max if bin_max != float('inf') else -1,
                count=int(count),
                mae=mae,
                median_ae=median_ae,
                std_ae=std_ae,
                mean_ctr=mean_ctr,
                mean_pred=mean_pred_val
            ))

        return results

    # Define bins
    opens_bins = [
        (0, 100, "0-100"),
        (101, 500, "101-500"),
        (501, 1000, "501-1,000"),
        (1001, 10000, "1,001-10,000"),
        (10001, 100000, "10,001-100,000"),
        (100001, float('inf'), "100,001+"),
    ]

    # Calculate MAE by bins
    bin_results = calculate_mae_by_bins(y_test, y_pred, opens_test, opens_bins)

    # Create summary DataFrame
    summary_data = {
        "Opens Bin": [r.bin_label for r in bin_results],
        "Count": [r.count for r in bin_results],
        "MAE": [r.mae for r in bin_results],
        "Median AE": [r.median_ae for r in bin_results],
        "Std AE": [r.std_ae for r in bin_results],
        "Mean CTR": [r.mean_ctr for r in bin_results],
        "Mean Pred": [r.mean_pred for r in bin_results],
    }

    summary_df = pl.DataFrame(summary_data)
    print("=== MAE by Opens Bins ===\n")
    print(summary_df)
    return (bin_results,)


@app.cell
def _(bin_results, mo):
    # Display as formatted table
    table_rows = []
    for r in bin_results:
        pct_data = r.count / sum(b.count for b in bin_results) * 100
        table_rows.append(f"| {r.bin_label} | {r.count:,} ({pct_data:.1f}%) | {r.mae:.6f} | {r.median_ae:.6f} | {r.mean_ctr:.6f} | {r.mean_pred:.6f} |")

    total_mae = sum(r.mae * r.count for r in bin_results) / sum(r.count for r in bin_results)

    table_md = f"""
    ## MAE by Opens Bins - Summary Table

    | Opens Bin | Count (%) | MAE | Median AE | Mean CTR | Mean Pred |
    |-----------|-----------|-----|-----------|----------|-----------|
    {chr(10).join(table_rows)}

    **Overall Test MAE: {total_mae:.6f}**
    """
    mo.md(table_md)
    return


@app.cell
def _(bar, bin_results, np, plt):
    # Visualization: Bar chart of MAE by Opens Bin
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: MAE by bin
    ax1 = axes[0]
    labels = [r.bin_label for r in bin_results]
    maes = [r.mae for r in bin_results]
    counts = [r.count for r in bin_results]

    x_pos = np.arange(len(labels))
    bars11 = ax1.bar(x_pos, maes, color='steelblue', edgecolor='black', alpha=0.8)

    ax1.set_xlabel('Opens Bin', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax1.set_title('MAE by Approved Opens Bin', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    # Add value labels on bars
    for bar_z, mae in zip(bars11, maes):
        height_2 = bar_z.get_height()
        ax1.annotate(f'{mae:.5f}',
                    xy=(bar_z.get_x() + bar_z.get_width() / 2, height_2),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Sample count by bin
    ax2 = axes[1]
    bars33 = ax2.bar(x_pos, counts, color='coral', edgecolor='black', alpha=0.8)

    ax2.set_xlabel('Opens Bin', fontsize=12)
    ax2.set_ylabel('Sample Count', fontsize=12)
    ax2.set_title('Sample Distribution by Opens Bin', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')

    # Add value labels on bars
    for bar_z, count in zip(bars33, counts):
        height_2 = bar.get_height()
        ax2.annotate(f'{count:,}',
                    xy=(bar_z.get_x() + bar_z.get_width() / 2, height_2),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(bin_results, np, plt):
    # Additional visualization: MAE vs Mean CTR by bin
    fig2, ax = plt.subplots(figsize=(10, 6))

    labels_2 = [r.bin_label for r in bin_results]
    maes_2 = [r.mae for r in bin_results]
    mean_ctrs = [r.mean_ctr for r in bin_results]
    mean_preds = [r.mean_pred for r in bin_results]

    x_pos_2 = np.arange(len(labels_2))
    width = 0.25

    bars1 = ax.bar(x_pos_2 - width, maes_2, width, label='MAE', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x_pos_2, mean_ctrs, width, label='Mean Actual CTR', color='seagreen', edgecolor='black')
    bars3 = ax.bar(x_pos_2 + width, mean_preds, width, label='Mean Predicted CTR', color='coral', edgecolor='black')

    ax.set_xlabel('Opens Bin', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('MAE, Mean Actual CTR, and Mean Predicted CTR by Opens Bin', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos_2)
    ax.set_xticklabels(labels_2, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(bin_results, mo):
    # Key insights
    min_mae_bin = min(bin_results, key=lambda x: x.mae if x.count > 0 else float('inf'))
    max_mae_bin = max(bin_results, key=lambda x: x.mae if x.count > 0 else 0)

    insights = f"""
    ## Key Insights

    ### MAE Trends by Audience Size

    1. **Lowest MAE**: The **{min_mae_bin.bin_label}** opens bin has the lowest MAE of **{min_mae_bin.mae:.6f}**
       - This bin contains {min_mae_bin.count:,} samples
       - Mean actual CTR: {min_mae_bin.mean_ctr:.6f}

    2. **Highest MAE**: The **{max_mae_bin.bin_label}** opens bin has the highest MAE of **{max_mae_bin.mae:.6f}**
       - This bin contains {max_mae_bin.count:,} samples
       - Mean actual CTR: {max_mae_bin.mean_ctr:.6f}

    ### Interpretation

    - **Small audience bins** (0-100, 101-500) tend to have more variable CTR due to sampling noise
    - **Large audience bins** (10,001+) have more stable CTR estimates, but may show different MAE patterns
    - The model's performance varies across audience sizes, which is important for understanding reliability of predictions
    """
    mo.md(insights)
    return


@app.cell
def _(bin_results, np, plt):
    # Weighted MAE contribution chart
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    labels_3 = [r.bin_label for r in bin_results]
    counts_3 = [r.count for r in bin_results]
    maes_3 = [r.mae for r in bin_results]
    total_samples = sum(counts_3)

    # Calculate weighted contribution to overall MAE
    weighted_mae_contrib = [(c * m) / total_samples for c, m in zip(counts_3, maes_3)]

    x_pos_3 = np.arange(len(labels_3))
    bars = ax3.bar(x_pos_3, weighted_mae_contrib, color='purple', edgecolor='black', alpha=0.7)

    ax3.set_xlabel('Opens Bin', fontsize=12)
    ax3.set_ylabel('Weighted MAE Contribution', fontsize=12)
    ax3.set_title('Contribution to Overall MAE by Opens Bin', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos_3)
    ax3.set_xticklabels(labels_3, rotation=45, ha='right')

    # Add value labels
    for bar, val in zip(bars, weighted_mae_contrib):
        height = bar.get_height()
        ax3.annotate(f'{val:.6f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\nSum of weighted MAE contributions: {sum(weighted_mae_contrib):.6f}")
    return (bar,)


if __name__ == "__main__":
    app.run()
