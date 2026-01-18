# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "polars>=1.37.1",
#     "matplotlib>=3.10.8",
#     "seaborn>=0.13.0",
#     "scipy>=1.15.3",
#     "scikit-learn>=1.5.0",
#     "sentence-transformers>=2.2.0",
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
    from typing import Tuple, Dict, List, Optional
    from dataclasses import dataclass

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    mo.md("""
    # XGBoost CTR Prediction Model

    This notebook builds an XGBoost model to predict Click-Through Rate (CTR) using the following features:

    ## Features
    1. **Historical CTR Statistics** - mean, std, count for campaign_id
    2. **TF-IDF Vectors** - based on publication tags
    3. **Audience Size** - historical mean of previous 3 months opens by publication_id
    4. **Target Gender** - one-hot encoded
    5. **Promoted Item Type** - one-hot encoded
    6. **Temporal Features** - day of week, hour bucket (morning/mid-day/night), month
    7. **Target Audience** - one-hot encoded for target_gender and target_incomes
    8. **Embedding Norms** - L2 norm of advertiser and publication embeddings

    ## Training
    - Historical train/test split (train on older data, test on newer)
    - **Sample weighting** by number of opens (more reliable CTR estimates get higher weight)

    ## Evaluation
    - Metrics: Binomial Log Loss and MAE (both weighted and unweighted)
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
    advertisers_df = pl.read_csv(DATA_DIR / "advertisers.csv")
    publication_mets_df = pl.read_csv(DATA_DIR / "publication_mets.csv")
    publication_tags_df = pl.read_csv(DATA_DIR / "publication_tags.csv")

    print("=== Data Shapes ===")
    print(f"Placements: {placements_df.shape}")
    print(f"Campaigns: {campaigns_df.shape}")
    print(f"Advertisers: {advertisers_df.shape}")
    print(f"Publication metadata: {publication_mets_df.shape}")
    print(f"Publication tags: {publication_tags_df.shape}")

    placements_df.head(5)
    return DATA_DIR, campaigns_df, placements_df, publication_tags_df


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
    ## 2. Historical Train/Test Split

    We split data chronologically to simulate real-world prediction scenarios:
    - Train on historical data
    - Test on more recent data
    """)
    return


@app.cell
def _(pl, placements_with_ctr):
    # Determine split point (80% train, 20% test by time)
    sorted_dates = placements_with_ctr.select("send_datetime").sort("send_datetime")
    split_idx = int(len(sorted_dates) * 0.8)
    split_date = sorted_dates.row(split_idx)[0]

    print(f"Split date: {split_date}")
    print(f"Training data: before {split_date}")
    print(f"Test data: on or after {split_date}")

    # Create train/test masks
    train_mask = pl.col("send_datetime") < split_date
    test_mask = pl.col("send_datetime") >= split_date

    train_placements = placements_with_ctr.filter(train_mask)
    test_placements = placements_with_ctr.filter(test_mask)

    print(f"\nTrain samples: {len(train_placements):,}")
    print(f"Test samples: {len(test_placements):,}")
    return split_date, train_placements


@app.cell
def _(mo):
    mo.md("""
    ## 3. Feature Engineering

    ### 3.1 Historical CTR Statistics for Campaigns
    Compute historical CTR mean, std, and count for each campaign_id using only training data.
    """)
    return


@app.cell
def _(pl, train_placements):
    # Calculate historical CTR statistics per campaign (from training data only)
    # TODO: change to be on a rolling window of last 3 months only
    campaign_ctr_stats = train_placements.group_by("campaign_id").agg([
        pl.col("ctr").mean().alias("campaign_ctr_mean"),
        pl.col("ctr").std().alias("campaign_ctr_std"),
        pl.col("ctr").count().alias("campaign_ctr_count"),
        pl.col("approved_clicks").sum().alias("campaign_total_clicks"),
        pl.col("approved_opens").sum().alias("campaign_total_opens"),
    ]).with_columns([
        (pl.col("campaign_total_clicks") / pl.col("campaign_total_opens")).alias("campaign_weighted_ctr"),
        pl.col("campaign_ctr_std").fill_null(0.0)  # Fill NaN std for single observations
    ])

    print(f"Campaigns with historical stats: {len(campaign_ctr_stats):,}")
    campaign_ctr_stats.head(10)
    return (campaign_ctr_stats,)


@app.cell
def _(mo):
    mo.md("""
    ### 3.2 Publication Audience Size
    Calculate historical mean opens (audience size proxy) for each publication using a rolling 3-month window.
    """)
    return


@app.cell
def _(pl, train_placements):
    # Calculate publication audience size metrics from training data
    # For simplicity, we'll use the overall historical mean opens per publication
    publication_audience = train_placements.group_by("publication_id").agg([
        pl.col("approved_opens").mean().alias("pub_avg_opens"),
        pl.col("approved_opens").std().alias("pub_std_opens"),
        pl.col("approved_opens").sum().alias("pub_total_opens"),
        pl.col("ctr").mean().alias("pub_ctr_mean"),
        pl.count().alias("pub_placement_count"),
    ]).with_columns([
        pl.col("pub_std_opens").fill_null(0.0)
    ])

    print(f"Publications with audience data: {len(publication_audience):,}")
    publication_audience.head(10)
    return (publication_audience,)


@app.cell
def _(mo):
    mo.md("""
    ### 3.3 TF-IDF Features from Publication Tags
    Create TF-IDF vectors from the publication tags.
    """)
    return


@app.cell
def _(pl, publication_tags_df):
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Clean and prepare tags for TF-IDF
    tags_cleaned = publication_tags_df.with_columns([
        # Clean the tags string: remove braces, quotes, and extra whitespace
        pl.col("tags")
        .str.replace_all(r"[\{\}'\"]", "")
        .str.replace_all(r",", " ")
        .str.to_lowercase()
        .alias("tags_cleaned")
    ]).filter(
        pl.col("tags_cleaned").is_not_null() & (pl.col("tags_cleaned") != "")
    )

    # Get publication IDs and tags as lists (avoiding pandas/pyarrow dependency)
    publication_ids = tags_cleaned["publication_id"].to_list()
    tags_texts = tags_cleaned["tags_cleaned"].to_list()

    # Replace None with empty string
    tags_texts = [t if t is not None else "" for t in tags_texts]

    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=50,  # Limit to top 50 features to avoid high dimensionality
        min_df=5,  # Minimum document frequency
        stop_words='english'
    )

    # Fit and transform
    tfidf_matrix = tfidf_vectorizer.fit_transform(tags_texts)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    print(f"TF-IDF features: {len(tfidf_feature_names)}")
    print(f"Feature names: {tfidf_feature_names[:20]}...")

    # Create DataFrame with TF-IDF features
    tfidf_data = {"publication_id": publication_ids}

    # Add TF-IDF columns
    for i, name in enumerate(tfidf_feature_names):
        tfidf_data[f"tfidf_{name}"] = tfidf_matrix[:, i].toarray().flatten().tolist()

    tfidf_df = pl.DataFrame(tfidf_data)

    print(f"\nTF-IDF DataFrame shape: {tfidf_df.shape}")
    tfidf_df.head(5)
    return (tfidf_df,)


@app.cell
def _(mo):
    mo.md("""
    ### 3.4 One-Hot Encoding for Categorical Features
    Encode target_gender, promoted_item_type, and target_incomes.
    """)
    return


@app.cell
def _(campaigns_df, pl):
    # Prepare campaign features with one-hot encoding
    # First, let's see unique values
    print("=== Unique Values ===")
    print(f"Target Gender: {campaigns_df['target_gender'].unique().to_list()}")
    print(f"Promoted Item: {campaigns_df['promoted_item'].unique().to_list()}")

    # Create one-hot encoded features for campaigns
    campaign_features = campaigns_df.select([
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

        # One-hot for promoted_item
        (pl.col("promoted_item") == "product").cast(pl.Int8).alias("item_product"),
        (pl.col("promoted_item") == "service").cast(pl.Int8).alias("item_service"),
        (pl.col("promoted_item") == "newsletter").cast(pl.Int8).alias("item_newsletter"),
        (pl.col("promoted_item") == "knowledge_product").cast(pl.Int8).alias("item_knowledge"),
        (pl.col("promoted_item") == "event").cast(pl.Int8).alias("item_event"),
        (pl.col("promoted_item") == "other").cast(pl.Int8).alias("item_other"),

        # Income targeting features (count of income ranges targeted)
        pl.col("target_incomes").str.count_matches(r"range_").fill_null(0).alias("num_income_targets"),

        # Age targeting features (count of age ranges targeted)
        pl.col("target_ages").str.count_matches(r"range_").fill_null(0).alias("num_age_targets"),
    ])

    campaign_features.head(10)
    return (campaign_features,)


@app.cell
def _(mo):
    mo.md("""
    ### 3.5 Temporal Features
    Extract day of week, hour bucket (morning/mid-day/night), and month.
    """)
    return


@app.cell
def _(pl, placements_with_ctr):
    # Add temporal features
    def add_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
        """Add temporal features to placements DataFrame."""
        return df.with_columns([
            # Day of week (0=Monday, 6=Sunday)
            pl.col("send_datetime").dt.weekday().alias("day_of_week"),

            # Hour of day
            pl.col("send_datetime").dt.hour().alias("hour"),

            # Month
            pl.col("send_datetime").dt.month().alias("month"),

            # Year (for temporal split reference)
            pl.col("send_datetime").dt.year().alias("year"),
        ]).with_columns([
            # Hour bucket: morning (6-11), mid-day (12-17), night (18-5)
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

    placements_temporal = add_temporal_features(placements_with_ctr)
    print("Temporal features added:")
    placements_temporal.select([
        "post_id", "send_datetime", "day_of_week", "hour", "month", "hour_bucket"
    ]).head(10)
    return (add_temporal_features,)


@app.cell
def _(mo):
    mo.md("""
    ### 3.6 Embedding Norm Features
    Compute L2 norm of advertiser and publication embeddings as features.
    Using norms instead of pairwise similarity is much faster and more memory efficient.
    """)
    return


@app.cell
def _():
    # from sentence_transformers import SentenceTransformer

    # # Load sentence transformer model
    # print("Loading Sentence Transformer model...")
    # embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # # Prepare advertiser text for embeddings
    # advertiser_text = advertisers_df.with_columns(
    #     pl.concat_str([
    #         pl.col("name").fill_null(""),
    #         pl.lit(": "),
    #         pl.col("description").fill_null("")
    #     ]).alias("text_for_embedding")
    # ).select(["advertiser_id", "text_for_embedding"])

    # # Prepare publication text for embeddings
    # publication_text = publication_mets_df.join(
    #     publication_tags_df.select(["publication_id", "tags"]),
    #     on="publication_id",
    #     how="left"
    # ).with_columns(
    #     pl.concat_str([
    #         pl.col("name").fill_null(""),
    #         pl.lit(". Tags: "),
    #         pl.col("tags").fill_null("").str.replace_all(r"[\{\}']", ""),
    #         pl.lit(". "),
    #         pl.col("description").fill_null("")
    #     ]).alias("text_for_embedding")
    # ).select(["publication_id", "text_for_embedding"])

    # print(f"Advertisers to embed: {len(advertiser_text)}")
    # print(f"Publications to embed: {len(publication_text)}")

    # # Generate embeddings
    # print("\nGenerating advertiser embeddings...")
    # advertiser_texts = advertiser_text["text_for_embedding"].to_list()
    # advertiser_ids = advertiser_text["advertiser_id"].to_list()
    # advertiser_embeddings = embedding_model.encode(advertiser_texts, show_progress_bar=True)

    # print("Generating publication embeddings...")
    # publication_texts = publication_text["text_for_embedding"].to_list()
    # publication_ids_emb = publication_text["publication_id"].to_list()
    # publication_embeddings = embedding_model.encode(publication_texts, show_progress_bar=True)

    # # Compute L2 norms and create lookup dictionaries
    # # Using norm as a proxy for "information richness" of the text
    # advertiser_norm_dict = {
    #     aid: float(np.linalg.norm(emb)) 
    #     for aid, emb in zip(advertiser_ids, advertiser_embeddings)
    # }
    # publication_norm_dict = {
    #     pid: float(np.linalg.norm(emb)) 
    #     for pid, emb in zip(publication_ids_emb, publication_embeddings)
    # }

    # print(f"\nAdvertiser embedding norms computed: {len(advertiser_norm_dict)}")
    # print(f"Publication embedding norms computed: {len(publication_norm_dict)}")
    # print(f"Advertiser norm range: [{min(advertiser_norm_dict.values()):.4f}, {max(advertiser_norm_dict.values()):.4f}]")
    # print(f"Publication norm range: [{min(publication_norm_dict.values()):.4f}, {max(publication_norm_dict.values()):.4f}]")
    return


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
    pl,
    placements_with_ctr,
    publication_audience,
    split_date,
    tfidf_df,
):
    def build_feature_matrix(placements_df: pl.DataFrame, is_training: bool = True) -> pl.DataFrame:
        """
        Build the complete feature matrix for model training/prediction.
        """
        # Add temporal features
        df = add_temporal_features(placements_df)

        # Join with campaign features
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

        # Join with campaign CTR stats
        df = df.join(
            campaign_ctr_stats.select([
                "campaign_id", 
                "campaign_ctr_mean", "campaign_ctr_std", "campaign_ctr_count",
                "campaign_weighted_ctr"
            ]),
            on="campaign_id",
            how="left"
        )

        # Join with publication audience data
        df = df.join(
            publication_audience.select([
                "publication_id",
                "pub_avg_opens", "pub_std_opens", "pub_ctr_mean", "pub_placement_count"
            ]),
            on="publication_id",
            how="left"
        )

        # Join with TF-IDF features
        df = df.join(
            tfidf_df,
            on="publication_id",
            how="left"
        )

        # Fill nulls for numeric columns
        numeric_cols = [
            "campaign_ctr_mean", "campaign_ctr_std", "campaign_ctr_count", "campaign_weighted_ctr",
            "pub_avg_opens", "pub_std_opens", "pub_ctr_mean", "pub_placement_count"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(0.0))

        # Fill nulls for one-hot encoded columns
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

        # Fill TF-IDF columns with 0
        tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
        for col in tfidf_cols:
            df = df.with_columns(pl.col(col).fill_null(0.0))

        return df

    # Build feature matrices for train and test
    print("Building feature matrix for training data...")
    train_df = build_feature_matrix(
        placements_with_ctr.filter(pl.col("send_datetime") < split_date),
        is_training=True
    )

    print("Building feature matrix for test data...")
    test_df = build_feature_matrix(
        placements_with_ctr.filter(pl.col("send_datetime") >= split_date),
        is_training=False
    )

    print(f"\nTrain feature matrix shape: {train_df.shape}")
    print(f"Test feature matrix shape: {test_df.shape}")
    return test_df, train_df


@app.cell
def _(advertiser_norm_dict, pl, publication_norm_dict, test_df, train_df):
    # Add embedding norm features - simple and memory efficient
    def add_embedding_norm_features(df: pl.DataFrame) -> pl.DataFrame:
        """Add advertiser and publication embedding L2 norm features.

        The L2 norm of an embedding can capture "information richness" of the text -
        more descriptive/detailed text tends to have different norm characteristics.
        """
        # Get advertiser and publication IDs
        adv_ids = df["advertiser_id"].to_list()
        pub_ids = df["publication_id"].to_list()

        # Look up norms for each row
        adv_norms = [
            advertiser_norm_dict.get(aid, 0.0) if aid is not None else 0.0
            for aid in adv_ids
        ]
        pub_norms = [
            publication_norm_dict.get(pid, 0.0) if pid is not None else 0.0
            for pid in pub_ids
        ]

        return df.with_columns([
            pl.lit(adv_norms).alias("advertiser_emb_norm"),
            pl.lit(pub_norms).alias("publication_emb_norm"),
        ])

    print("Adding embedding norm features to training data...")
    train_df_final = train_df #add_embedding_norm_features(train_df)

    print("Adding embedding norm features to test data...")
    test_df_final = test_df #add_embedding_norm_features(test_df)

    print(f"\nFinal train shape: {train_df_final.shape}")
    print(f"Final test shape: {test_df_final.shape}")

    # Show sample of features
    train_df_final.select([
        "post_id", "ctr", #"advertiser_emb_norm", "publication_emb_norm",
        "campaign_ctr_mean", "pub_avg_opens", "hour_morning"
    ]).head(10)
    return test_df_final, train_df_final


@app.cell
def _(mo):
    mo.md("""
    ## 5. Prepare Data for XGBoost
    """)
    return


@app.cell
def _(np, test_df_final, train_df_final):
    # Define feature columns
    feature_columns = [
        # Campaign CTR stats
        "campaign_ctr_mean", "campaign_ctr_std", "campaign_ctr_count", "campaign_weighted_ctr",

        # Publication stats
        "pub_avg_opens", "pub_std_opens", "pub_ctr_mean", "pub_placement_count",

        # Target gender one-hot
        "gender_no_pref", "gender_balanced", "gender_male", "gender_female",

        # Promoted item one-hot
        "item_product", "item_service", "item_newsletter", "item_knowledge", 
        "item_event", "item_other",

        # Targeting counts
        "num_income_targets", "num_age_targets",

        # Temporal features
        "month",
        "hour_morning", "hour_midday", "hour_night",
        "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun",

        # Embedding norm features
        #"advertiser_emb_norm", "publication_emb_norm",
    ]

    # Add TF-IDF columns
    tfidf_cols = [c for c in train_df_final.columns if c.startswith("tfidf_")]
    feature_columns.extend(tfidf_cols)

    # Filter to only existing columns
    feature_columns = [c for c in feature_columns if c in train_df_final.columns]

    print(f"Total features: {len(feature_columns)}")
    print(f"Features: {feature_columns}")

    # Create numpy arrays for XGBoost
    X_train = train_df_final.select(feature_columns).to_numpy()
    y_train = train_df_final.select("ctr").to_numpy().flatten()

    X_test = test_df_final.select(feature_columns).to_numpy()
    y_test = test_df_final.select("ctr").to_numpy().flatten()

    # Extract sample weights based on approved_opens
    # Higher opens = more reliable CTR estimate = higher weight
    w_train = train_df_final.select("approved_opens").to_numpy().flatten().astype(np.float32)
    w_test = test_df_final.select("approved_opens").to_numpy().flatten().astype(np.float32)

    # Normalize weights to sum to number of samples (keeps loss scale similar)
    w_train = w_train / w_train.mean()
    w_test = w_test / w_test.mean()

    # Replace NaN/Inf with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    print(f"\nSample weights (approved_opens):")
    print(f"Train - Min: {w_train.min():.4f}, Max: {w_train.max():.4f}, Mean: {w_train.mean():.4f}")
    print(f"Test - Min: {w_test.min():.4f}, Max: {w_test.max():.4f}, Mean: {w_test.mean():.4f}")

    print(f"\nTarget (CTR) statistics:")
    print(f"Train - Mean: {y_train.mean():.6f}, Std: {y_train.std():.6f}")
    print(f"Test - Mean: {y_test.mean():.6f}, Std: {y_test.std():.6f}")
    return X_test, X_train, feature_columns, w_test, w_train, y_test, y_train


@app.cell
def _(mo):
    mo.md("""
    ## 6. Train XGBoost Model
    """)
    return


@app.cell
def _(X_train, w_train, y_train):
    import xgboost as xgb

    # Create DMatrix for XGBoost WITH sample weights
    # Weights are based on approved_opens - samples with more opens have more reliable CTR estimates
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)

    print(f"Created DMatrix with {len(y_train):,} samples and sample weights")
    print(f"Weight range: [{w_train.min():.4f}, {w_train.max():.4f}]")

    # XGBoost parameters for regression with log loss
    # Since CTR is bounded [0, 1], we use reg:squarederror and clip predictions
    # For true binomial log loss, we'd need binary classification setup
    params = {
        'objective': 'reg:squarederror',  # Regression objective
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

    # Train with early stopping using cross-validation
    # Note: CV uses the weights from dtrain automatically
    print("\nTraining XGBoost model with sample weights...")
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=500,
        nfold=5,
        early_stopping_rounds=50,
        verbose_eval=50,
        seed=42
    )

    print(f"\nBest number of rounds: {len(cv_results)}")
    print(f"Best CV MAE: {cv_results['test-mae-mean'].iloc[-1]:.6f} (+/- {cv_results['test-mae-std'].iloc[-1]:.6f})")
    print(f"Best CV RMSE: {cv_results['test-rmse-mean'].iloc[-1]:.6f} (+/- {cv_results['test-rmse-std'].iloc[-1]:.6f})")
    return dtrain, params, xgb


@app.cell
def _(dtrain, params, xgb):
    # Train final model with optimal number of rounds
    best_rounds = 200  # Using a reasonable number based on CV

    print(f"Training final model with {best_rounds} rounds...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=best_rounds,
        verbose_eval=50
    )

    print("Model training complete!")
    return (model,)


@app.cell
def _(mo):
    mo.md("""
    ## 7. Model Evaluation

    Evaluate using:
    - **Binomial Log Loss** (cross-entropy): Measures probability calibration
    - **MAE** (Mean Absolute Error): Measures average prediction error
    """)
    return


@app.cell
def _(X_test, X_train, model, np, w_test, w_train, xgb, y_test, y_train):
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Make predictions
    dtest = xgb.DMatrix(X_test)
    y_pred_test = model.predict(dtest)

    dtrain_pred = xgb.DMatrix(X_train)
    y_pred_train = model.predict(dtrain_pred)

    # Clip predictions to [0, 1] for CTR
    y_pred_test = np.clip(y_pred_test, 0.001, 0.999)
    y_pred_train = np.clip(y_pred_train, 0.001, 0.999)

    # Also clip targets for log loss calculation
    y_test_clipped = np.clip(y_test, 0.001, 0.999)
    y_train_clipped = np.clip(y_train, 0.001, 0.999)

    # Calculate unweighted metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Calculate weighted metrics (weighted by approved_opens)
    def weighted_mae(y_true, y_pred, weights):
        return np.average(np.abs(y_true - y_pred), weights=weights)

    def weighted_rmse(y_true, y_pred, weights):
        return np.sqrt(np.average((y_true - y_pred) ** 2, weights=weights))

    train_mae_weighted = weighted_mae(y_train, y_pred_train, w_train)
    test_mae_weighted = weighted_mae(y_test, y_pred_test, w_test)
    train_rmse_weighted = weighted_rmse(y_train, y_pred_train, w_train)
    test_rmse_weighted = weighted_rmse(y_test, y_pred_test, w_test)

    # Binomial Log Loss (treating CTR as probability)
    def binary_cross_entropy(y_true, y_pred, weights=None):
        """Compute binary cross-entropy for continuous targets, optionally weighted."""
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        y_true = np.clip(y_true, eps, 1 - eps)
        bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        if weights is not None:
            return np.average(bce, weights=weights)
        return np.mean(bce)

    train_logloss = binary_cross_entropy(y_train_clipped, y_pred_train)
    test_logloss = binary_cross_entropy(y_test_clipped, y_pred_test)
    train_logloss_weighted = binary_cross_entropy(y_train_clipped, y_pred_train, w_train)
    test_logloss_weighted = binary_cross_entropy(y_test_clipped, y_pred_test, w_test)

    print("=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Train':>15} {'Test':>15}")
    print("-" * 70)
    print(f"{'MAE (unweighted)':<30} {train_mae:>15.6f} {test_mae:>15.6f}")
    print(f"{'MAE (weighted by opens)':<30} {train_mae_weighted:>15.6f} {test_mae_weighted:>15.6f}")
    print(f"{'RMSE (unweighted)':<30} {train_rmse:>15.6f} {test_rmse:>15.6f}")
    print(f"{'RMSE (weighted by opens)':<30} {train_rmse_weighted:>15.6f} {test_rmse_weighted:>15.6f}")
    print(f"{'Log Loss (unweighted)':<30} {train_logloss:>15.6f} {test_logloss:>15.6f}")
    print(f"{'Log Loss (weighted by opens)':<30} {train_logloss_weighted:>15.6f} {test_logloss_weighted:>15.6f}")
    print("-" * 70)

    # Baseline comparison (predict mean)
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_mae_weighted = weighted_mae(y_test, baseline_pred, w_test)

    print(f"\nBaseline (predict mean) MAE: {baseline_mae:.6f}")
    print(f"Baseline (predict mean) Weighted MAE: {baseline_mae_weighted:.6f}")
    print(f"Improvement over baseline (unweighted): {(baseline_mae - test_mae) / baseline_mae * 100:.2f}%")
    print(f"Improvement over baseline (weighted): {(baseline_mae_weighted - test_mae_weighted) / baseline_mae_weighted * 100:.2f}%")
    return (
        test_logloss,
        test_logloss_weighted,
        test_mae,
        test_mae_weighted,
        test_rmse,
        test_rmse_weighted,
        train_logloss,
        train_logloss_weighted,
        train_mae,
        train_mae_weighted,
        train_rmse,
        train_rmse_weighted,
        y_pred_test,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 8. Feature Importance Analysis
    """)
    return


@app.cell
def _(feature_columns, model, np, plt):
    # Get feature importance
    importance_dict = model.get_score(importance_type='gain')

    # Map feature indices to names
    feature_importance = []
    for i_2, name_2 in enumerate(feature_columns):
        feat_key = f'f{i_2}'
        imp = importance_dict.get(feat_key, 0)
        feature_importance.append((name_2, imp))

    # Sort by importance
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    # Plot top 20 features
    top_n = 20
    top_features = feature_importance[:top_n]

    _fig, _ax = plt.subplots(figsize=(12, 8))
    _names = [f[0] for f in top_features]
    _values = [f[1] for f in top_features]

    _y_pos = np.arange(len(_names))
    _ax.barh(_y_pos, _values, color='steelblue', edgecolor='black')
    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels(_names)
    _ax.set_xlabel('Feature Importance (Gain)')
    _ax.set_title(f'Top {top_n} Feature Importance')
    _ax.invert_yaxis()

    plt.tight_layout()
    plt.show()

    print("\nTop 20 Features by Importance:")
    for i_2, (name_2, imp) in enumerate(feature_importance[:20], 1):
        print(f"{i_2:2d}. {name_2:<35} {imp:>10.2f}")
    return (feature_importance,)


@app.cell
def _(mo):
    mo.md("""
    ## 9. Prediction Analysis
    """)
    return


@app.cell
def _(np, plt, y_pred_test, y_test):
    # Analyze predictions vs actual
    _fig, _axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Scatter plot: Predicted vs Actual
    _ax = _axes[0, 0]
    sample_idx = np.random.choice(len(y_test), min(5000, len(y_test)), replace=False)
    _ax.scatter(y_test[sample_idx], y_pred_test[sample_idx], alpha=0.3, s=5)
    _ax.plot([0, 0.2], [0, 0.2], 'r--', linewidth=2, label='Perfect prediction')
    _ax.set_xlabel('Actual CTR')
    _ax.set_ylabel('Predicted CTR')
    _ax.set_title('Predicted vs Actual CTR (5K sample)')
    _ax.set_xlim(0, 0.2)
    _ax.set_ylim(0, 0.2)
    _ax.legend()

    # 2. Residual distribution
    _ax = _axes[0, 1]
    residuals = y_pred_test - y_test
    _ax.hist(residuals, bins=100, edgecolor='black', alpha=0.7)
    _ax.axvline(0, color='red', linestyle='--', linewidth=2)
    _ax.set_xlabel('Residual (Predicted - Actual)')
    _ax.set_ylabel('Frequency')
    _ax.set_title(f'Residual Distribution\nMean: {np.mean(residuals):.6f}, Std: {np.std(residuals):.6f}')

    # 3. Prediction distribution
    _ax = _axes[1, 0]
    _ax.hist(y_pred_test, bins=100, alpha=0.7, label='Predicted', edgecolor='black')
    _ax.hist(y_test, bins=100, alpha=0.5, label='Actual', edgecolor='black')
    _ax.set_xlabel('CTR')
    _ax.set_ylabel('Frequency')
    _ax.set_title('Distribution of Predicted vs Actual CTR')
    _ax.legend()
    _ax.set_xlim(0, 0.15)

    # 4. Error by actual CTR bin
    _ax = _axes[1, 1]
    ctr_bins = np.linspace(0, 0.1, 11)
    bin_errors = []
    bin_centers = []

    for i_3 in range(len(ctr_bins) - 1):
        mask = (y_test >= ctr_bins[i_3]) & (y_test < ctr_bins[i_3 + 1])
        if np.sum(mask) > 0:
            bin_mae = np.mean(np.abs(y_pred_test[mask] - y_test[mask]))
            bin_errors.append(bin_mae)
            bin_centers.append((ctr_bins[i_3] + ctr_bins[i_3 + 1]) / 2)

    _ax.bar(bin_centers, bin_errors, width=0.008, edgecolor='black', alpha=0.7)
    _ax.set_xlabel('Actual CTR Bin')
    _ax.set_ylabel('MAE')
    _ax.set_title('MAE by Actual CTR Range')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. Summary and Conclusions
    """)
    return


@app.cell
def _(
    feature_importance,
    mo,
    test_logloss,
    test_logloss_weighted,
    test_mae,
    test_mae_weighted,
    test_rmse,
    test_rmse_weighted,
    train_logloss,
    train_logloss_weighted,
    train_mae,
    train_mae_weighted,
    train_rmse,
    train_rmse_weighted,
):
    top_5_features = [f[0] for f in feature_importance[:5]]

    _summary = f"""
    ## Model Performance Summary

    ### Evaluation Metrics (Unweighted)

    | Metric | Train | Test |
    |--------|-------|------|
    | MAE | {train_mae:.6f} | {test_mae:.6f} |
    | RMSE | {train_rmse:.6f} | {test_rmse:.6f} |
    | Binomial Log Loss | {train_logloss:.6f} | {test_logloss:.6f} |

    ### Evaluation Metrics (Weighted by Opens)

    These metrics weight each sample by its number of opens, giving more importance to
    placements with more reliable CTR estimates.

    | Metric | Train | Test |
    |--------|-------|------|
    | Weighted MAE | {train_mae_weighted:.6f} | {test_mae_weighted:.6f} |
    | Weighted RMSE | {train_rmse_weighted:.6f} | {test_rmse_weighted:.6f} |
    | Weighted Log Loss | {train_logloss_weighted:.6f} | {test_logloss_weighted:.6f} |

    ### Top 5 Most Important Features
    1. {top_5_features[0]}
    2. {top_5_features[1]}
    3. {top_5_features[2]}
    4. {top_5_features[3]}
    5. {top_5_features[4]}

    ### Key Observations

    1. **Sample Weighting**: The model is trained with sample weights proportional to `approved_opens`,
       giving more importance to placements with larger audiences (more reliable CTR estimates).

    2. **Historical CTR features** (campaign_ctr_mean, pub_ctr_mean) are typically the most predictive,
       as past performance is a strong indicator of future performance.

    3. **Embedding norms** capture the "information richness" of advertiser and publication descriptions.
       More detailed/descriptive text may have different embedding characteristics.

    4. **Temporal features** help capture seasonality and day-of-week effects on user engagement.

    5. **TF-IDF features** from publication tags add content-based signals for understanding
       what types of content generate higher engagement.

    ### Recommendations for Improvement

    1. **Feature Engineering**: Add more interaction features (e.g., publication-campaign pair features)
    2. **Hyperparameter Tuning**: Use Bayesian optimization for better hyperparameters
    3. **Ensemble Methods**: Combine with other models (LightGBM, CatBoost)
    4. **More Temporal Features**: Add recency features, time since last campaign, etc.
    """
    mo.md(_summary)
    return


@app.cell
def _(DATA_DIR, model):
    # Save the model
    model_path = DATA_DIR / "xgboost_ctr_model.json"
    model.save_model(str(model_path))
    print(f"Model saved to: {model_path}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
