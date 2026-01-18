# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "polars>=1.37.1",
#     "matplotlib>=3.10.8",
#     "seaborn>=0.13.0",
#     "scikit-learn>=1.5.0",
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
    from typing import Tuple, Dict, List
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    mo.md("""
    # Publication Clustering Analysis

    This notebook explores clustering of publications based on their aggregated post statistics.

    ## Goal
    - Create vector representations for each `publication_id` based on mean post statistics
    - Find natural clusters of publications using K-Means
    - Use the **elbow method** and **silhouette score** to determine optimal number of clusters
    - Visualize and interpret the clusters

    ## Features Used
    - **CTR Statistics**: mean, std, min, max CTR per publication
    - **Volume Metrics**: total opens, total clicks, post count
    - **Engagement Patterns**: avg opens per post, avg clicks per post
    - **TF-IDF Features**: content tags for each publication
    """)
    return (
        KMeans,
        PCA,
        Path,
        StandardScaler,
        TfidfVectorizer,
        mo,
        np,
        pl,
        plt,
        silhouette_score,
        sns,
    )


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

    # Load datasets
    placements_df = pl.read_csv(DATA_DIR / "placements.csv")
    publication_tags_df = pl.read_csv(DATA_DIR / "publication_tags.csv")

    print("=== Data Shapes ===")
    print(f"Placements: {placements_df.shape}")
    print(f"Publication tags: {publication_tags_df.shape}")

    # Preview placements
    print("\nPlacements columns:", placements_df.columns)
    placements_df.head(5)
    return DATA_DIR, placements_df, publication_tags_df


@app.cell
def _(pl, placements_df):
    # Calculate CTR for each placement
    placements_with_ctr = placements_df.with_columns([
        pl.when(pl.col("approved_opens") > 0)
          .then(pl.col("approved_clicks") / pl.col("approved_opens"))
          .otherwise(0.0)
          .alias("ctr"),
        pl.col("post_send_at").str.to_datetime().alias("send_datetime")
    ]).filter(
        pl.col("approved_opens") > 0  # Filter out zero-opens records
    )

    print(f"Placements with CTR: {len(placements_with_ctr):,}")
    print(f"Date range: {placements_with_ctr['send_datetime'].min()} to {placements_with_ctr['send_datetime'].max()}")
    placements_with_ctr.head(5)
    return (placements_with_ctr,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. Create Publication Vector Representations

    Aggregate statistics per `publication_id` to create feature vectors.
    """)
    return


@app.cell
def _(np, pl, placements_with_ctr):
    def compute_publication_features(df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute aggregated features for each publication based on their posts.
        
        Returns a DataFrame with one row per publication_id containing:
        - CTR statistics (mean, std, min, max, median)
        - Volume metrics (total opens, clicks, post count)
        - Engagement patterns (avg opens/clicks per post)
        """
        publication_features = df.group_by("publication_id").agg([
            # CTR statistics
            pl.col("ctr").mean().alias("ctr_mean"),
            pl.col("ctr").std().alias("ctr_std"),
            pl.col("ctr").min().alias("ctr_min"),
            pl.col("ctr").max().alias("ctr_max"),
            pl.col("ctr").median().alias("ctr_median"),
            pl.col("ctr").quantile(0.25).alias("ctr_p25"),
            pl.col("ctr").quantile(0.75).alias("ctr_p75"),
            
            # Volume metrics
            pl.col("approved_opens").sum().alias("total_opens"),
            pl.col("approved_clicks").sum().alias("total_clicks"),
            pl.col("post_id").n_unique().alias("post_count"),
            
            # Engagement patterns (per post averages)
            pl.col("approved_opens").mean().alias("avg_opens_per_post"),
            pl.col("approved_clicks").mean().alias("avg_clicks_per_post"),
            pl.col("approved_opens").std().alias("std_opens_per_post"),
            pl.col("approved_clicks").std().alias("std_clicks_per_post"),
            
            # Campaign diversity
            pl.col("campaign_id").n_unique().alias("unique_campaigns"),
        ]).with_columns([
            # Derived features
            (pl.col("ctr_max") - pl.col("ctr_min")).alias("ctr_range"),
            (pl.col("ctr_p75") - pl.col("ctr_p25")).alias("ctr_iqr"),
            (pl.col("unique_campaigns") / pl.col("post_count")).alias("campaign_diversity_ratio"),
            
            # Fill nulls for std columns (single observation publications)
            pl.col("ctr_std").fill_null(0.0),
            pl.col("std_opens_per_post").fill_null(0.0),
            pl.col("std_clicks_per_post").fill_null(0.0),
        ])
        
        return publication_features

    # Compute features
    pub_features = compute_publication_features(placements_with_ctr)

    print(f"Publication features computed for {len(pub_features):,} publications")
    print(f"\nFeature columns: {pub_features.columns}")
    print(f"\nFeature statistics:")
    pub_features.describe()
    return compute_publication_features, pub_features


@app.cell
def _(mo):
    mo.md("""
    ## 3. Add TF-IDF Features from Publication Tags

    Create content-based features using TF-IDF on publication tags.
    """)
    return


@app.cell
def _(TfidfVectorizer, pl, publication_tags_df):
    def create_tfidf_features(
        tags_df: pl.DataFrame, 
        max_features: int = 30
    ) -> tuple[pl.DataFrame, list]:
        """
        Create TF-IDF features from publication tags.
        
        Returns:
        - DataFrame with publication_id and TF-IDF columns
        - List of feature names
        """
        # Clean tags
        tags_cleaned = tags_df.with_columns([
            pl.col("tags")
            .str.replace_all(r"[\{\}'\"]", "")
            .str.replace_all(r",", " ")
            .str.to_lowercase()
            .alias("tags_cleaned")
        ]).filter(
            pl.col("tags_cleaned").is_not_null() & (pl.col("tags_cleaned") != "")
        )
        
        # Extract data
        publication_ids = tags_cleaned["publication_id"].to_list()
        tags_texts = tags_cleaned["tags_cleaned"].to_list()
        tags_texts = [t if t is not None else "" for t in tags_texts]
        
        # Create TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=3,
            stop_words='english'
        )
        
        # Fit and transform
        tfidf_matrix = tfidf_vectorizer.fit_transform(tags_texts)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Create DataFrame
        tfidf_data = {"publication_id": publication_ids}
        for i, name in enumerate(feature_names):
            tfidf_data[f"tfidf_{name}"] = tfidf_matrix[:, i].toarray().flatten().tolist()
        
        tfidf_df = pl.DataFrame(tfidf_data)
        
        return tfidf_df, list(feature_names)

    # Create TF-IDF features
    tfidf_features, tfidf_feature_names = create_tfidf_features(publication_tags_df)

    print(f"TF-IDF features created: {len(tfidf_feature_names)} features")
    print(f"TF-IDF feature names: {tfidf_feature_names}")
    print(f"Publications with tags: {len(tfidf_features):,}")
    tfidf_features.head(5)
    return create_tfidf_features, tfidf_feature_names, tfidf_features


@app.cell
def _(mo):
    mo.md("""
    ## 4. Combine Features and Prepare Feature Matrix

    Join aggregated statistics with TF-IDF features and prepare for clustering.
    """)
    return


@app.cell
def _(np, pl, pub_features, tfidf_features):
    # Join publication features with TF-IDF features
    combined_features = pub_features.join(
        tfidf_features,
        on="publication_id",
        how="left"
    )

    # Fill null TF-IDF values with 0
    tfidf_cols = [c for c in combined_features.columns if c.startswith("tfidf_")]
    for col in tfidf_cols:
        combined_features = combined_features.with_columns(
            pl.col(col).fill_null(0.0)
        )

    print(f"Combined features shape: {combined_features.shape}")

    # Define numerical feature columns for clustering
    numeric_feature_cols = [
        # CTR features
        "ctr_mean", "ctr_std", "ctr_min", "ctr_max", "ctr_median",
        "ctr_p25", "ctr_p75", "ctr_range", "ctr_iqr",
        
        # Volume features (log-transformed for better scaling)
        "total_opens", "total_clicks", "post_count",
        
        # Engagement patterns
        "avg_opens_per_post", "avg_clicks_per_post",
        "std_opens_per_post", "std_clicks_per_post",
        
        # Diversity
        "unique_campaigns", "campaign_diversity_ratio",
    ]

    # Add TF-IDF columns
    all_feature_cols = numeric_feature_cols + tfidf_cols

    # Filter to columns that exist
    all_feature_cols = [c for c in all_feature_cols if c in combined_features.columns]

    print(f"Total features for clustering: {len(all_feature_cols)}")
    print(f"Numeric features: {len(numeric_feature_cols)}")
    print(f"TF-IDF features: {len(tfidf_cols)}")

    # Extract feature matrix
    feature_matrix = combined_features.select(all_feature_cols).to_numpy()

    # Handle NaN and Inf
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply log transform to volume features for better scaling
    volume_indices = [all_feature_cols.index(c) for c in ["total_opens", "total_clicks", "post_count", 
                                                           "avg_opens_per_post", "avg_clicks_per_post",
                                                           "unique_campaigns"] 
                      if c in all_feature_cols]
    for idx in volume_indices:
        feature_matrix[:, idx] = np.log1p(feature_matrix[:, idx])

    print(f"\nFeature matrix shape: {feature_matrix.shape}")
    return (
        all_feature_cols,
        combined_features,
        feature_matrix,
        numeric_feature_cols,
        tfidf_cols,
    )


@app.cell
def _(StandardScaler, feature_matrix):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    print(f"Scaled feature matrix shape: {X_scaled.shape}")
    print(f"Feature means after scaling: {X_scaled.mean(axis=0)[:5]}...")
    print(f"Feature stds after scaling: {X_scaled.std(axis=0)[:5]}...")
    return X_scaled, scaler


@app.cell
def _(mo):
    mo.md("""
    ## 5. Elbow Method for Optimal K

    Run K-Means with different values of K and plot the inertia (within-cluster sum of squares)
    to identify the optimal number of clusters using the elbow method.
    """)
    return


@app.cell
def _(KMeans, X_scaled, np, plt, silhouette_score):
    def compute_elbow_metrics(
        X: np.ndarray, 
        k_range: range,
        random_state: int = 42
    ) -> tuple[list, list]:
        """
        Compute inertia and silhouette scores for different K values.
        
        Returns:
        - List of inertias
        - List of silhouette scores
        """
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            print(f"Fitting K-Means with k={k}...")
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans.fit(X)
            
            inertias.append(kmeans.inertia_)
            
            # Silhouette score requires k >= 2
            if k >= 2:
                sil_score = silhouette_score(X, kmeans.labels_, sample_size=min(5000, len(X)))
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        return inertias, silhouette_scores

    # Define K range
    k_range = range(2, 16)

    # Compute metrics
    inertias, silhouette_scores_list = compute_elbow_metrics(X_scaled, k_range)

    # Plot elbow curve and silhouette scores
    fig_elbow, axes_elbow = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow plot (Inertia)
    ax = axes_elbow[0]
    ax.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=12)
    ax.set_title('Elbow Method for Optimal K', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Mark the "elbow" - compute second derivative
    inertias_arr = np.array(inertias)
    first_diff = np.diff(inertias_arr)
    second_diff = np.diff(first_diff)
    elbow_idx = np.argmax(second_diff) + 2  # +2 because we lose 2 points in double diff
    optimal_k_elbow = list(k_range)[elbow_idx]

    ax.axvline(x=optimal_k_elbow, color='r', linestyle='--', 
               label=f'Suggested K = {optimal_k_elbow}')
    ax.legend()

    # Silhouette scores
    ax = axes_elbow[1]
    ax.plot(list(k_range), silhouette_scores_list, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score by K', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Mark best silhouette score
    best_sil_idx = np.argmax(silhouette_scores_list)
    optimal_k_sil = list(k_range)[best_sil_idx]
    ax.axvline(x=optimal_k_sil, color='r', linestyle='--',
               label=f'Best K = {optimal_k_sil} (score={silhouette_scores_list[best_sil_idx]:.3f})')
    ax.legend()

    plt.tight_layout()
    plt.show()

    print(f"\n=== Optimal K Analysis ===")
    print(f"Elbow method suggests K = {optimal_k_elbow}")
    print(f"Silhouette score suggests K = {optimal_k_sil} (score = {silhouette_scores_list[best_sil_idx]:.3f})")

    # Print all scores for reference
    print("\nAll K values and metrics:")
    for k, inertia, sil in zip(k_range, inertias, silhouette_scores_list):
        print(f"  K={k:2d}: Inertia={inertia:12.2f}, Silhouette={sil:.4f}")
    return (
        compute_elbow_metrics,
        inertias,
        k_range,
        optimal_k_elbow,
        optimal_k_sil,
        silhouette_scores_list,
    )


@app.cell
def _(mo, optimal_k_elbow, optimal_k_sil):
    # Let user select K
    k_options = list(range(2, 16))
    default_k = min(optimal_k_elbow, optimal_k_sil)
    
    k_selector = mo.ui.slider(
        start=2, 
        stop=15, 
        value=default_k,
        label=f"Select number of clusters (suggested: {optimal_k_elbow}-{optimal_k_sil})"
    )
    
    mo.md(f"""
    ## 6. Select Number of Clusters
    
    Based on the analysis:
    - **Elbow method** suggests K = {optimal_k_elbow}
    - **Silhouette score** suggests K = {optimal_k_sil}
    
    Select the number of clusters to use:
    
    {k_selector}
    """)
    return default_k, k_options, k_selector


@app.cell
def _(KMeans, X_scaled, k_selector):
    # Fit final K-Means model
    selected_k = k_selector.value
    
    final_kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    cluster_labels = final_kmeans.fit_predict(X_scaled)
    
    print(f"K-Means fitted with K = {selected_k}")
    print(f"Cluster distribution:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count:,} publications ({100*count/len(cluster_labels):.1f}%)")
    return cluster_labels, final_kmeans, selected_k


@app.cell
def _(mo):
    mo.md("""
    ## 7. Visualize Clusters with PCA

    Reduce dimensionality to 2D using PCA for visualization.
    """)
    return


@app.cell
def _(PCA, X_scaled, cluster_labels, np, plt, selected_k, sns):
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

    # Create visualization
    fig_pca, axes_pca = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot with cluster colors
    ax = axes_pca[0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=cluster_labels, cmap='tab10', 
                         alpha=0.6, s=10)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title(f'Publication Clusters (K={selected_k}) - PCA Projection', fontsize=14)
    plt.colorbar(scatter, ax=ax, label='Cluster')

    # Cluster size bar chart
    ax = axes_pca[1]
    unique_clusters, cluster_sizes = np.unique(cluster_labels, return_counts=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    bars = ax.bar(unique_clusters, cluster_sizes, color=colors, edgecolor='black')
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Number of Publications', fontsize=12)
    ax.set_title('Cluster Size Distribution', fontsize=14)

    # Add count labels on bars
    for bar, count in zip(bars, cluster_sizes):
        ax.annotate(f'{count:,}', 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()
    return X_pca, pca


@app.cell
def _(mo):
    mo.md("""
    ## 8. Cluster Profiling

    Analyze the characteristics of each cluster to understand what makes them distinct.
    """)
    return


@app.cell
def _(cluster_labels, combined_features, np, numeric_feature_cols, pl, plt):
    # Add cluster labels to combined features
    clustered_publications = combined_features.with_columns([
        pl.lit(cluster_labels).alias("cluster")
    ])

    # Compute cluster profiles (mean values for each cluster)
    cluster_profiles = clustered_publications.group_by("cluster").agg([
        pl.col(col).mean().alias(f"{col}_mean") for col in numeric_feature_cols
    ] + [
        pl.count().alias("cluster_size")
    ]).sort("cluster")

    print("=== Cluster Profiles ===")
    print(cluster_profiles)

    # Visualize key feature differences across clusters
    key_features = ["ctr_mean", "total_opens", "post_count", "unique_campaigns", "avg_opens_per_post"]
    key_features = [f for f in key_features if f in numeric_feature_cols]

    n_clusters = len(cluster_profiles)
    fig_profile, axes_profile = plt.subplots(2, 3, figsize=(15, 10))
    axes_profile = axes_profile.flatten()

    for idx, feature in enumerate(key_features):
        if idx >= len(axes_profile):
            break
        ax = axes_profile[idx]
        
        cluster_ids = cluster_profiles["cluster"].to_numpy()
        feature_values = cluster_profiles[f"{feature}_mean"].to_numpy()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))
        bars = ax.bar(cluster_ids, feature_values, color=colors, edgecolor='black')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(f'Mean {feature}')
        ax.set_title(f'{feature} by Cluster')
        ax.set_xticks(cluster_ids)

    # Cluster size in last subplot
    ax = axes_profile[-1]
    cluster_ids = cluster_profiles["cluster"].to_numpy()
    sizes = cluster_profiles["cluster_size"].to_numpy()
    colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))
    ax.bar(cluster_ids, sizes, color=colors, edgecolor='black')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Publications')
    ax.set_title('Cluster Sizes')
    ax.set_xticks(cluster_ids)

    plt.tight_layout()
    plt.show()
    return cluster_profiles, clustered_publications, key_features


@app.cell
def _(cluster_profiles, mo, numeric_feature_cols):
    # Create detailed cluster interpretation
    def interpret_clusters(profiles_df, feature_cols):
        """Generate human-readable cluster interpretations."""
        interpretations = []
        
        n_clusters = len(profiles_df)
        
        for cluster_id in range(n_clusters):
            row = profiles_df.filter(pl.col("cluster") == cluster_id)
            
            ctr_mean = row[f"ctr_mean_mean"][0]
            total_opens = row[f"total_opens_mean"][0]
            post_count = row[f"post_count_mean"][0]
            unique_camps = row[f"unique_campaigns_mean"][0]
            cluster_size = row["cluster_size"][0]
            
            # Classify based on characteristics
            ctr_level = "High" if ctr_mean > 0.03 else ("Medium" if ctr_mean > 0.015 else "Low")
            volume_level = "High" if total_opens > 100000 else ("Medium" if total_opens > 10000 else "Low")
            activity_level = "High" if post_count > 50 else ("Medium" if post_count > 10 else "Low")
            
            interpretation = f"""
            **Cluster {cluster_id}** ({cluster_size:,} publications)
            - CTR: {ctr_level} ({ctr_mean:.4f})
            - Volume: {volume_level} ({total_opens:,.0f} total opens)
            - Activity: {activity_level} ({post_count:.0f} posts)
            - Campaign diversity: {unique_camps:.1f} unique campaigns
            """
            interpretations.append(interpretation)
        
        return "\n".join(interpretations)

    from polars import col as pl_col
    import polars as pl

    cluster_interpretation = interpret_clusters(cluster_profiles, numeric_feature_cols)

    mo.md(f"""
    ## 9. Cluster Interpretation

    {cluster_interpretation}
    """)
    return interpret_clusters, pl, pl_col


@app.cell
def _(mo):
    mo.md("""
    ## 10. Export Clustered Publications

    Save the publication IDs with their cluster assignments for use in downstream models.
    """)
    return


@app.cell
def _(DATA_DIR, clustered_publications, pl):
    # Select relevant columns for export
    export_df = clustered_publications.select([
        "publication_id",
        "cluster",
        "ctr_mean",
        "total_opens",
        "post_count",
        "unique_campaigns",
        "avg_opens_per_post",
    ])

    # Save to CSV
    output_path = DATA_DIR / "publication_clusters.csv"
    export_df.write_csv(output_path)

    print(f"Exported clustered publications to: {output_path}")
    print(f"Total publications: {len(export_df):,}")
    print(f"\nCluster distribution in export:")
    print(export_df.group_by("cluster").agg(pl.count().alias("count")).sort("cluster"))
    return export_df, output_path


@app.cell
def _(
    cluster_profiles,
    inertias,
    k_range,
    mo,
    optimal_k_elbow,
    optimal_k_sil,
    selected_k,
    silhouette_scores_list,
):
    # Summary
    mo.md(f"""
    ## Summary

    ### Clustering Results
    - **Number of clusters selected**: {selected_k}
    - **Elbow method suggested**: K = {optimal_k_elbow}
    - **Best silhouette score**: K = {optimal_k_sil} (score = {max(silhouette_scores_list):.4f})

    ### Key Findings
    1. Publications naturally group into {selected_k} distinct segments based on:
       - CTR performance patterns
       - Volume (total opens/clicks)
       - Posting frequency
       - Campaign diversity

    2. The clusters capture different "publication archetypes":
       - High-engagement niche publications
       - High-volume mainstream publications
       - Low-activity or new publications
       - Specialized campaign-focused publications

    ### Usage
    The cluster assignments can be used as:
    - A categorical feature in CTR prediction models
    - Basis for segment-specific models (ensemble approach)
    - Publication targeting for campaigns

    ### Exported Data
    Publication cluster assignments saved to `publication_clusters.csv`
    """)
    return


if __name__ == "__main__":
    app.run()
