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
    """Import all dependencies and configure plotting."""
    import marimo as mo
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from typing import Dict
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    plt.style.use('seaborn-v0_8-whitegrid')

    mo.md("""
    # Publication Clustering Analysis

    This notebook explores clustering of publications based on their aggregated post embeddings.

    ## Goal
    - Create vector representations for each `publication_id` based on **mean post embeddings**
    - Find natural clusters of publications using K-Means
    - Use the **elbow method** and **silhouette score** to determine optimal number of clusters
    - Visualize and interpret the clusters

    ## Features Used
    - **Mean Post Embeddings**: 1024-dimensional vectors from `post_embeddings.csv`
    - Embeddings are computed using Qwen3-Embedding-0.6B model
    - Memory-efficient chunked processing for the 10GB+ embeddings file
    """)
    return (
        Dict,
        KMeans,
        PCA,
        Path,
        StandardScaler,
        mo,
        np,
        pl,
        plt,
        silhouette_score,
    )


@app.cell
def _(Dict, np, pl):
    """Define data loading and embedding processing functions."""

    def load_placements(data_dir) -> pl.DataFrame:
        """Load placements dataset with CTR calculation."""
        placements_df = pl.read_csv(data_dir / "placements.csv")
        return placements_df.with_columns([
            pl.when(pl.col("approved_opens") > 0)
              .then(pl.col("approved_clicks") / pl.col("approved_opens"))
              .otherwise(0.0)
              .alias("ctr"),
            pl.col("post_send_at").str.to_datetime().alias("send_datetime")
        ]).filter(pl.col("approved_opens") > 0)

    def parse_embedding_string(embedding_str: str) -> np.ndarray:
        """Parse PostgreSQL array string format {val1,val2,...} to numpy array."""
        values_str = embedding_str.strip('{}')
        return np.array([float(x) for x in values_str.split(',')])

    def compute_publication_embeddings_chunked(
        embeddings_path,
        chunk_size: int = 50_000,
        embedding_dim: int = 1024
    ) -> Dict[str, np.ndarray]:
        """
        Compute mean embeddings per publication_id by processing file in chunks.

        Memory-efficient approach:
        1. Read file in chunks
        2. Accumulate sum of embeddings and count per publication_id
        3. Compute mean at the end

        Args:
            embeddings_path: Path to post_embeddings.csv
            chunk_size: Number of rows to process at a time
            embedding_dim: Dimension of embedding vectors (1024 for Qwen3)

        Returns:
            Dictionary mapping publication_id to mean embedding vector
        """
        # Accumulators for running mean computation
        publication_sums: Dict[str, np.ndarray] = {}
        publication_counts: Dict[str, int] = {}

        print(f"Processing embeddings file in chunks of {chunk_size:,} rows...")

        # Use polars lazy reader with batching for memory efficiency
        reader = pl.read_csv_batched(
            embeddings_path,
            batch_size=chunk_size,
        )

        chunk_num = 0
        total_rows = 0

        while True:
            batches = reader.next_batches(1)
            if not batches:
                break

            chunk_df = batches[0]
            chunk_num += 1
            total_rows += len(chunk_df)

            if chunk_num % 10 == 1:
                print(f"  Processing chunk {chunk_num}, total rows so far: {total_rows:,}")

            # Process each row in the chunk
            pub_ids = chunk_df["publication_id"].to_list()
            embeddings = chunk_df["embedding"].to_list()

            for pub_id, emb_str in zip(pub_ids, embeddings):
                if pub_id is None or emb_str is None:
                    continue

                try:
                    embedding = parse_embedding_string(emb_str)

                    if pub_id not in publication_sums:
                        publication_sums[pub_id] = np.zeros(embedding_dim, dtype=np.float64)
                        publication_counts[pub_id] = 0

                    publication_sums[pub_id] += embedding
                    publication_counts[pub_id] += 1
                except (ValueError, AttributeError):
                    continue

        print(f"  Total rows processed: {total_rows:,}")
        print(f"  Unique publications: {len(publication_sums):,}")

        # Compute means
        publication_embeddings = {
            pub_id: pub_sum / publication_counts[pub_id]
            for pub_id, pub_sum in publication_sums.items()
        }

        return publication_embeddings

    def embeddings_dict_to_dataframe(
        embeddings_dict: Dict[str, np.ndarray],
        prefix: str = "emb_"
    ) -> pl.DataFrame:
        """Convert embeddings dictionary to polars DataFrame."""
        pub_ids = list(embeddings_dict.keys())
        embedding_matrix = np.array([embeddings_dict[pid] for pid in pub_ids])

        # Create DataFrame with publication_id and embedding columns
        data = {"publication_id": pub_ids}
        for i in range(embedding_matrix.shape[1]):
            data[f"{prefix}{i}"] = embedding_matrix[:, i].tolist()

        return pl.DataFrame(data)
    return (
        compute_publication_embeddings_chunked,
        embeddings_dict_to_dataframe,
        load_placements,
    )


@app.cell
def _(np, pl):
    """Define publication statistics computation functions."""

    def compute_publication_stats(df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute aggregated statistics for each publication.

        Returns DataFrame with:
        - CTR statistics (mean, std)
        - Volume metrics (total opens, clicks, post count)
        - Engagement patterns
        """
        return df.group_by("publication_id").agg([
            # CTR statistics
            pl.col("ctr").mean().alias("ctr_mean"),
            pl.col("ctr").std().alias("ctr_std"),

            # Volume metrics
            pl.col("approved_opens").sum().alias("total_opens"),
            pl.col("approved_clicks").sum().alias("total_clicks"),
            pl.col("post_id").n_unique().alias("post_count"),

            # Engagement patterns
            pl.col("approved_opens").mean().alias("avg_opens_per_post"),
            pl.col("campaign_id").n_unique().alias("unique_campaigns"),
        ]).with_columns([
            pl.col("ctr_std").fill_null(0.0),
        ])

    def combine_stats_and_embeddings(
        stats_df: pl.DataFrame,
        embeddings_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Join publication statistics with embeddings."""
        combined = stats_df.join(embeddings_df, on="publication_id", how="inner")

        # Fill any null embedding values with 0
        emb_cols = [c for c in combined.columns if c.startswith("emb_")]
        for col in emb_cols:
            combined = combined.with_columns(pl.col(col).fill_null(0.0))

        return combined

    def prepare_feature_matrix(
        combined_df: pl.DataFrame,
        use_stats: bool = True
    ) -> tuple[np.ndarray, list, pl.DataFrame]:
        """
        Prepare feature matrix for clustering.

        Args:
            combined_df: DataFrame with stats and embeddings
            use_stats: Whether to include publication stats in features

        Returns:
            - Feature matrix as numpy array
            - List of feature column names
            - Filtered DataFrame (matching rows)
        """
        emb_cols = [c for c in combined_df.columns if c.startswith("emb_")]

        if use_stats:
            stat_cols = ["ctr_mean", "ctr_std", "avg_opens_per_post"]
            # Log transform volume features
            combined_df = combined_df.with_columns([
                pl.col("total_opens").log1p().alias("log_total_opens"),
                pl.col("post_count").log1p().alias("log_post_count"),
                pl.col("unique_campaigns").log1p().alias("log_unique_campaigns"),
            ])
            stat_cols += ["log_total_opens", "log_post_count", "log_unique_campaigns"]
            feature_cols = stat_cols + emb_cols
        else:
            feature_cols = emb_cols

        feature_cols = [c for c in feature_cols if c in combined_df.columns]

        feature_matrix = combined_df.select(feature_cols).to_numpy()
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_matrix, feature_cols, combined_df
    return (
        combine_stats_and_embeddings,
        compute_publication_stats,
        prepare_feature_matrix,
    )


@app.cell
def _(KMeans, np, silhouette_score):
    """Define clustering utility functions."""

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
        silhouette_scores_list = []

        for k in k_range:
            print(f"Fitting K-Means with k={k}...")
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans.fit(X)

            inertias.append(kmeans.inertia_)

            if k >= 2:
                sil_score = silhouette_score(
                    X, kmeans.labels_,
                    sample_size=min(5000, len(X))
                )
                silhouette_scores_list.append(sil_score)
            else:
                silhouette_scores_list.append(0)

        return inertias, silhouette_scores_list

    def find_optimal_k(
        inertias: list,
        silhouette_scores_list: list,
        k_range: range
    ) -> tuple[int, int]:
        """
        Find optimal K using elbow method and silhouette score.

        Returns:
        - Optimal K from elbow method
        - Optimal K from silhouette score
        """
        inertias_arr = np.array(inertias)
        first_diff = np.diff(inertias_arr)
        second_diff = np.diff(first_diff)
        elbow_idx = np.argmax(second_diff) + 2
        optimal_k_elbow = list(k_range)[min(elbow_idx, len(k_range) - 1)]

        best_sil_idx = np.argmax(silhouette_scores_list)
        optimal_k_sil = list(k_range)[best_sil_idx]

        return optimal_k_elbow, optimal_k_sil

    def fit_kmeans(
        X: np.ndarray,
        n_clusters: int,
        random_state: int = 42
    ) -> tuple[KMeans, np.ndarray]:
        """Fit K-Means model and return model and labels."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        return kmeans, labels

    def compute_cluster_profiles(
        combined_df,
        cluster_labels: np.ndarray,
        stat_cols: list,
        pl_module
    ):
        """Compute cluster profiles with mean values for each feature."""
        clustered = combined_df.with_columns([
            pl_module.lit(cluster_labels).alias("cluster")
        ])

        profiles = clustered.group_by("cluster").agg([
            pl_module.col(col).mean().alias(f"{col}_mean") for col in stat_cols
        ] + [
            pl_module.count().alias("cluster_size")
        ]).sort("cluster")

        return clustered, profiles
    return (
        compute_cluster_profiles,
        compute_elbow_metrics,
        find_optimal_k,
        fit_kmeans,
    )


@app.cell
def _(np, plt):
    """Define visualization functions."""

    def plot_elbow_analysis(
        k_range: range,
        inertias: list,
        silhouette_scores_list: list,
        optimal_k_elbow: int,
        optimal_k_sil: int
    ):
        """Plot elbow curve and silhouette scores."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Elbow plot
        ax = axes[0]
        ax.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax.set_ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=12)
        ax.set_title('Elbow Method for Optimal K', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=optimal_k_elbow, color='r', linestyle='--',
                   label=f'Suggested K = {optimal_k_elbow}')
        ax.legend()

        # Silhouette plot
        ax = axes[1]
        ax.plot(list(k_range), silhouette_scores_list, 'go-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title('Silhouette Score by K', fontsize=14)
        ax.grid(True, alpha=0.3)
        best_sil_idx = np.argmax(silhouette_scores_list)
        ax.axvline(x=optimal_k_sil, color='r', linestyle='--',
                   label=f'Best K = {optimal_k_sil} (score={silhouette_scores_list[best_sil_idx]:.3f})')
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_pca_clusters(
        X_pca: np.ndarray,
        cluster_labels: np.ndarray,
        pca_variance_ratio: tuple,
        selected_k: int
    ):
        """Plot PCA projection of clusters and cluster size distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Scatter plot
        ax = axes[0]
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                            c=cluster_labels, cmap='tab10',
                            alpha=0.6, s=10)
        ax.set_xlabel(f'PC1 ({pca_variance_ratio[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca_variance_ratio[1]:.1%} variance)', fontsize=12)
        ax.set_title(f'Publication Clusters (K={selected_k}) - PCA Projection', fontsize=14)
        plt.colorbar(scatter, ax=ax, label='Cluster')

        # Cluster size bar chart
        ax = axes[1]
        unique_clusters, cluster_sizes = np.unique(cluster_labels, return_counts=True)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        bars = ax.bar(unique_clusters, cluster_sizes, color=colors, edgecolor='black')
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Number of Publications', fontsize=12)
        ax.set_title('Cluster Size Distribution', fontsize=14)

        for bar, count in zip(bars, cluster_sizes):
            ax.annotate(f'{count:,}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        return fig

    def plot_cluster_profiles(cluster_profiles, key_features: list):
        """Plot feature profiles for each cluster."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, feature in enumerate(key_features):
            if idx >= len(axes) - 1:
                break

            col_name = f"{feature}_mean"
            if col_name not in cluster_profiles.columns:
                continue

            ax = axes[idx]
            cluster_ids = cluster_profiles["cluster"].to_numpy()
            feature_values = cluster_profiles[col_name].to_numpy()

            colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))
            ax.bar(cluster_ids, feature_values, color=colors, edgecolor='black')
            ax.set_xlabel('Cluster')
            ax.set_ylabel(f'Mean {feature}')
            ax.set_title(f'{feature} by Cluster')
            ax.set_xticks(cluster_ids)

        # Cluster size in last subplot
        ax = axes[-1]
        cluster_ids = cluster_profiles["cluster"].to_numpy()
        sizes = cluster_profiles["cluster_size"].to_numpy()
        colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))
        ax.bar(cluster_ids, sizes, color=colors, edgecolor='black')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Publications')
        ax.set_title('Cluster Sizes')
        ax.set_xticks(cluster_ids)

        plt.tight_layout()
        return fig
    return plot_cluster_profiles, plot_elbow_analysis, plot_pca_clusters


@app.cell
def _(pl):
    """Define interpretation and export functions."""

    def interpret_clusters(profiles_df, n_clusters: int) -> str:
        """Generate human-readable cluster interpretations."""
        interpretations = []

        for cluster_id in range(n_clusters):
            row = profiles_df.filter(pl.col("cluster") == cluster_id)

            if len(row) == 0:
                continue

            ctr_mean = row["ctr_mean_mean"][0]
            total_opens = row["total_opens_mean"][0]
            post_count = row["post_count_mean"][0]
            unique_camps = row["unique_campaigns_mean"][0]
            cluster_size = row["cluster_size"][0]

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

    def export_clustered_publications(clustered_df, output_path, export_cols: list):
        """Export clustered publications to CSV."""
        export_df = clustered_df.select(export_cols)
        export_df.write_csv(output_path)
        return export_df
    return export_clustered_publications, interpret_clusters


@app.cell
def _(mo):
    mo.md("""
    ## 1. Data Loading
    """)
    return


@app.cell
def _(Path, load_placements):
    """Load placements data."""
    DATA_DIR = Path("../data")

    placements_with_ctr = load_placements(DATA_DIR)

    print(f"Placements with CTR: {len(placements_with_ctr):,}")
    print(f"Date range: {placements_with_ctr['send_datetime'].min()} to {placements_with_ctr['send_datetime'].max()}")

    placements_with_ctr.head(5)
    return DATA_DIR, placements_with_ctr


@app.cell
def _(mo):
    mo.md("""
    ## 2. Compute Publication Mean Embeddings

    Processing the 10GB+ `post_embeddings.csv` file in chunks to compute mean embedding
    vectors for each publication. This is memory-efficient and avoids loading the entire file.
    """)
    return


@app.cell
def _(
    DATA_DIR,
    compute_publication_embeddings_chunked,
    embeddings_dict_to_dataframe,
):
    """Compute mean embeddings per publication using chunked processing."""
    embeddings_path = DATA_DIR / "post_embeddings.csv"

    # Process embeddings in chunks (memory-efficient)
    publication_embeddings = compute_publication_embeddings_chunked(
        embeddings_path,
        chunk_size=50_000,
        embedding_dim=1024
    )

    # Convert to DataFrame
    embeddings_df = embeddings_dict_to_dataframe(publication_embeddings, prefix="emb_")

    print(f"\nEmbeddings DataFrame shape: {embeddings_df.shape}")
    print(f"Embedding dimension: {len([c for c in embeddings_df.columns if c.startswith('emb_')])}")

    embeddings_df.head(3)
    return (embeddings_df,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Compute Publication Statistics and Combine with Embeddings
    """)
    return


@app.cell
def _(
    combine_stats_and_embeddings,
    compute_publication_stats,
    embeddings_df,
    placements_with_ctr,
):
    """Compute publication stats and combine with embeddings."""
    # Compute publication statistics
    pub_stats = compute_publication_stats(placements_with_ctr)
    print(f"Publication stats computed for {len(pub_stats):,} publications")

    # Combine with embeddings (inner join - only publications with embeddings)
    combined_features = combine_stats_and_embeddings(pub_stats, embeddings_df)
    print(f"Combined features shape: {combined_features.shape}")
    print(f"Publications with both stats and embeddings: {len(combined_features):,}")

    combined_features.select([
        "publication_id", "ctr_mean", "total_opens", "post_count", "emb_0", "emb_1"
    ]).head(5)
    return (combined_features,)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Prepare Feature Matrix
    """)
    return


@app.cell
def _(combined_features, prepare_feature_matrix):
    """Prepare feature matrix for clustering."""
    feature_matrix, feature_cols, combined_df = prepare_feature_matrix(
        combined_features,
        use_stats=True
    )

    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Total features: {len(feature_cols)}")
    print(f"  - Stat features: {len([c for c in feature_cols if not c.startswith('emb_')])}")
    print(f"  - Embedding features: {len([c for c in feature_cols if c.startswith('emb_')])}")

    # Define stat columns for profiling
    STAT_COLS = ["ctr_mean", "ctr_std", "total_opens", "post_count", "avg_opens_per_post", "unique_campaigns"]
    return STAT_COLS, combined_df, feature_matrix


@app.cell
def _(StandardScaler, feature_matrix):
    """Standardize features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    print(f"Scaled feature matrix shape: {X_scaled.shape}")
    print(f"Feature means after scaling (first 5): {X_scaled.mean(axis=0)[:5]}")
    print(f"Feature stds after scaling (first 5): {X_scaled.std(axis=0)[:5]}")
    return (X_scaled,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Elbow Method for Optimal K
    """)
    return


@app.cell
def _(
    X_scaled,
    compute_elbow_metrics,
    find_optimal_k,
    plot_elbow_analysis,
    plt,
):
    """Run elbow analysis to find optimal K."""
    K_RANGE = range(2, 16)

    inertias, silhouette_scores_list = compute_elbow_metrics(X_scaled, K_RANGE)
    optimal_k_elbow, optimal_k_sil = find_optimal_k(inertias, silhouette_scores_list, K_RANGE)

    print(f"\n=== Optimal K Analysis ===")
    print(f"Elbow method suggests K = {optimal_k_elbow}")
    print(f"Silhouette score suggests K = {optimal_k_sil}")

    # Print all scores
    print("\nAll K values and metrics:")
    for k, inertia, sil in zip(K_RANGE, inertias, silhouette_scores_list):
        print(f"  K={k:2d}: Inertia={inertia:12.2f}, Silhouette={sil:.4f}")

    # Plot
    fig_2 = plot_elbow_analysis(K_RANGE, inertias, silhouette_scores_list, optimal_k_elbow, optimal_k_sil)
    plt.show()
    return optimal_k_elbow, optimal_k_sil, silhouette_scores_list


@app.cell
def _(mo, optimal_k_elbow, optimal_k_sil):
    """Create K selector UI."""
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
    return (k_selector,)


@app.cell
def _(X_scaled, fit_kmeans, k_selector, np):
    """Fit final K-Means model."""
    selected_k = k_selector.value

    final_kmeans, cluster_labels = fit_kmeans(X_scaled, selected_k)

    print(f"K-Means fitted with K = {selected_k}")
    print(f"Cluster distribution:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count:,} publications ({100*count/len(cluster_labels):.1f}%)")
    return cluster_labels, selected_k


@app.cell
def _(mo):
    mo.md("""
    ## 7. Visualize Clusters with PCA
    """)
    return


@app.cell
def _(PCA, X_scaled, cluster_labels, plot_pca_clusters, plt, selected_k):
    """Apply PCA and visualize clusters."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

    fig_3 = plot_pca_clusters(X_pca, cluster_labels, pca.explained_variance_ratio_, selected_k)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Cluster Profiling
    """)
    return


@app.cell
def _(
    STAT_COLS,
    cluster_labels,
    combined_df,
    compute_cluster_profiles,
    pl,
    plot_cluster_profiles,
    plt,
):
    """Compute and visualize cluster profiles."""
    KEY_FEATURES = ["ctr_mean", "total_opens", "post_count", "unique_campaigns", "avg_opens_per_post"]

    clustered_publications, cluster_profiles = compute_cluster_profiles(
        combined_df,
        cluster_labels,
        STAT_COLS,
        pl
    )

    print("=== Cluster Profiles ===")
    print(cluster_profiles)

    fig = plot_cluster_profiles(cluster_profiles, KEY_FEATURES)
    plt.show()
    return cluster_profiles, clustered_publications


@app.cell
def _(cluster_profiles, interpret_clusters, mo, selected_k):
    """Generate cluster interpretation."""
    cluster_interpretation = interpret_clusters(cluster_profiles, selected_k)

    mo.md(f"""
    ## 9. Cluster Interpretation

    {cluster_interpretation}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. Export Clustered Publications
    """)
    return


@app.cell
def _(DATA_DIR, clustered_publications, export_clustered_publications, pl):
    """Export clustered publications."""
    EXPORT_COLS = [
        "publication_id",
        "cluster",
        "ctr_mean",
        "total_opens",
        "post_count",
        "unique_campaigns",
        "avg_opens_per_post",
    ]

    output_path = DATA_DIR / "publication_clusters.csv"
    export_df = export_clustered_publications(clustered_publications, output_path, EXPORT_COLS)

    print(f"Exported clustered publications to: {output_path}")
    print(f"Total publications: {len(export_df):,}")
    print(f"\nCluster distribution in export:")
    print(export_df.group_by("cluster").agg(pl.count().alias("count")).sort("cluster"))
    return


@app.cell
def _(mo, optimal_k_elbow, optimal_k_sil, selected_k, silhouette_scores_list):
    """Display summary."""
    mo.md(f"""
    ## Summary

    ### Clustering Approach
    - **Feature representation**: Mean of 1024-dimensional post embeddings per publication
    - **Embedding model**: Qwen3-Embedding-0.6B
    - **Memory-efficient processing**: Chunked reading of 10GB+ embeddings file

    ### Clustering Results
    - **Number of clusters selected**: {selected_k}
    - **Elbow method suggested**: K = {optimal_k_elbow}
    - **Best silhouette score**: K = {optimal_k_sil} (score = {max(silhouette_scores_list):.4f})

    ### Key Findings
    1. Publications naturally group into {selected_k} distinct segments based on:
       - Content semantics (via embeddings)
       - CTR performance patterns
       - Volume (total opens/clicks)
       - Posting frequency

    2. The clusters capture different "publication archetypes" based on content similarity
       and engagement patterns.

    ### Usage
    The cluster assignments can be used as:
    - A categorical feature in CTR prediction models
    - Basis for segment-specific models (ensemble approach)
    - Publication targeting for campaigns
    - Content-based recommendation

    ### Exported Data
    Publication cluster assignments saved to `publication_clusters.csv`
    """)
    return


if __name__ == "__main__":
    app.run()
