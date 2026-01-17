import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Embedding-based CTR Analysis

    This notebook creates embeddings for advertisers and publications using Sentence Transformers,
    then analyzes the correlation between these embeddings and CTR (Click-Through Rate).

    ## Approach
    1. Load and preprocess advertiser data (name + description)
    2. Load and preprocess publication data (tags + description)
    3. Generate embeddings using sentence-transformers
    4. Join with placements data to compute CTR
    5. Analyze correlation between embedding dimensions and CTR
    """)
    return


@app.cell
def _():
    import polars as pl
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    return PCA, Path, SentenceTransformer, np, pl, plt, sns, stats


@app.cell
def _(Path):
    # Define data paths
    DATA_DIR = Path("../data")
    ADVERTISERS_PATH = DATA_DIR / "advertisers.csv"
    CAMPAIGNS_PATH = DATA_DIR / "campaigns.csv"
    PLACEMENTS_PATH = DATA_DIR / "placements.csv"
    PUBLICATION_METS_PATH = DATA_DIR / "publication_mets.csv"
    PUBLICATION_TAGS_PATH = DATA_DIR / "publication_tags.csv"
    return (
        ADVERTISERS_PATH,
        CAMPAIGNS_PATH,
        DATA_DIR,
        PLACEMENTS_PATH,
        PUBLICATION_METS_PATH,
        PUBLICATION_TAGS_PATH,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Load Data
    """)
    return


@app.cell
def _(
    ADVERTISERS_PATH,
    CAMPAIGNS_PATH,
    PLACEMENTS_PATH,
    PUBLICATION_METS_PATH,
    PUBLICATION_TAGS_PATH,
    pl,
):
    # Load all data files
    advertisers_df = pl.read_csv(ADVERTISERS_PATH)
    campaigns_df = pl.read_csv(CAMPAIGNS_PATH)
    placements_df = pl.read_csv(PLACEMENTS_PATH)
    publication_mets_df = pl.read_csv(PUBLICATION_METS_PATH)
    publication_tags_df = pl.read_csv(PUBLICATION_TAGS_PATH)

    print(f"Advertisers: {advertisers_df.shape}")
    print(f"Campaigns: {campaigns_df.shape}")
    print(f"Placements: {placements_df.shape}")
    print(f"Publication metadata: {publication_mets_df.shape}")
    print(f"Publication tags: {publication_tags_df.shape}")
    return (
        advertisers_df,
        campaigns_df,
        placements_df,
        publication_mets_df,
        publication_tags_df,
    )


@app.cell
def _(campaigns_df, pl, placements_df):
    # Calculate CTR per placement
    placements_with_ctr = placements_df.with_columns(
        (pl.col("approved_clicks") / pl.col("approved_opens")).alias("ctr")
    ).filter(
        pl.col("approved_opens") > 0  # Avoid division by zero
    )

    # Join placements with campaigns to get advertiser_id
    placements_with_advertiser = placements_with_ctr.join(
        campaigns_df.select(["campaign_id", "advertiser_id"]),
        on="campaign_id",
        how="left"
    )

    # Create aggregated CTR by advertiser
    advertiser_ctr = placements_with_advertiser.group_by("advertiser_id").agg([
        pl.col("ctr").mean().alias("mean_ctr"),
        pl.col("ctr").std().alias("std_ctr"),
        pl.col("approved_clicks").sum().alias("total_clicks"),
        pl.col("approved_opens").sum().alias("total_opens"),
        pl.len().alias("placement_count")
    ]).with_columns(
        (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
    )

    # Create aggregated CTR by publication
    publication_ctr = placements_with_ctr.group_by("publication_id").agg([
        pl.col("ctr").mean().alias("mean_ctr"),
        pl.col("ctr").std().alias("std_ctr"),
        pl.col("approved_clicks").sum().alias("total_clicks"),
        pl.col("approved_opens").sum().alias("total_opens"),
        pl.len().alias("placement_count")
    ]).with_columns(
        (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
    )

    print(f"Advertisers with CTR data: {advertiser_ctr.shape[0]}")
    print(f"Publications with CTR data: {publication_ctr.shape[0]}")
    return advertiser_ctr, publication_ctr


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Prepare Text for Embeddings
    """)
    return


@app.cell
def _(advertisers_df, pl):
    # Prepare advertiser text: combine name and description
    advertiser_text = advertisers_df.with_columns(
        pl.concat_str(
            [
                pl.col("name").fill_null(""),
                pl.lit(": "),
                pl.col("description").fill_null("")
            ]
        ).alias("text_for_embedding")
    ).select(["advertiser_id", "name", "text_for_embedding"])

    advertiser_text.head(5)
    return (advertiser_text,)


@app.cell
def _(pl, publication_mets_df, publication_tags_df):
    # Prepare publication text: combine tags and description
    publication_combined = publication_mets_df.join(
        publication_tags_df.select(["publication_id", "tags"]),
        on="publication_id",
        how="left"
    ).with_columns(
        pl.concat_str(
            [
                pl.col("name").fill_null(""),
                pl.lit(". Tags: "),
                pl.col("tags").fill_null("").str.replace_all(r"[\{\}']", ""),
                pl.lit(". "),
                pl.col("description").fill_null("")
            ]
        ).alias("text_for_embedding")
    ).select(["publication_id", "name", "text_for_embedding"])

    publication_combined.head(5)
    return (publication_combined,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Generate Embeddings
    """)
    return


@app.cell
def _(SentenceTransformer, mo):
    mo.md(r"""Loading sentence transformer model (this may take a moment)...""")

    # Load the sentence transformer model
    # Using a smaller, efficient model suitable for semantic similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return (model,)


@app.cell
def _(advertiser_text, model):
    # Generate embeddings for advertisers
    advertiser_texts = advertiser_text["text_for_embedding"].to_list()
    advertiser_ids = advertiser_text["advertiser_id"].to_list()

    print(f"Generating embeddings for {len(advertiser_texts)} advertisers...")
    advertiser_embeddings = model.encode(advertiser_texts, show_progress_bar=True)
    print(f"Advertiser embedding shape: {advertiser_embeddings.shape}")

    # Create a mapping of advertiser_id -> embedding
    advertiser_embedding_dict = {
        aid: emb for aid, emb in zip(advertiser_ids, advertiser_embeddings)
    }
    return (advertiser_embedding_dict,)


@app.cell
def _(model, publication_combined):
    # Generate embeddings for publications
    publication_texts = publication_combined["text_for_embedding"].to_list()
    publication_ids = publication_combined["publication_id"].to_list()

    print(f"Generating embeddings for {len(publication_texts)} publications...")
    publication_embeddings = model.encode(publication_texts, show_progress_bar=True)
    print(f"Publication embedding shape: {publication_embeddings.shape}")

    # Create a mapping of publication_id -> embedding
    publication_embedding_dict = {
        pid: emb for pid, emb in zip(publication_ids, publication_embeddings)
    }
    return (publication_embedding_dict,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Correlation Analysis: Advertiser Embeddings vs CTR
    """)
    return


@app.cell
def _(advertiser_ctr, advertiser_embedding_dict, np, pl, stats):
    def cor_adv():
        # Create dataset with advertiser embeddings and CTR
        valid_advertisers = []
        valid_embeddings_adv = []
        valid_ctrs_adv = []
    
        for row in advertiser_ctr.iter_rows(named=True):
            adv_id = row["advertiser_id"]
            if adv_id in advertiser_embedding_dict:
                valid_advertisers.append(adv_id)
                valid_embeddings_adv.append(advertiser_embedding_dict[adv_id])
                valid_ctrs_adv.append(row["weighted_ctr"])
    
        embeddings_array_adv = np.array(valid_embeddings_adv)
        ctr_array_adv = np.array(valid_ctrs_adv)
    
        print(f"Matched advertisers with embeddings and CTR: {len(valid_advertisers)}")
        print(f"Embedding dimensions: {embeddings_array_adv.shape[1]}")
    
        # Calculate correlation for each embedding dimension with CTR
        correlations_adv = []
        for dim in range(embeddings_array_adv.shape[1]):
            corr, p_value = stats.pearsonr(embeddings_array_adv[:, dim], ctr_array_adv)
            correlations_adv.append({
                "dimension": dim,
                "correlation": corr,
                "p_value": p_value,
                "significant": p_value < 0.05
            })
    
        correlations_adv_df = pl.DataFrame(correlations_adv)
        print(f"\nSignificant correlations (p < 0.05): {correlations_adv_df.filter(pl.col('significant') == 1).shape[0]}")
        return correlations_adv_df, embeddings_array_adv , ctr_array_adv

    correlations_adv_df, embeddings_array_adv, ctr_array_adv = cor_adv()
    return correlations_adv_df, ctr_array_adv, embeddings_array_adv


@app.cell
def _(correlations_adv_df):
    correlations_adv_df
    return


@app.cell
def _(correlations_adv_df, pl, plt, sns):
    # Visualize advertiser embedding correlations
    fig_adv, axes_adv = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of correlations
    sns.histplot(correlations_adv_df["correlation"].to_numpy(), bins=30, ax=axes_adv[0])
    axes_adv[0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes_adv[0].set_xlabel("Pearson Correlation")
    axes_adv[0].set_ylabel("Count")
    axes_adv[0].set_title("Distribution of Embedding Dimension Correlations with CTR\n(Advertisers)")

    # Top positive and negative correlations
    sorted_corrs = correlations_adv_df.sort("correlation")
    top_negative = sorted_corrs.head(10)
    top_positive = sorted_corrs.tail(10)
    top_corrs = pl.concat([top_negative, top_positive])

    colors = ['red' if x < 0 else 'green' for x in top_corrs["correlation"].to_list()]
    axes_adv[1].barh(
        [f"Dim {d}" for d in top_corrs["dimension"].to_list()],
        top_corrs["correlation"].to_list(),
        color=colors
    )
    axes_adv[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes_adv[1].set_xlabel("Correlation with CTR")
    axes_adv[1].set_title("Top 10 Positive and Negative Correlations\n(Advertiser Embeddings)")

    plt.tight_layout()
    fig_adv
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Correlation Analysis: Publication Embeddings vs CTR
    """)
    return


@app.cell
def _(np, pl, publication_ctr, publication_embedding_dict, stats):
    def cor_pub():
        # Create dataset with publication embeddings and CTR
        valid_publications = []
        valid_embeddings_pub = []
        valid_ctrs_pub = []
    
        for row in publication_ctr.iter_rows(named=True):
            pub_id = row["publication_id"]
            if pub_id in publication_embedding_dict:
                valid_publications.append(pub_id)
                valid_embeddings_pub.append(publication_embedding_dict[pub_id])
                valid_ctrs_pub.append(row["weighted_ctr"])
    
        embeddings_array_pub = np.array(valid_embeddings_pub)
        ctr_array_pub = np.array(valid_ctrs_pub)
    
        print(f"Matched publications with embeddings and CTR: {len(valid_publications)}")
        print(f"Embedding dimensions: {embeddings_array_pub.shape[1]}")
    
        # Calculate correlation for each embedding dimension with CTR
        correlations_pub = []
        for dim in range(embeddings_array_pub.shape[1]):
            corr_pub, p_value_pub = stats.pearsonr(embeddings_array_pub[:, dim], ctr_array_pub)
            correlations_pub.append({
                "dimension": dim,
                "correlation": corr_pub,
                "p_value": p_value_pub,
                "significant": p_value_pub < 0.05
            })
    
        correlations_pub_df = pl.DataFrame(correlations_pub)
        print(f"\nSignificant correlations (p < 0.05): {correlations_pub_df.filter(pl.col('significant') == 1).shape[0]}")
        return correlations_pub_df, embeddings_array_pub, ctr_array_pub

    correlations_pub_df, embeddings_array_pub, ctr_array_pub = cor_pub()
    return correlations_pub_df, ctr_array_pub, embeddings_array_pub


@app.cell
def _(correlations_pub_df, pl, plt, sns):
    # Visualize publication embedding correlations
    fig_pub, axes_pub = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of correlations
    sns.histplot(correlations_pub_df["correlation"].to_numpy(), bins=30, ax=axes_pub[0])
    axes_pub[0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes_pub[0].set_xlabel("Pearson Correlation")
    axes_pub[0].set_ylabel("Count")
    axes_pub[0].set_title("Distribution of Embedding Dimension Correlations with CTR\n(Publications)")

    # Top positive and negative correlations
    sorted_corrs_pub = correlations_pub_df.sort("correlation")
    top_negative_pub = sorted_corrs_pub.head(10)
    top_positive_pub = sorted_corrs_pub.tail(10)
    top_corrs_pub = pl.concat([top_negative_pub, top_positive_pub])

    colors_pub = ['red' if x < 0 else 'green' for x in top_corrs_pub["correlation"].to_list()]
    axes_pub[1].barh(
        [f"Dim {d}" for d in top_corrs_pub["dimension"].to_list()],
        top_corrs_pub["correlation"].to_list(),
        color=colors_pub
    )
    axes_pub[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes_pub[1].set_xlabel("Correlation with CTR")
    axes_pub[1].set_title("Top 10 Positive and Negative Correlations\n(Publication Embeddings)")

    plt.tight_layout()
    fig_pub
    return


@app.cell
def _(correlations_pub_df):
    correlations_pub_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. PCA Analysis and Overall Correlation
    """)
    return


@app.cell
def _(PCA, ctr_array_adv, embeddings_array_adv, plt, stats):

    # PCA on advertiser embeddings
    n_components = 10
    pca_adv = PCA(n_components=n_components)
    embeddings_pca_adv = pca_adv.fit_transform(embeddings_array_adv)

    print("Advertiser Embeddings PCA:")
    print(f"Explained variance ratio: {pca_adv.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca_adv.explained_variance_ratio_):.4f}")

    # Correlations with PCA components
    pca_correlations_adv = []
    for i in range(n_components):
        corr, p_value = stats.pearsonr(embeddings_pca_adv[:, i], ctr_array_adv)
        pca_correlations_adv.append({
            "component": i,
            "correlation": corr,
            "p_value": p_value,
            "explained_var": pca_adv.explained_variance_ratio_[i]
        })
        print(f"PC{i}: correlation={corr:.4f}, p-value={p_value:.4f}, var={pca_adv.explained_variance_ratio_[i]:.4f}")

    # Scatter plot of first two PCs colored by CTR
    fig_pca_adv, ax_pca_adv = plt.subplots(figsize=(10, 8))
    scatter = ax_pca_adv.scatter(
        embeddings_pca_adv[:, 0],
        embeddings_pca_adv[:, 1],
        c=ctr_array_adv,
        cmap='viridis',
        alpha=0.6,
        s=50
    )
    plt.colorbar(scatter, label='CTR')
    ax_pca_adv.set_xlabel(f"PC1 ({pca_adv.explained_variance_ratio_[0]:.2%} variance)")
    ax_pca_adv.set_ylabel(f"PC2 ({pca_adv.explained_variance_ratio_[1]:.2%} variance)")
    ax_pca_adv.set_title("Advertiser Embeddings: PCA Visualization colored by CTR")
    fig_pca_adv
    return (n_components,)


@app.cell
def _(PCA, ctr_array_pub, embeddings_array_pub, n_components, plt, stats):
    # PCA on publication embeddings
    def pub_graph(n_components, embeddings_array_pub):
        pca_pub = PCA(n_components=n_components)
        embeddings_pca_pub = pca_pub.fit_transform(embeddings_array_pub)
    
        print("Publication Embeddings PCA:")
        print(f"Explained variance ratio: {pca_pub.explained_variance_ratio_}")
        print(f"Total explained variance: {sum(pca_pub.explained_variance_ratio_):.4f}")
    
        # Correlations with PCA components
        pca_correlations_pub = []
        for i in range(n_components):
            corr, p_value = stats.pearsonr(embeddings_pca_pub[:, i], ctr_array_pub)
            pca_correlations_pub.append({
                "component": i,
                "correlation": corr,
                "p_value": p_value,
                "explained_var": pca_pub.explained_variance_ratio_[i]
            })
            print(f"PC{i}: correlation={corr:.4f}, p-value={p_value:.4f}, var={pca_pub.explained_variance_ratio_[i]:.4f}")
        return embeddings_pca_pub, pca_pub
    
    embeddings_pca_pub, pca_pub = pub_graph(n_components, embeddings_array_pub)

    # Scatter plot of first two PCs colored by CTR
    fig_pca_pub, ax_pca_pub = plt.subplots(figsize=(10, 8))
    scatter_pub = ax_pca_pub.scatter(
        embeddings_pca_pub[:, 0],
        embeddings_pca_pub[:, 1],
        c=ctr_array_pub,
        cmap='viridis',
        alpha=0.6,
        s=50
    )
    plt.colorbar(scatter_pub, label='CTR')
    ax_pca_pub.set_xlabel(f"PC1 ({pca_pub.explained_variance_ratio_[0]:.2%} variance)")
    ax_pca_pub.set_ylabel(f"PC2 ({pca_pub.explained_variance_ratio_[1]:.2%} variance)")
    ax_pca_pub.set_title("Publication Embeddings: PCA Visualization colored by CTR")
    fig_pca_pub
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Summary Statistics
    """)
    return


@app.cell
def _(correlations_adv_df, correlations_pub_df, pl):
    # Summary statistics
    def summarize_correlations(corr_df, name):
        abs_corrs = corr_df["correlation"].abs()
        sig_corrs = corr_df.filter(pl.col("significant") == 1)

        return {
            "Entity": name,
            "Mean |correlation|": abs_corrs.mean(),
            "Max |correlation|": abs_corrs.max(),
            "Significant dims (p<0.05)": sig_corrs.shape[0],
            "% Significant": (sig_corrs.shape[0] / corr_df.shape[0]) * 100,
            "Max positive": corr_df["correlation"].max(),
            "Max negative": corr_df["correlation"].min(),
        }

    summary = pl.DataFrame([
        summarize_correlations(correlations_adv_df, "Advertiser"),
        summarize_correlations(correlations_pub_df, "Publication")
    ])
    summary
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation

    The analysis shows:

    1. **Embedding Dimension Correlations**: Individual embedding dimensions from sentence transformers
       show varying correlations with CTR. Significant correlations (p < 0.05) indicate dimensions
       that capture semantic features related to CTR performance.

    2. **Advertiser Embeddings**: The semantic representation of advertiser name + description
       captures features that may be predictive of CTR. The distribution and strength of correlations
       indicates how well advertiser identity/description relates to click behavior.

    3. **Publication Embeddings**: Similarly, publication tags + description embeddings show
       correlations with CTR, potentially capturing audience interests and content quality signals.

    4. **PCA Components**: The principal components reveal the main semantic axes of variation
       in the embeddings and their relationship with CTR.

    **Key Takeaways**:
    - If many dimensions show significant correlations, the embeddings capture meaningful
      CTR-related information
    - The magnitude of correlations indicates the strength of the relationship
    - PCA visualization helps understand if high/low CTR entities cluster in semantic space
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Export Embeddings for Further Analysis
    """)
    return


@app.cell
def _(
    DATA_DIR,
    advertiser_ctr,
    advertiser_embedding_dict,
    pl,
    publication_ctr,
    publication_embedding_dict,
):
    # Create dataframes with embeddings for export
    def create_embedding_df(embedding_dict, ctr_df, id_col):
        rows = []
        for row in ctr_df.iter_rows(named=True):
            entity_id = row[id_col]
            if entity_id in embedding_dict:
                emb = embedding_dict[entity_id]
                row_data = {
                    id_col: entity_id,
                    "weighted_ctr": row["weighted_ctr"],
                    "mean_ctr": row["mean_ctr"],
                    "total_clicks": row["total_clicks"],
                    "total_opens": row["total_opens"],
                }
                # Add embedding dimensions
                for i, val in enumerate(emb):
                    row_data[f"emb_{i}"] = float(val)
                rows.append(row_data)
        return pl.DataFrame(rows)

    advertiser_embeddings_df = create_embedding_df(
        advertiser_embedding_dict, advertiser_ctr, "advertiser_id"
    )
    publication_embeddings_df = create_embedding_df(
        publication_embedding_dict, publication_ctr, "publication_id"
    )

    print(f"Advertiser embeddings shape: {advertiser_embeddings_df.shape}")
    print(f"Publication embeddings shape: {publication_embeddings_df.shape}")

    # Save to CSV
    advertiser_embeddings_df.write_csv(DATA_DIR / "advertiser_embeddings.csv")
    publication_embeddings_df.write_csv(DATA_DIR / "publication_embeddings.csv")
    print("\nEmbeddings saved to data/advertiser_embeddings.csv and data/publication_embeddings.csv")
    return


if __name__ == "__main__":
    app.run()
