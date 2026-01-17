# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "polars>=1.37.1",
#     "matplotlib>=3.10.8",
#     "seaborn>=0.13.0",
#     "scipy>=1.15.3",
# ]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy import stats

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    mo.md("""
    # Variance Explainability Analysis - CTR Prediction

    This notebook analyzes which features explain variance in CTR **without target contamination**.

    ## Key Principle
    We avoid features that contain information derived from the target variable itself, such as:
    - Publication-Campaign historical CTR (direct leakage)
    - Any aggregated metrics computed from clicks/opens

    ## Features to Analyze
    1. **Target Gender** - Campaign targeting setting
    2. **Publication-Advertiser** - The pair of publisher and advertiser (not campaign)
    3. **Publication Tags** - Content categories of publications
    4. **Promoted Item Type** - Type of product/service being advertised
    5. **Target Income/Age Ranges** - Campaign demographic targeting

    ## Contamination Threshold
    Any feature explaining **>70% of variance** is flagged as potential contamination.
    """)
    return mo, np, pl, plt, stats


@app.cell
def _(mo):
    mo.md("""
    ## 1. Data Loading
    """)
    return


@app.cell
def _(pl):
    DATA_PATH = "../data/"

    placements = pl.read_csv(DATA_PATH + "placements.csv")
    campaigns = pl.read_csv(DATA_PATH + "campaigns.csv")
    advertisers = pl.read_csv(DATA_PATH + "advertisers.csv")
    publication_tags = pl.read_csv(DATA_PATH + "publication_tags.csv")

    # Calculate CTR
    placements_with_ctr = placements.with_columns([
        (pl.col("approved_clicks") / pl.col("approved_opens")).alias("ctr")
    ]).filter(pl.col("approved_opens") > 0)

    # Join with campaigns to get advertiser_id and targeting info
    placements_full = placements_with_ctr.join(
        campaigns.select([
            "campaign_id", "advertiser_id", "target_gender", 
            "target_incomes", "target_ages", "promoted_item"
        ]),
        on="campaign_id",
        how="left"
    )

    # Join with publication tags
    placements_full = placements_full.join(
        publication_tags.select(["publication_id", "tags"]),
        on="publication_id",
        how="left"
    )

    print(f"Total placements with CTR: {len(placements_full):,}")
    print(f"Unique publications: {placements_full['publication_id'].n_unique():,}")
    print(f"Unique campaigns: {placements_full['campaign_id'].n_unique():,}")
    print(f"Unique advertisers: {placements_full['advertiser_id'].n_unique():,}")

    # Overall CTR statistics
    overall_ctr_mean = placements_full.select("ctr").mean().item()
    overall_ctr_var = placements_full.select("ctr").var().item()

    print(f"\nOverall CTR Mean: {overall_ctr_mean:.6f}")
    print(f"Overall CTR Variance: {overall_ctr_var:.8f}")
    return overall_ctr_mean, overall_ctr_var, placements_full


@app.cell
def _(mo):
    mo.md("""
    ## 2. Variance Explainability Framework

    We use **variance of group means** as a measure of how much a categorical feature explains the target variance.

    For a feature F with groups {g1, g2, ..., gk}:
    - Calculate the mean CTR for each group: μ_gi
    - Calculate the variance of these group means: Var(μ_gi)
    - Ratio = Var(μ_gi) / Var(CTR_overall)

    This is similar to the between-group variance in ANOVA.
    """)
    return


@app.cell
def _(np, overall_ctr_var, pl):
    # Define contamination threshold
    CONTAMINATION_THRESHOLD = 0.70

    def calculate_variance_explained(df: pl.DataFrame, group_col: str, target_col: str = "ctr") -> dict:
        """
        Calculate the variance explained by a grouping variable.
        Returns dict with stats and contamination flag.
        """
        # Remove nulls in group column
        df_clean = df.filter(pl.col(group_col).is_not_null())

        if len(df_clean) == 0:
            return {
                "feature": group_col,
                "n_groups": 0,
                "n_samples": 0,
                "var_between_groups": np.nan,
                "var_explained_ratio": np.nan,
                "is_contaminated": False,
                "group_means_std": np.nan
            }

        # Calculate group means
        group_stats = df_clean.group_by(group_col).agg([
            pl.col(target_col).mean().alias("group_mean"),
            pl.col(target_col).std().alias("group_std"),
            pl.count().alias("group_count")
        ])

        n_groups = len(group_stats)
        n_samples = len(df_clean)

        # Variance of group means
        group_means = group_stats.select("group_mean").to_numpy().flatten()
        var_between = np.var(group_means)

        # Ratio of variance explained
        var_explained_ratio = var_between / overall_ctr_var if overall_ctr_var > 0 else 0

        return {
            "feature": group_col,
            "n_groups": n_groups,
            "n_samples": n_samples,
            "var_between_groups": var_between,
            "var_explained_ratio": var_explained_ratio,
            "is_contaminated": var_explained_ratio > CONTAMINATION_THRESHOLD,
            "group_means_std": np.std(group_means)
        }

    print(f"Contamination threshold: {CONTAMINATION_THRESHOLD*100:.0f}%")
    return CONTAMINATION_THRESHOLD, calculate_variance_explained


@app.cell
def _(mo):
    mo.md("""
    ## 3. Target Gender - Variance Explainability
    """)
    return


@app.cell
def _(
    calculate_variance_explained,
    np,
    overall_ctr_mean,
    pl,
    placements_full,
    plt,
):
    # Analyze target_gender
    gender_result = calculate_variance_explained(placements_full, "target_gender")

    print("=== Target Gender Analysis ===")
    print(f"Number of groups: {gender_result['n_groups']}")
    print(f"Number of samples: {gender_result['n_samples']:,}")
    print(f"Variance between groups: {gender_result['var_between_groups']:.8f}")
    print(f"Variance explained ratio: {gender_result['var_explained_ratio']*100:.2f}%")
    print(f"⚠️ CONTAMINATED: {gender_result['is_contaminated']}")

    # Detailed breakdown
    gender_stats = placements_full.filter(
        pl.col("target_gender").is_not_null()
    ).group_by("target_gender").agg([
        pl.col("ctr").mean().alias("mean_ctr"),
        pl.col("ctr").std().alias("std_ctr"),
        pl.count().alias("count"),
        pl.col("approved_opens").sum().alias("total_opens"),
        pl.col("approved_clicks").sum().alias("total_clicks")
    ]).with_columns(
        (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
    ).sort("count", descending=True)

    print("\nGender Group Statistics:")
    print(gender_stats)

    # Visualization
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))

    _genders = gender_stats.select("target_gender").to_numpy().flatten()
    _weighted_ctrs = gender_stats.select("weighted_ctr").to_numpy().flatten()
    _counts = gender_stats.select("count").to_numpy().flatten()

    _ax = _axes[0]
    _colors = plt.cm.Set2(np.linspace(0, 1, len(_genders)))
    _bars = _ax.bar(range(len(_genders)), _weighted_ctrs * 100, color=_colors)
    _ax.axhline(y=overall_ctr_mean * 100, color='red', linestyle='--', label=f'Overall: {overall_ctr_mean*100:.2f}%')
    _ax.set_xticks(range(len(_genders)))
    _ax.set_xticklabels(_genders, rotation=45, ha='right')
    _ax.set_ylabel('Weighted CTR (%)')
    _ax.set_title(f'CTR by Target Gender\nVariance Explained: {gender_result["var_explained_ratio"]*100:.2f}%')
    _ax.legend()

    _ax = _axes[1]
    _ax.bar(range(len(_genders)), _counts, color=_colors)
    _ax.set_xticks(range(len(_genders)))
    _ax.set_xticklabels(_genders, rotation=45, ha='right')
    _ax.set_ylabel('Number of Placements')
    _ax.set_title('Sample Size by Target Gender')

    plt.tight_layout()
    plt.show()
    return (gender_result,)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Promoted Item Type - Variance Explainability
    """)
    return


@app.cell
def _(
    calculate_variance_explained,
    np,
    overall_ctr_mean,
    pl,
    placements_full,
    plt,
):
    # Analyze promoted_item
    item_result = calculate_variance_explained(placements_full, "promoted_item")

    print("=== Promoted Item Type Analysis ===")
    print(f"Number of groups: {item_result['n_groups']}")
    print(f"Number of samples: {item_result['n_samples']:,}")
    print(f"Variance between groups: {item_result['var_between_groups']:.8f}")
    print(f"Variance explained ratio: {item_result['var_explained_ratio']*100:.2f}%")
    print(f"⚠️ CONTAMINATED: {item_result['is_contaminated']}")

    # Detailed breakdown
    item_stats = placements_full.filter(
        pl.col("promoted_item").is_not_null() & (pl.col("promoted_item") != "")
    ).group_by("promoted_item").agg([
        pl.col("ctr").mean().alias("mean_ctr"),
        pl.col("ctr").std().alias("std_ctr"),
        pl.count().alias("count"),
        pl.col("approved_opens").sum().alias("total_opens"),
        pl.col("approved_clicks").sum().alias("total_clicks")
    ]).with_columns(
        (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
    ).sort("count", descending=True)

    print("\nPromoted Item Statistics:")
    print(item_stats)

    # Visualization
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))

    _items = item_stats.select("promoted_item").to_numpy().flatten()
    _item_ctrs = item_stats.select("weighted_ctr").to_numpy().flatten()
    _item_counts = item_stats.select("count").to_numpy().flatten()

    _ax = _axes[0]
    _colors = plt.cm.Set2(np.linspace(0, 1, len(_items)))
    _ax.bar(range(len(_items)), _item_ctrs * 100, color=_colors)
    _ax.axhline(y=overall_ctr_mean * 100, color='red', linestyle='--', label=f'Overall: {overall_ctr_mean*100:.2f}%')
    _ax.set_xticks(range(len(_items)))
    _ax.set_xticklabels(_items, rotation=45, ha='right')
    _ax.set_ylabel('Weighted CTR (%)')
    _ax.set_title(f'CTR by Promoted Item\nVariance Explained: {item_result["var_explained_ratio"]*100:.2f}%')
    _ax.legend()

    _ax = _axes[1]
    _ax.bar(range(len(_items)), _item_counts, color=_colors)
    _ax.set_xticks(range(len(_items)))
    _ax.set_xticklabels(_items, rotation=45, ha='right')
    _ax.set_ylabel('Number of Placements')
    _ax.set_title('Sample Size by Promoted Item')

    plt.tight_layout()
    plt.show()
    return (item_result,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Publication-Advertiser Pair - Variance Explainability

    This analyzes the interaction between which publisher shows ads from which advertiser,
    **without** using campaign-level information that could leak CTR.
    """)
    return


@app.cell
def _(
    calculate_variance_explained,
    overall_ctr_mean,
    pl,
    placements_full,
    plt,
):
    # Create publication-advertiser pair
    placements_pub_adv = placements_full.with_columns([
        pl.concat_str([pl.col("publication_id"), pl.lit("_"), pl.col("advertiser_id")]).alias("pub_advertiser_pair")
    ])

    pub_adv_result = calculate_variance_explained(placements_pub_adv, "pub_advertiser_pair")

    print("=== Publication-Advertiser Pair Analysis ===")
    print(f"Number of unique pairs: {pub_adv_result['n_groups']:,}")
    print(f"Number of samples: {pub_adv_result['n_samples']:,}")
    print(f"Variance between groups: {pub_adv_result['var_between_groups']:.8f}")
    print(f"Variance explained ratio: {pub_adv_result['var_explained_ratio']*100:.2f}%")
    print(f"⚠️ CONTAMINATED: {pub_adv_result['is_contaminated']}")

    # Also analyze publication alone and advertiser alone for comparison
    pub_only_result = calculate_variance_explained(placements_full, "publication_id")
    adv_only_result = calculate_variance_explained(placements_full, "advertiser_id")

    print(f"\n--- Comparison ---")
    print(f"Publication only: {pub_only_result['var_explained_ratio']*100:.2f}% variance explained")
    print(f"Advertiser only: {adv_only_result['var_explained_ratio']*100:.2f}% variance explained")
    print(f"Publication-Advertiser pair: {pub_adv_result['var_explained_ratio']*100:.2f}% variance explained")

    # Distribution of pair CTRs
    _pair_stats = placements_pub_adv.group_by("pub_advertiser_pair").agg([
        pl.col("ctr").mean().alias("mean_ctr"),
        pl.count().alias("count")
    ]).filter(pl.col("count") >= 5)  # At least 5 placements

    _pair_means = _pair_stats.select("mean_ctr").to_numpy().flatten()

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution of pair mean CTRs
    _ax = _axes[0]
    _ax.hist(_pair_means[_pair_means < 0.1], bins=50, edgecolor='black', alpha=0.7, color='#3498db')
    _ax.axvline(overall_ctr_mean, color='red', linestyle='--', linewidth=2, label=f'Overall: {overall_ctr_mean:.4f}')
    _ax.set_xlabel('Mean CTR')
    _ax.set_ylabel('Number of Pub-Advertiser Pairs')
    _ax.set_title(f'Distribution of Pub-Advertiser Mean CTR (min 5 placements)\nVariance Explained: {pub_adv_result["var_explained_ratio"]*100:.2f}%')
    _ax.legend()

    # Comparison bar chart
    _ax = _axes[1]
    _features = ['Publication\nOnly', 'Advertiser\nOnly', 'Pub-Adv\nPair']
    _var_explained = [
        pub_only_result['var_explained_ratio'] * 100,
        adv_only_result['var_explained_ratio'] * 100,
        pub_adv_result['var_explained_ratio'] * 100
    ]
    _colors = ['#2ecc71' if v < 70 else '#e74c3c' for v in _var_explained]
    _ax.bar(_features, _var_explained, color=_colors, edgecolor='black')
    _ax.axhline(y=70, color='red', linestyle='--', linewidth=2, label='Contamination Threshold (70%)')
    _ax.set_ylabel('Variance Explained (%)')
    _ax.set_title('Variance Explained Comparison')
    _ax.legend()

    plt.tight_layout()
    plt.show()
    return adv_only_result, pub_adv_result, pub_only_result


@app.cell
def _(mo):
    mo.md("""
    ## 6. Publication Tags - Variance Explainability
    """)
    return


@app.cell
def _(
    calculate_variance_explained,
    np,
    overall_ctr_mean,
    pl,
    placements_full,
    plt,
):
    # Analyze publication tags
    tags_result = calculate_variance_explained(placements_full, "tags")

    print("=== Publication Tags Analysis ===")
    print(f"Number of unique tag combinations: {tags_result['n_groups']:,}")
    print(f"Number of samples: {tags_result['n_samples']:,}")
    print(f"Variance between groups: {tags_result['var_between_groups']:.8f}")
    print(f"Variance explained ratio: {tags_result['var_explained_ratio']*100:.2f}%")
    print(f"⚠️ CONTAMINATED: {tags_result['is_contaminated']}")

    # Top tag combinations by volume
    tag_stats = placements_full.filter(
        pl.col("tags").is_not_null()
    ).group_by("tags").agg([
        pl.col("ctr").mean().alias("mean_ctr"),
        pl.col("ctr").std().alias("std_ctr"),
        pl.count().alias("count"),
        pl.col("approved_opens").sum().alias("total_opens"),
        pl.col("approved_clicks").sum().alias("total_clicks")
    ]).with_columns(
        (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
    ).filter(pl.col("count") >= 100).sort("count", descending=True)

    print(f"\nTop 20 Tag Combinations (min 100 placements):")
    print(tag_stats.head(20))

    # Visualization
    _top_20_tags = tag_stats.head(20)
    _tags_names = _top_20_tags.select("tags").to_numpy().flatten()
    _tags_ctrs = _top_20_tags.select("weighted_ctr").to_numpy().flatten()
    _tags_counts = _top_20_tags.select("count").to_numpy().flatten()

    _fig, _axes = plt.subplots(1, 2, figsize=(16, 8))

    # CTR by tag
    _ax = _axes[0]
    _y_pos = np.arange(len(_tags_names))
    _colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(_tags_names)))
    _ax.barh(_y_pos, _tags_ctrs * 100, color=_colors)
    _ax.axvline(overall_ctr_mean * 100, color='red', linestyle='--', linewidth=2, label=f'Overall: {overall_ctr_mean*100:.2f}%')
    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels([str(t)[:50] for t in _tags_names], fontsize=8)
    _ax.set_xlabel('Weighted CTR (%)')
    _ax.set_title(f'CTR by Publication Tags (Top 20 by volume)\nVariance Explained: {tags_result["var_explained_ratio"]*100:.2f}%')
    _ax.invert_yaxis()
    _ax.legend()

    # Volume by tag
    _ax = _axes[1]
    _ax.barh(_y_pos, _tags_counts, color=_colors)
    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels([str(t)[:50] for t in _tags_names], fontsize=8)
    _ax.set_xlabel('Number of Placements')
    _ax.set_title('Placement Volume by Tag')
    _ax.invert_yaxis()

    plt.tight_layout()
    plt.show()
    return (tags_result,)


@app.cell
def _(mo):
    mo.md("""
    ## 7. Target Income Ranges - Variance Explainability
    """)
    return


@app.cell
def _(
    calculate_variance_explained,
    np,
    overall_ctr_mean,
    pl,
    placements_full,
    plt,
):
    # Analyze target_incomes
    income_result = calculate_variance_explained(placements_full, "target_incomes")

    print("=== Target Income Ranges Analysis ===")
    print(f"Number of unique combinations: {income_result['n_groups']:,}")
    print(f"Number of samples: {income_result['n_samples']:,}")
    print(f"Variance between groups: {income_result['var_between_groups']:.8f}")
    print(f"Variance explained ratio: {income_result['var_explained_ratio']*100:.2f}%")
    print(f"⚠️ CONTAMINATED: {income_result['is_contaminated']}")

    # Top income targeting by volume
    income_stats = placements_full.filter(
        pl.col("target_incomes").is_not_null()
    ).group_by("target_incomes").agg([
        pl.col("ctr").mean().alias("mean_ctr"),
        pl.count().alias("count"),
        pl.col("approved_opens").sum().alias("total_opens"),
        pl.col("approved_clicks").sum().alias("total_clicks")
    ]).with_columns(
        (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
    ).filter(pl.col("count") >= 100).sort("count", descending=True)

    print(f"\nTop Income Targeting Combinations (min 100 placements):")
    print(income_stats.head(15))

    # Visualization
    _top_income = income_stats.head(15)
    _income_names = _top_income.select("target_incomes").to_numpy().flatten()
    _income_ctrs = _top_income.select("weighted_ctr").to_numpy().flatten()

    _fig, _ax = plt.subplots(figsize=(12, 6))
    _y_pos = np.arange(len(_income_names))
    _colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(_income_names)))
    _ax.barh(_y_pos, _income_ctrs * 100, color=_colors)
    _ax.axvline(overall_ctr_mean * 100, color='red', linestyle='--', linewidth=2, label=f'Overall: {overall_ctr_mean*100:.2f}%')
    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels([str(t)[:60] for t in _income_names], fontsize=8)
    _ax.set_xlabel('Weighted CTR (%)')
    _ax.set_title(f'CTR by Target Income Ranges\nVariance Explained: {income_result["var_explained_ratio"]*100:.2f}%')
    _ax.invert_yaxis()
    _ax.legend()
    plt.tight_layout()
    plt.show()
    return (income_result,)


@app.cell
def _(mo):
    mo.md("""
    ## 8. Target Age Ranges - Variance Explainability
    """)
    return


@app.cell
def _(
    calculate_variance_explained,
    np,
    overall_ctr_mean,
    pl,
    placements_full,
    plt,
):
    # Analyze target_ages
    age_result = calculate_variance_explained(placements_full, "target_ages")

    print("=== Target Age Ranges Analysis ===")
    print(f"Number of unique combinations: {age_result['n_groups']:,}")
    print(f"Number of samples: {age_result['n_samples']:,}")
    print(f"Variance between groups: {age_result['var_between_groups']:.8f}")
    print(f"Variance explained ratio: {age_result['var_explained_ratio']*100:.2f}%")
    print(f"⚠️ CONTAMINATED: {age_result['is_contaminated']}")

    # Top age targeting by volume
    age_stats = placements_full.filter(
        pl.col("target_ages").is_not_null()
    ).group_by("target_ages").agg([
        pl.col("ctr").mean().alias("mean_ctr"),
        pl.count().alias("count"),
        pl.col("approved_opens").sum().alias("total_opens"),
        pl.col("approved_clicks").sum().alias("total_clicks")
    ]).with_columns(
        (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
    ).filter(pl.col("count") >= 100).sort("count", descending=True)

    print(f"\nTop Age Targeting Combinations (min 100 placements):")
    print(age_stats.head(15))

    # Visualization
    _top_age = age_stats.head(15)
    _age_names = _top_age.select("target_ages").to_numpy().flatten()
    _age_ctrs = _top_age.select("weighted_ctr").to_numpy().flatten()

    _fig, _ax = plt.subplots(figsize=(12, 6))
    _y_pos = np.arange(len(_age_names))
    _colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(_age_names)))
    _ax.barh(_y_pos, _age_ctrs * 100, color=_colors)
    _ax.axvline(overall_ctr_mean * 100, color='red', linestyle='--', linewidth=2, label=f'Overall: {overall_ctr_mean*100:.2f}%')
    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels([str(t)[:60] for t in _age_names], fontsize=8)
    _ax.set_xlabel('Weighted CTR (%)')
    _ax.set_title(f'CTR by Target Age Ranges\nVariance Explained: {age_result["var_explained_ratio"]*100:.2f}%')
    _ax.invert_yaxis()
    _ax.legend()
    plt.tight_layout()
    plt.show()
    return (age_result,)


@app.cell
def _(mo):
    mo.md("""
    ## 9. Summary - All Features Variance Explainability
    """)
    return


@app.cell
def _(
    CONTAMINATION_THRESHOLD,
    adv_only_result,
    age_result,
    gender_result,
    income_result,
    item_result,
    np,
    plt,
    pub_adv_result,
    pub_only_result,
    tags_result,
):
    # Compile all results
    all_results = [
        {"feature": "Target Gender", **gender_result},
        {"feature": "Promoted Item", **item_result},
        {"feature": "Publication Only", **pub_only_result},
        {"feature": "Advertiser Only", **adv_only_result},
        {"feature": "Publication-Advertiser", **pub_adv_result},
        {"feature": "Publication Tags", **tags_result},
        {"feature": "Target Income", **income_result},
        {"feature": "Target Age", **age_result},
    ]

    # Sort by variance explained
    all_results_sorted = sorted(all_results, key=lambda x: x['var_explained_ratio'], reverse=True)

    print("=" * 80)
    print("VARIANCE EXPLAINABILITY SUMMARY")
    print("=" * 80)
    print(f"{'Feature':<25} {'Groups':>10} {'Var Explained':>15} {'Status':>15}")
    print("-" * 80)
    for r in all_results_sorted:
        status = "⚠️  CONTAMINATED" if r['is_contaminated'] else "✓ OK"
        var_pct = f"{r['var_explained_ratio']*100:.2f}%"
        print(f"{r['feature']:<25} {r['n_groups']:>10,} {var_pct:>15} {status:>15}")
    print("=" * 80)

    # Visualization
    _fig, _ax = plt.subplots(figsize=(12, 8))

    _features = [r['feature'] for r in all_results_sorted]
    _var_explained = [r['var_explained_ratio'] * 100 for r in all_results_sorted]
    _contaminated = [r['is_contaminated'] for r in all_results_sorted]

    _colors = ['#e74c3c' if c else '#2ecc71' for c in _contaminated]
    _y_pos = np.arange(len(_features))

    _bars = _ax.barh(_y_pos, _var_explained, color=_colors, edgecolor='black')
    _ax.axvline(x=CONTAMINATION_THRESHOLD * 100, color='red', linestyle='--', linewidth=2, 
                label=f'Contamination Threshold ({CONTAMINATION_THRESHOLD*100:.0f}%)')

    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels(_features)
    _ax.set_xlabel('Variance Explained (%)')
    _ax.set_title('Feature Variance Explainability Summary\n(Red = Potential Contamination)')
    _ax.legend(loc='lower right')
    _ax.invert_yaxis()

    # Add value labels
    for i, (_bar, val) in enumerate(zip(_bars, _var_explained)):
        _ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()
    return (all_results_sorted,)


@app.cell
def _(CONTAMINATION_THRESHOLD, all_results_sorted, mo):
    # Generate recommendations
    clean_features = [r for r in all_results_sorted if not r['is_contaminated']]
    contaminated_features = [r for r in all_results_sorted if r['is_contaminated']]

    clean_list = "\n".join([f"- **{r['feature']}**: {r['var_explained_ratio']*100:.2f}% variance explained" 
                           for r in clean_features])
    contaminated_list = "\n".join([f"- **{r['feature']}**: {r['var_explained_ratio']*100:.2f}% variance explained" 
                                   for r in contaminated_features]) if contaminated_features else "None detected"

    _summary = f"""
    ## Summary & Recommendations

    ### Contamination Threshold: {CONTAMINATION_THRESHOLD*100:.0f}%

    ### ✓ Clean Features (Safe to Use)
    {clean_list}

    ### ⚠️ Potentially Contaminated Features (>70% variance explained)
    {contaminated_list}

    ### Key Insights

    1. **Target Gender** explains only a small portion of CTR variance, suggesting demographic targeting
       alone isn't highly predictive.

    2. **Publication-Advertiser pairs** show how much the match between publisher and advertiser matters,
       independent of specific campaign performance.

    3. **Publication Tags** capture content affinity - certain content categories naturally have
       different engagement rates.

    4. **Promoted Item Type** shows whether the type of product/service affects engagement.

    ### Modeling Recommendations

    For features with low variance explained (<10%):
    - May still be useful in combination with other features
    - Consider as interaction terms

    For features with moderate variance explained (10-40%):
    - Good candidates for primary model features
    - Likely to provide predictive signal without leakage

    For features approaching the threshold (>50%):
    - Use with caution
    - Validate with time-based cross-validation
    - Consider if the grouping is too granular (many groups → overfitting risk)

    ### Next Steps
    1. Create feature engineering pipeline with clean features
    2. Build baseline models using these features
    3. Compare against more complex models
    4. Ensure temporal validation to prevent leakage
    """
    mo.md(_summary)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. Statistical Significance Tests

    Let's verify the significance of variance explained using ANOVA F-tests.
    """)
    return


@app.cell
def _(np, pl, placements_full, stats):
    def perform_anova(df: pl.DataFrame, group_col: str, target_col: str = "ctr"):
        """Perform one-way ANOVA and return F-statistic and p-value."""
        df_clean = df.filter(pl.col(group_col).is_not_null())

        groups = df_clean.group_by(group_col).agg(
            pl.col(target_col).alias("values")
        )

        group_values = [np.array(row["values"]) for row in groups.iter_rows(named=True)]

        # Filter groups with enough samples
        group_values = [g for g in group_values if len(g) >= 2]

        if len(group_values) < 2:
            return np.nan, np.nan

        f_stat, p_value = stats.f_oneway(*group_values)
        return f_stat, p_value

    # Run ANOVA for each feature
    anova_results = {}

    features_to_test = [
        ("target_gender", "Target Gender"),
        ("promoted_item", "Promoted Item"),
        ("tags", "Publication Tags"),
        ("target_incomes", "Target Income"),
        ("target_ages", "Target Age"),
        ("advertiser_id", "Advertiser"),
    ]

    print("=" * 70)
    print("ANOVA F-TEST RESULTS")
    print("=" * 70)
    print(f"{'Feature':<25} {'F-statistic':>15} {'p-value':>20}")
    print("-" * 70)

    for _col, _name in features_to_test:
        _f_stat, _p_val = perform_anova(placements_full, _col)
        anova_results[_name] = {"f_stat": _f_stat, "p_value": _p_val}

        _p_val_str = f"{_p_val:.2e}" if not np.isnan(_p_val) else "N/A"
        _f_stat_str = f"{_f_stat:.2f}" if not np.isnan(_f_stat) else "N/A"
        _sig = "***" if _p_val < 0.001 else "**" if _p_val < 0.01 else "*" if _p_val < 0.05 else ""

        print(f"{_name:<25} {_f_stat_str:>15} {_p_val_str:>17} {_sig:>3}")

    print("-" * 70)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
