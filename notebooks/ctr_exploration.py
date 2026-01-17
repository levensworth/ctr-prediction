# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "polars>=1.37.1",
#     "matplotlib>=3.10.8",
#     "seaborn>=0.13.0",
#     "scipy>=1.15.3",
#     "scikit-learn>=1.5.0",
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
    from sklearn.preprocessing import LabelEncoder

    # Configure plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    mo.md("""
    # CTR Prediction - Data Exploration Notebook

    This notebook explores the datasets for building a Click-Through Rate (CTR) prediction model.

    ## Goal
    We want to predict CTR given:
    - `publication_id`: The newsletter publisher showing the ad
    - `campaign_id`: The ad campaign being displayed

    ## Available Data
    - **placements.csv**: Performance data for Ad/Publication pairs
    - **campaigns.csv**: Campaign information with targeting settings
    - **advertisers.csv**: Advertiser details
    - **publication_mets.csv**: Publication metadata
    - **publication_tags.csv**: Content tags for publications
    - **post_embeddings.csv**: Content embeddings (10GB+, will be handled separately)

    Since we have aggregated data (total clicks/opens per post, not per-user), we're essentially 
    predicting the expected CTR for a given publication-campaign combination.
    """)
    return mo, np, pl, plt, sns, stats


@app.cell
def _(mo):
    mo.md("""
    ## 1. Data Loading

    Let's load all the datasets and examine their structure.
    """)
    return


@app.cell
def _(pl):
    # Load all datasets
    DATA_PATH = "../data/"

    placements = pl.read_csv(DATA_PATH + "placements.csv")
    campaigns = pl.read_csv(DATA_PATH + "campaigns.csv")
    advertisers = pl.read_csv(DATA_PATH + "advertisers.csv")
    publications = pl.read_csv(DATA_PATH + "publication_mets.csv")
    publication_tags = pl.read_csv(DATA_PATH + "publication_tags.csv")

    print(f"Placements: {placements.shape[0]:,} rows, {placements.shape[1]} columns")
    print(f"Campaigns: {campaigns.shape[0]:,} rows, {campaigns.shape[1]} columns")
    print(f"Advertisers: {advertisers.shape[0]:,} rows, {advertisers.shape[1]} columns")
    print(f"Publications: {publications.shape[0]:,} rows, {publications.shape[1]} columns")
    print(f"Publication Tags: {publication_tags.shape[0]:,} rows, {publication_tags.shape[1]} columns")
    return campaigns, placements, publication_tags


@app.cell
def _(mo, placements):
    mo.md(f"""
    ## 2. Placements Dataset - Core Performance Data

    This is our main dataset with {placements.shape[0]:,} records of ad placements.

    **Columns:**
    - `post_id`: Unique identifier for each placement
    - `publication_id`: Newsletter that displayed the ad
    - `campaign_id`: Ad campaign
    - `approved_clicks`: Number of clicks received
    - `approved_opens`: Number of email opens (impressions)
    - `post_send_at`: Timestamp when the email was sent

    **Target Variable:** CTR = approved_clicks / approved_opens
    """)
    return


@app.cell
def _(pl, placements):
    # Calculate CTR and explore the placements dataset
    placements_with_ctr = placements.with_columns([
        (pl.col("approved_clicks") / pl.col("approved_opens")).alias("ctr"),
        pl.col("post_send_at").str.to_datetime().alias("send_datetime")
    ]).filter(
        pl.col("approved_opens") > 0  # Remove records with no opens to avoid division by zero
    )

    print("Placements Schema:")
    print(placements_with_ctr.schema)
    print("\nSample Data:")
    placements_with_ctr.head(10)
    return (placements_with_ctr,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. CTR Distribution Analysis

    Let's understand the distribution of our target variable (CTR).
    """)
    return


@app.cell
def _(np, placements_with_ctr, plt):
    # CTR Distribution Analysis
    ctr_values = placements_with_ctr.select("ctr").to_numpy().flatten()

    # Filter out extreme values for better visualization
    ctr_filtered = ctr_values[(ctr_values >= 0) & (ctr_values <= 0.2)]

    _fig, _axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of CTR (filtered)
    _ax1 = _axes[0, 0]
    _ax1.hist(ctr_filtered, bins=100, edgecolor='black', alpha=0.7, color='#2E86AB')
    _ax1.set_xlabel('CTR')
    _ax1.set_ylabel('Frequency')
    _ax1.set_title('CTR Distribution (0-20%)')
    _ax1.axvline(np.median(ctr_filtered), color='red', linestyle='--', label=f'Median: {np.median(ctr_filtered):.4f}')
    _ax1.axvline(np.mean(ctr_filtered), color='green', linestyle='--', label=f'Mean: {np.mean(ctr_filtered):.4f}')
    _ax1.legend()

    # 2. Log-scale histogram
    _ax2 = _axes[0, 1]
    ctr_positive = ctr_values[ctr_values > 0]
    _ax2.hist(np.log10(ctr_positive + 1e-10), bins=100, edgecolor='black', alpha=0.7, color='#A23B72')
    _ax2.set_xlabel('log10(CTR)')
    _ax2.set_ylabel('Frequency')
    _ax2.set_title('CTR Distribution (Log Scale)')

    # 3. Box plot
    _ax3 = _axes[1, 0]
    box_data = ctr_values[(ctr_values >= 0) & (ctr_values <= 0.1)]
    _ax3.boxplot(box_data, vert=True)
    _ax3.set_ylabel('CTR')
    _ax3.set_title('CTR Box Plot (0-10%)')

    # 4. CDF
    _ax4 = _axes[1, 1]
    sorted_ctr = np.sort(ctr_filtered)
    cdf = np.arange(1, len(sorted_ctr) + 1) / len(sorted_ctr)
    _ax4.plot(sorted_ctr, cdf, color='#F18F01', linewidth=2)
    _ax4.set_xlabel('CTR')
    _ax4.set_ylabel('Cumulative Probability')
    _ax4.set_title('CTR Cumulative Distribution Function')
    _ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Statistics
    print("\n=== CTR Statistics ===")
    print(f"Total records: {len(ctr_values):,}")
    print(f"Records with CTR = 0: {np.sum(ctr_values == 0):,} ({100*np.sum(ctr_values == 0)/len(ctr_values):.2f}%)")
    print(f"Min CTR: {np.min(ctr_values):.6f}")
    print(f"Max CTR: {np.max(ctr_values):.6f}")
    print(f"Mean CTR: {np.mean(ctr_values):.6f}")
    print(f"Median CTR: {np.median(ctr_values):.6f}")
    print(f"Std Dev: {np.std(ctr_values):.6f}")
    print(f"25th percentile: {np.percentile(ctr_values, 25):.6f}")
    print(f"75th percentile: {np.percentile(ctr_values, 75):.6f}")
    print(f"95th percentile: {np.percentile(ctr_values, 95):.6f}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Volume Analysis - Clicks and Opens Distribution

    Understanding the scale of impressions (opens) and clicks helps us weigh our CTR calculations.
    """)
    return


@app.cell
def _(np, placements_with_ctr, plt):
    # Clicks and Opens Distribution
    opens = placements_with_ctr.select("approved_opens").to_numpy().flatten()
    clicks = placements_with_ctr.select("approved_clicks").to_numpy().flatten()

    _fig2, _axes2 = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Opens distribution (log scale)
    _ax = _axes2[0, 0]
    _ax.hist(np.log10(opens + 1), bins=100, edgecolor='black', alpha=0.7, color='#3A86FF')
    _ax.set_xlabel('log10(Opens + 1)')
    _ax.set_ylabel('Frequency')
    _ax.set_title('Distribution of Email Opens (Log Scale)')
    _ax.axvline(np.log10(np.median(opens) + 1), color='red', linestyle='--', 
               label=f'Median: {np.median(opens):,.0f}')
    _ax.legend()

    # 2. Clicks distribution (log scale)
    _ax = _axes2[0, 1]
    _ax.hist(np.log10(clicks + 1), bins=100, edgecolor='black', alpha=0.7, color='#FF006E')
    _ax.set_xlabel('log10(Clicks + 1)')
    _ax.set_ylabel('Frequency')
    _ax.set_title('Distribution of Clicks (Log Scale)')
    _ax.axvline(np.log10(np.median(clicks) + 1), color='red', linestyle='--',
               label=f'Median: {np.median(clicks):,.0f}')
    _ax.legend()

    # 3. Scatter: Opens vs Clicks
    _ax = _axes2[1, 0]
    sample_idx = np.random.choice(len(opens), min(10000, len(opens)), replace=False)
    _ax.scatter(np.log10(opens[sample_idx] + 1), np.log10(clicks[sample_idx] + 1), 
               alpha=0.3, s=5, color='#8338EC')
    _ax.set_xlabel('log10(Opens + 1)')
    _ax.set_ylabel('log10(Clicks + 1)')
    _ax.set_title('Opens vs Clicks (10K sample)')

    # 4. CTR vs Opens volume (does CTR vary with scale?)
    _ax = _axes2[1, 1]
    ctr_plot = placements_with_ctr.select("ctr").to_numpy().flatten()
    _ax.scatter(np.log10(opens[sample_idx] + 1), ctr_plot[sample_idx], 
               alpha=0.3, s=5, color='#FB5607')
    _ax.set_xlabel('log10(Opens + 1)')
    _ax.set_ylabel('CTR')
    _ax.set_title('CTR vs Volume (10K sample)')
    _ax.set_ylim(0, 0.1)

    plt.tight_layout()
    plt.show()

    print("\n=== Volume Statistics ===")
    print(f"Total Opens: {opens.sum():,}")
    print(f"Total Clicks: {clicks.sum():,}")
    print(f"Overall CTR: {clicks.sum()/opens.sum():.6f}")
    print(f"\nOpens per placement: Median={np.median(opens):,.0f}, Mean={np.mean(opens):,.0f}")
    print(f"Clicks per placement: Median={np.median(clicks):,.0f}, Mean={np.mean(clicks):,.0f}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Campaign Analysis

    Let's analyze the campaigns dataset and understand how campaign attributes relate to CTR.
    """)
    return


@app.cell
def _(campaigns, pl):
    # Explore campaigns dataset
    print("Campaigns Schema:")
    print(campaigns.schema)
    print("\nSample Data:")
    display_campaigns = campaigns.head(10)

    # Analyze campaign attributes
    print("\n=== Campaign Targeting Analysis ===")

    # Target Gender distribution
    gender_dist = campaigns.group_by("target_gender").agg(pl.count().alias("count"))
    print("\nTarget Gender Distribution:")
    print(gender_dist)

    # Promoted Item distribution
    item_dist = campaigns.group_by("promoted_item").agg(pl.count().alias("count")).sort("count", descending=True)
    print("\nPromoted Item Distribution:")
    print(item_dist)

    display_campaigns
    return


@app.cell
def _(campaigns, pl, placements_with_ctr, plt, sns):
    # Merge placements with campaigns to analyze CTR by campaign attributes
    placements_campaigns = placements_with_ctr.join(
        campaigns, 
        on="campaign_id", 
        how="left"
    )

    # CTR by target gender
    ctr_by_gender = placements_campaigns.group_by("target_gender").agg([
        pl.col("ctr").mean().alias("mean_ctr"),
        pl.col("ctr").median().alias("median_ctr"),
        pl.col("ctr").std().alias("std_ctr"),
        pl.count().alias("count"),
        pl.col("approved_opens").sum().alias("total_opens"),
        pl.col("approved_clicks").sum().alias("total_clicks")
    ]).with_columns(
        (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
    ).sort("count", descending=True)

    print("CTR by Target Gender:")
    print(ctr_by_gender)

    # CTR by promoted item
    ctr_by_item = placements_campaigns.group_by("promoted_item").agg([
        pl.col("ctr").mean().alias("mean_ctr"),
        pl.col("ctr").median().alias("median_ctr"),
        pl.col("ctr").std().alias("std_ctr"),
        pl.count().alias("count"),
        pl.col("approved_opens").sum().alias("total_opens"),
        pl.col("approved_clicks").sum().alias("total_clicks")
    ]).with_columns(
        (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
    ).sort("count", descending=True)

    print("\nCTR by Promoted Item:")
    print(ctr_by_item)

    # Visualization
    _fig3, _axes3 = plt.subplots(1, 2, figsize=(14, 5))

    # Gender plot
    gender_data = ctr_by_gender.filter(pl.col("target_gender").is_not_null())
    if len(gender_data) > 0:
        genders = gender_data.select("target_gender").to_numpy().flatten()
        gender_ctrs = gender_data.select("weighted_ctr").to_numpy().flatten()
        gender_counts = gender_data.select("count").to_numpy().flatten()

        _ax = _axes3[0]
        _bars = _ax.bar(range(len(genders)), gender_ctrs * 100, color=sns.color_palette("husl", len(genders)))
        _ax.set_xticks(range(len(genders)))
        _ax.set_xticklabels(genders, rotation=45, ha='right')
        _ax.set_ylabel('CTR (%)')
        _ax.set_title('CTR by Target Gender')
        for i, (_bar, count) in enumerate(zip(_bars, gender_counts)):
            _ax.annotate(f'n={count:,}', xy=(_bar.get_x() + _bar.get_width()/2, _bar.get_height()),
                       ha='center', va='bottom', fontsize=8)

    # Item type plot
    item_data = ctr_by_item.filter(pl.col("promoted_item").is_not_null()).head(10)
    if len(item_data) > 0:
        items = item_data.select("promoted_item").to_numpy().flatten()
        item_ctrs = item_data.select("weighted_ctr").to_numpy().flatten()
        item_counts = item_data.select("count").to_numpy().flatten()

        _ax = _axes3[1]
        _bars = _ax.bar(range(len(items)), item_ctrs * 100, color=sns.color_palette("husl", len(items)))
        _ax.set_xticks(range(len(items)))
        _ax.set_xticklabels(items, rotation=45, ha='right')
        _ax.set_ylabel('CTR (%)')
        _ax.set_title('CTR by Promoted Item Type')
        for i, (_bar, count) in enumerate(zip(_bars, item_counts)):
            _ax.annotate(f'n={count:,}', xy=(_bar.get_x() + _bar.get_width()/2, _bar.get_height()),
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()
    return (placements_campaigns,)


@app.cell
def _(mo):
    mo.md("""
    ## 6. Publication Analysis

    Let's analyze how different publications perform in terms of CTR.
    """)
    return


@app.cell
def _(np, pl, placements_with_ctr, plt):
    # CTR by Publication
    ctr_by_publication = placements_with_ctr.group_by("publication_id").agg([
        pl.col("ctr").mean().alias("mean_ctr"),
        pl.col("ctr").median().alias("median_ctr"),
        pl.col("ctr").std().alias("std_ctr"),
        pl.count().alias("placement_count"),
        pl.col("approved_opens").sum().alias("total_opens"),
        pl.col("approved_clicks").sum().alias("total_clicks"),
        pl.col("campaign_id").n_unique().alias("unique_campaigns")
    ]).with_columns(
        (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
    ).sort("placement_count", descending=True)

    print(f"Total unique publications: {len(ctr_by_publication):,}")
    print("\nTop 20 publications by placement count:")
    print(ctr_by_publication.head(20))

    # Distribution of publication CTRs
    pub_ctrs = ctr_by_publication.select("weighted_ctr").to_numpy().flatten()
    pub_counts = ctr_by_publication.select("placement_count").to_numpy().flatten()

    _fig4, _axes4 = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distribution of publication CTRs
    _ax = _axes4[0, 0]
    _ax.hist(pub_ctrs[pub_ctrs < 0.1], bins=50, edgecolor='black', alpha=0.7, color='#2E86AB')
    _ax.set_xlabel('Publication CTR')
    _ax.set_ylabel('Number of Publications')
    _ax.set_title('Distribution of Publication-level CTR')
    _ax.axvline(np.median(pub_ctrs), color='red', linestyle='--', label=f'Median: {np.median(pub_ctrs):.4f}')
    _ax.legend()

    # 2. Publication volume distribution
    _ax = _axes4[0, 1]
    _ax.hist(np.log10(pub_counts + 1), bins=50, edgecolor='black', alpha=0.7, color='#A23B72')
    _ax.set_xlabel('log10(Placement Count)')
    _ax.set_ylabel('Number of Publications')
    _ax.set_title('Distribution of Publication Activity')

    # 3. CTR vs Volume for publications
    _ax = _axes4[1, 0]
    _ax.scatter(np.log10(pub_counts + 1), pub_ctrs, alpha=0.5, s=10, color='#F18F01')
    _ax.set_xlabel('log10(Placement Count)')
    _ax.set_ylabel('CTR')
    _ax.set_title('Publication CTR vs Activity Level')
    _ax.set_ylim(0, 0.15)

    # 4. Variance in CTR across publications (stability)
    pub_std = ctr_by_publication.filter(pl.col("placement_count") >= 10).select("std_ctr").to_numpy().flatten()
    _ax = _axes4[1, 1]
    _ax.hist(pub_std[~np.isnan(pub_std)], bins=50, edgecolor='black', alpha=0.7, color='#8338EC')
    _ax.set_xlabel('CTR Standard Deviation')
    _ax.set_ylabel('Number of Publications')
    _ax.set_title('CTR Variance by Publication (min 10 placements)')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Publication Tags Analysis

    Let's analyze how content categories (tags) relate to CTR.
    """)
    return


@app.cell
def _(np, pl, placements_with_ctr, plt, publication_tags):
    # Parse publication tags and analyze CTR by tag
    # The tags column contains sets like "{'Design'}" - we need to extract them

    # Join placements with tags
    placements_tags = placements_with_ctr.join(
        publication_tags,
        on="publication_id",
        how="left"
    )

    # Extract individual tags (simplified - just using the tags column)
    # We'll analyze tag frequency first
    tag_stats = placements_tags.group_by("tags").agg([
        pl.col("ctr").mean().alias("mean_ctr"),
        pl.col("ctr").median().alias("median_ctr"),
        pl.count().alias("count"),
        pl.col("approved_opens").sum().alias("total_opens"),
        pl.col("approved_clicks").sum().alias("total_clicks")
    ]).with_columns(
        (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
    ).filter(
        pl.col("count") >= 100  # Minimum sample size
    ).sort("count", descending=True)

    print("CTR by Publication Tags (min 100 placements):")
    print(tag_stats.head(30))

    # Visualize top tags by CTR
    _fig5, _axes5 = plt.subplots(1, 2, figsize=(16, 6))

    top_tags = tag_stats.head(20)
    tags_names = top_tags.select("tags").to_numpy().flatten()
    tags_ctrs = top_tags.select("weighted_ctr").to_numpy().flatten()
    tags_counts = top_tags.select("count").to_numpy().flatten()

    # 1. CTR by tag
    _ax = _axes5[0]
    _y_pos = np.arange(len(tags_names))
    _colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(tags_names)))
    _ax.barh(_y_pos, tags_ctrs * 100, color=_colors)
    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels([str(t)[:40] for t in tags_names], fontsize=8)
    _ax.set_xlabel('CTR (%)')
    _ax.set_title('CTR by Publication Tag Category (Top 20 by volume)')
    _ax.invert_yaxis()

    # 2. Volume by tag
    _ax = _axes5[1]
    _ax.barh(_y_pos, tags_counts, color=_colors)
    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels([str(t)[:40] for t in tags_names], fontsize=8)
    _ax.set_xlabel('Number of Placements')
    _ax.set_title('Placement Volume by Tag Category')
    _ax.invert_yaxis()

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Temporal Analysis

    Let's analyze how CTR varies over time.
    """)
    return


@app.cell
def _(pl, placements_with_ctr, plt):
    # Temporal analysis
    placements_temporal = placements_with_ctr.with_columns([
        pl.col("send_datetime").dt.year().alias("year"),
        pl.col("send_datetime").dt.month().alias("month"),
        pl.col("send_datetime").dt.weekday().alias("weekday"),
        pl.col("send_datetime").dt.hour().alias("hour")
    ])

    # CTR by month
    ctr_by_month = placements_temporal.group_by(["year", "month"]).agg([
        pl.col("ctr").mean().alias("mean_ctr"),
        pl.count().alias("count"),
        pl.col("approved_opens").sum().alias("total_opens"),
        pl.col("approved_clicks").sum().alias("total_clicks")
    ]).with_columns(
        (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
    ).sort(["year", "month"])

    print("CTR by Month:")
    print(ctr_by_month)

    # CTR by day of week
    ctr_by_weekday = placements_temporal.group_by("weekday").agg([
        pl.col("ctr").mean().alias("mean_ctr"),
        pl.count().alias("count"),
        pl.col("approved_opens").sum().alias("total_opens"),
        pl.col("approved_clicks").sum().alias("total_clicks")
    ]).with_columns(
        (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
    ).sort("weekday")

    # CTR by hour
    ctr_by_hour = placements_temporal.group_by("hour").agg([
        pl.col("ctr").mean().alias("mean_ctr"),
        pl.count().alias("count"),
        pl.col("approved_opens").sum().alias("total_opens"),
        pl.col("approved_clicks").sum().alias("total_clicks")
    ]).with_columns(
        (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
    ).sort("hour")

    # Visualization
    _fig6, _axes6 = plt.subplots(2, 2, figsize=(14, 10))

    # 1. CTR over time (monthly)
    _ax = _axes6[0, 0]
    months = ctr_by_month.select(pl.concat_str([pl.col("year"), pl.lit("-"), pl.col("month").cast(pl.Utf8).str.zfill(2)])).to_numpy().flatten()
    monthly_ctrs = ctr_by_month.select("weighted_ctr").to_numpy().flatten()
    _ax.plot(range(len(months)), monthly_ctrs * 100, marker='o', linewidth=2, color='#2E86AB')
    _ax.set_xticks(range(len(months)))
    _ax.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    _ax.set_ylabel('CTR (%)')
    _ax.set_title('CTR Trend Over Time (Monthly)')
    _ax.grid(True, alpha=0.3)

    # 2. Placement volume over time
    _ax = _axes6[0, 1]
    monthly_counts = ctr_by_month.select("count").to_numpy().flatten()
    _ax.bar(range(len(months)), monthly_counts, color='#A23B72', alpha=0.7)
    _ax.set_xticks(range(len(months)))
    _ax.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    _ax.set_ylabel('Number of Placements')
    _ax.set_title('Placement Volume Over Time')

    # 3. CTR by day of week
    _ax = _axes6[1, 0]
    weekdays = ['EvryDay','Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekday_ctrs = ctr_by_weekday.select("weighted_ctr").to_numpy().flatten()
    _ax.bar(weekdays, weekday_ctrs * 100, color='#F18F01', alpha=0.8)
    _ax.set_ylabel('CTR (%)')
    _ax.set_title('CTR by Day of Week')

    # 4. CTR by hour
    _ax = _axes6[1, 1]
    hours = ctr_by_hour.select("hour").to_numpy().flatten()
    hourly_ctrs = ctr_by_hour.select("weighted_ctr").to_numpy().flatten()
    _ax.plot(hours, hourly_ctrs * 100, marker='o', linewidth=2, color='#8338EC')
    _ax.set_xlabel('Hour of Day')
    _ax.set_ylabel('CTR (%)')
    _ax.set_title('CTR by Hour of Day')
    _ax.set_xticks(range(0, 24, 2))
    _ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return (placements_temporal,)


@app.cell
def _(mo):
    mo.md("""
    ## 9. Publication-Campaign Interaction Analysis

    Let's understand how the combination of publication and campaign affects CTR.
    """)
    return


@app.cell
def _(np, pl, placements_with_ctr, plt):
    # Analyze publication-campaign combinations
    pub_campaign_stats = placements_with_ctr.group_by(["publication_id", "campaign_id"]).agg([
        pl.col("ctr").mean().alias("mean_ctr"),
        pl.count().alias("placement_count"),
        pl.col("approved_opens").sum().alias("total_opens"),
        pl.col("approved_clicks").sum().alias("total_clicks")
    ]).with_columns(
        (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
    )

    print(f"Unique publication-campaign combinations: {len(pub_campaign_stats):,}")

    # How many times does each combination appear?
    combo_freq = pub_campaign_stats.group_by("placement_count").agg(pl.count().alias("num_combos")).sort("placement_count")
    print("\nFrequency of pub-campaign combinations:")
    print(combo_freq.head(20))

    # For publications with multiple campaigns, how much does CTR vary?
    pub_campaign_variance = pub_campaign_stats.filter(
        pl.col("placement_count") >= 3
    ).group_by("publication_id").agg([
        pl.col("weighted_ctr").mean().alias("mean_ctr_across_campaigns"),
        pl.col("weighted_ctr").std().alias("std_ctr_across_campaigns"),
        pl.col("campaign_id").n_unique().alias("num_campaigns"),
        pl.count().alias("num_combos")
    ]).filter(pl.col("num_campaigns") >= 3)

    # Same for campaigns across publications
    campaign_pub_variance = pub_campaign_stats.filter(
        pl.col("placement_count") >= 3
    ).group_by("campaign_id").agg([
        pl.col("weighted_ctr").mean().alias("mean_ctr_across_pubs"),
        pl.col("weighted_ctr").std().alias("std_ctr_across_pubs"),
        pl.col("publication_id").n_unique().alias("num_publications"),
        pl.count().alias("num_combos")
    ]).filter(pl.col("num_publications") >= 3)

    _fig7, _axes7 = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distribution of CTR variance across campaigns (for each publication)
    _ax = _axes7[0, 0]
    pub_variance_values = pub_campaign_variance.select("std_ctr_across_campaigns").to_numpy().flatten()
    pub_variance_values = pub_variance_values[~np.isnan(pub_variance_values)]
    _ax.hist(pub_variance_values, bins=50, edgecolor='black', alpha=0.7, color='#2E86AB')
    _ax.set_xlabel('Std Dev of CTR across Campaigns')
    _ax.set_ylabel('Number of Publications')
    _ax.set_title('How much does CTR vary for a publication across campaigns?')
    _ax.axvline(np.median(pub_variance_values), color='red', linestyle='--', label=f'Median: {np.median(pub_variance_values):.4f}')
    _ax.legend()

    # 2. Distribution of CTR variance across publications (for each campaign)
    _ax = _axes7[0, 1]
    camp_variance_values = campaign_pub_variance.select("std_ctr_across_pubs").to_numpy().flatten()
    camp_variance_values = camp_variance_values[~np.isnan(camp_variance_values)]
    _ax.hist(camp_variance_values, bins=50, edgecolor='black', alpha=0.7, color='#A23B72')
    _ax.set_xlabel('Std Dev of CTR across Publications')
    _ax.set_ylabel('Number of Campaigns')
    _ax.set_title('How much does CTR vary for a campaign across publications?')
    _ax.axvline(np.median(camp_variance_values), color='red', linestyle='--', label=f'Median: {np.median(camp_variance_values):.4f}')
    _ax.legend()

    # 3. Mean vs Std for publications
    _ax = _axes7[1, 0]
    _pub_means_plot = pub_campaign_variance.select("mean_ctr_across_campaigns").to_numpy().flatten()
    _pub_stds_plot = pub_campaign_variance.select("std_ctr_across_campaigns").to_numpy().flatten()
    _ax.scatter(_pub_means_plot, _pub_stds_plot, alpha=0.5, s=10, color='#F18F01')
    _ax.set_xlabel('Mean CTR')
    _ax.set_ylabel('Std Dev CTR')
    _ax.set_title('Publication: Mean vs Variance in CTR across Campaigns')
    _ax.set_xlim(0, 0.1)
    _ax.set_ylim(0, 0.05)

    # 4. Mean vs Std for campaigns
    _ax = _axes7[1, 1]
    _camp_means_plot = campaign_pub_variance.select("mean_ctr_across_pubs").to_numpy().flatten()
    _camp_stds_plot = campaign_pub_variance.select("std_ctr_across_pubs").to_numpy().flatten()
    _ax.scatter(_camp_means_plot, _camp_stds_plot, alpha=0.5, s=10, color='#8338EC')
    _ax.set_xlabel('Mean CTR')
    _ax.set_ylabel('Std Dev CTR')
    _ax.set_title('Campaign: Mean vs Variance in CTR across Publications')
    _ax.set_xlim(0, 0.1)
    _ax.set_ylim(0, 0.05)

    plt.tight_layout()
    plt.show()

    print(f"\n=== Variance Analysis ===")
    print(f"Publications with 3+ campaigns: {len(pub_campaign_variance):,}")
    print(f"  Median CTR std across campaigns: {np.median(pub_variance_values):.4f}")
    print(f"Campaigns with 3+ publications: {len(campaign_pub_variance):,}")
    print(f"  Median CTR std across publications: {np.median(camp_variance_values):.4f}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. Correlation Analysis

    Let's compute correlations between features and CTR to identify the most predictive features.
    """)
    return


@app.cell
def _(campaigns, np, pl, placements_temporal, plt, publication_tags, sns):
    # Create a feature matrix for correlation analysis
    # We'll use publication and campaign level aggregates

    # Prepare campaigns with encoded features
    campaigns_encoded = campaigns.with_columns([
        pl.col("target_gender").fill_null("unknown"),
        pl.col("promoted_item").fill_null("unknown")
    ])

    # Create numerical features from campaigns
    gender_map = {"unknown": 0, "no_preference": 1, "balanced": 2, "predominantly_male": 3, "predominantly_female": 4}
    item_map = {"unknown": 0, "": 0, "product": 1, "service": 2, "newsletter": 3, "knowledge_product": 4, "event": 5, "other": 6}

    campaigns_features = campaigns_encoded.with_columns([
        pl.col("target_gender").replace(gender_map, default=0).alias("gender_encoded"),
        pl.col("promoted_item").replace(item_map, default=0).alias("item_encoded"),
        # Count target income ranges
        pl.col("target_incomes").str.count_matches(r"range_").alias("num_income_targets"),
        # Count target age ranges
        pl.col("target_ages").str.count_matches(r"range_").alias("num_age_targets")
    ])

    # Join all features
    full_data = placements_temporal.join(
        campaigns_features.select(["campaign_id", "gender_encoded", "item_encoded", "num_income_targets", "num_age_targets"]),
        on="campaign_id",
        how="left"
    ).join(
        publication_tags.select(["publication_id", "tag_ids"]),
        on="publication_id",
        how="left"
    ).with_columns([
        # Number of tags for publication
        pl.col("tag_ids").str.count_matches(r"\d+").fill_null(0).alias("num_tags")
    ])

    # Select numerical features for correlation
    correlation_features = full_data.select([
        "ctr",
        "approved_opens",
        "approved_clicks",
        "year",
        "month",
        "weekday",
        "hour",
        "gender_encoded",
        "item_encoded",
        "num_income_targets",
        "num_age_targets",
        "num_tags"
    ]).drop_nulls()

    # Convert to numpy for correlation calculation
    feature_names = correlation_features.columns
    data_matrix = correlation_features.to_numpy()

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(data_matrix.T)

    # Create correlation heatmap
    _fig8, _ax8 = plt.subplots(figsize=(12, 10))
    _mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, 
                mask=_mask,
                xticklabels=feature_names, 
                yticklabels=feature_names, 
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                square=True,
                ax=_ax8)
    _ax8.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

    # Print correlations with CTR
    print("\n=== Correlations with CTR ===")
    ctr_idx = feature_names.index("ctr")
    correlations_with_ctr = [(feature_names[i], corr_matrix[ctr_idx, i]) for i in range(len(feature_names)) if i != ctr_idx]
    correlations_with_ctr.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in correlations_with_ctr:
        print(f"  {feat}: {corr:.4f}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 11. Variance Decomposition Analysis

    Let's understand how much variance in CTR is explained by publications vs campaigns.
    """)
    return


@app.cell
def _(pl, placements_with_ctr, plt):
    # Variance decomposition using simple ANOVA-style analysis
    # We want to understand: How much of CTR variance is due to publication vs campaign?

    overall_ctr = placements_with_ctr.select("ctr").mean().item()
    overall_var = placements_with_ctr.select("ctr").var().item()

    print(f"Overall CTR mean: {overall_ctr:.6f}")
    print(f"Overall CTR variance: {overall_var:.8f}")

    # Publication effect: variance of publication means
    pub_means_decomp = placements_with_ctr.group_by("publication_id").agg(
        pl.col("ctr").mean().alias("pub_mean_ctr")
    )
    var_between_pubs = pub_means_decomp.select("pub_mean_ctr").var().item()

    # Campaign effect: variance of campaign means
    camp_means_decomp = placements_with_ctr.group_by("campaign_id").agg(
        pl.col("ctr").mean().alias("camp_mean_ctr")
    )
    var_between_camps = camp_means_decomp.select("camp_mean_ctr").var().item()

    # Interaction: variance of pub-campaign combination means
    combo_means_decomp = placements_with_ctr.group_by(["publication_id", "campaign_id"]).agg(
        pl.col("ctr").mean().alias("combo_mean_ctr")
    )
    var_between_combos = combo_means_decomp.select("combo_mean_ctr").var().item()

    print(f"\n=== Variance Decomposition ===")
    print(f"Variance between publications: {var_between_pubs:.8f} ({100*var_between_pubs/overall_var:.1f}% of total)")
    print(f"Variance between campaigns: {var_between_camps:.8f} ({100*var_between_camps/overall_var:.1f}% of total)")
    print(f"Variance between pub-campaign combos: {var_between_combos:.8f} ({100*var_between_combos/overall_var:.1f}% of total)")

    # Visualize
    _fig9, _axes9 = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Publication means distribution
    _ax = _axes9[0]
    pub_ctr_values_decomp = pub_means_decomp.select("pub_mean_ctr").to_numpy().flatten()
    _ax.hist(pub_ctr_values_decomp[pub_ctr_values_decomp < 0.1], bins=50, edgecolor='black', alpha=0.7, color='#2E86AB')
    _ax.axvline(overall_ctr, color='red', linestyle='--', linewidth=2, label=f'Overall: {overall_ctr:.4f}')
    _ax.set_xlabel('Mean CTR')
    _ax.set_ylabel('Number of Publications')
    _ax.set_title(f'Distribution of Publication Mean CTR\nVar explains {100*var_between_pubs/overall_var:.1f}% of total')
    _ax.legend()

    # 2. Campaign means distribution
    _ax = _axes9[1]
    camp_ctr_values_decomp = camp_means_decomp.select("camp_mean_ctr").to_numpy().flatten()
    _ax.hist(camp_ctr_values_decomp[camp_ctr_values_decomp < 0.1], bins=50, edgecolor='black', alpha=0.7, color='#A23B72')
    _ax.axvline(overall_ctr, color='red', linestyle='--', linewidth=2, label=f'Overall: {overall_ctr:.4f}')
    _ax.set_xlabel('Mean CTR')
    _ax.set_ylabel('Number of Campaigns')
    _ax.set_title(f'Distribution of Campaign Mean CTR\nVar explains {100*var_between_camps/overall_var:.1f}% of total')
    _ax.legend()

    # 3. Combination means distribution
    _ax = _axes9[2]
    combo_ctr_values_decomp = combo_means_decomp.select("combo_mean_ctr").to_numpy().flatten()
    _ax.hist(combo_ctr_values_decomp[combo_ctr_values_decomp < 0.1], bins=50, edgecolor='black', alpha=0.7, color='#F18F01')
    _ax.axvline(overall_ctr, color='red', linestyle='--', linewidth=2, label=f'Overall: {overall_ctr:.4f}')
    _ax.set_xlabel('Mean CTR')
    _ax.set_ylabel('Number of Combinations')
    _ax.set_title(f'Distribution of Pub-Campaign Mean CTR\nVar explains {100*var_between_combos/overall_var:.1f}% of total')
    _ax.legend()

    plt.tight_layout()
    plt.show()
    return overall_var, var_between_camps, var_between_combos, var_between_pubs


@app.cell
def _(mo):
    mo.md("""
    ## 12. Statistical Tests for Feature Importance

    Let's perform statistical tests to validate the significance of feature relationships.
    """)
    return


@app.cell
def _(np, pl, placements_campaigns, stats):
    # Statistical tests for categorical features

    # 1. ANOVA for target_gender effect on CTR
    gender_groups = placements_campaigns.filter(
        pl.col("target_gender").is_not_null()
    ).group_by("target_gender").agg(
        pl.col("ctr").alias("ctr_values")
    )

    gender_group_dict = {row["target_gender"]: row["ctr_values"] for row in gender_groups.iter_rows(named=True)}

    gender_anova_result = None
    if len(gender_group_dict) >= 2:
        _f_stat, _p_value = stats.f_oneway(*[np.array(v) for v in gender_group_dict.values()])
        gender_anova_result = (_f_stat, _p_value)
        print(f"ANOVA: Target Gender effect on CTR")
        print(f"  F-statistic: {_f_stat:.4f}")
        print(f"  p-value: {_p_value:.2e}")
        print(f"  Significant: {'Yes' if _p_value < 0.05 else 'No'}")

    # 2. ANOVA for promoted_item effect on CTR
    item_groups = placements_campaigns.filter(
        pl.col("promoted_item").is_not_null() & (pl.col("promoted_item") != "")
    ).group_by("promoted_item").agg(
        pl.col("ctr").alias("ctr_values")
    )

    item_group_dict = {row["promoted_item"]: row["ctr_values"] for row in item_groups.iter_rows(named=True)}

    item_anova_result = None
    if len(item_group_dict) >= 2:
        _f_stat, _p_value = stats.f_oneway(*[np.array(v) for v in item_group_dict.values()])
        item_anova_result = (_f_stat, _p_value)
        print(f"\nANOVA: Promoted Item effect on CTR")
        print(f"  F-statistic: {_f_stat:.4f}")
        print(f"  p-value: {_p_value:.2e}")
        print(f"  Significant: {'Yes' if _p_value < 0.05 else 'No'}")

    # 3. Effect sizes
    print("\n=== Effect Sizes (Cohen's d) ===")

    # Calculate effect sizes between groups
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

    # Compare male vs female targeting
    effect_size_gender = None
    if 'predominantly_male' in gender_group_dict and 'predominantly_female' in gender_group_dict:
        effect_size_gender = cohens_d(np.array(gender_group_dict['predominantly_male']), 
                     np.array(gender_group_dict['predominantly_female']))
        print(f"Male vs Female targeting: d = {effect_size_gender:.4f}")

    # Compare product vs service
    effect_size_item = None
    if 'product' in item_group_dict and 'service' in item_group_dict:
        effect_size_item = cohens_d(np.array(item_group_dict['product']), 
                     np.array(item_group_dict['service']))
        print(f"Product vs Service: d = {effect_size_item:.4f}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 13. Feature Engineering Recommendations

    Based on our analysis, here are the recommended features for CTR prediction.
    """)
    return


@app.cell
def _(
    mo,
    overall_var,
    var_between_camps,
    var_between_combos,
    var_between_pubs,
):
    # Summary of findings and recommendations

    _recommendations = f"""
    ## Summary of Findings

    ### 1. Target Variable (CTR) Characteristics
    - CTR follows a highly skewed distribution with many zeros
    - Consider using log-transform or beta distribution modeling
    - Median CTR is typically much lower than mean due to outliers

    ### 2. Variance Explainability
    - **Publication effect**: {100*var_between_pubs/overall_var:.1f}% of total variance
    - **Campaign effect**: {100*var_between_camps/overall_var:.1f}% of total variance  
    - **Combined effect**: {100*var_between_combos/overall_var:.1f}% of total variance

    The pub-campaign combination explains most variance, suggesting strong interaction effects.

    ### 3. Recommended Features for Modeling

    #### Publication Features:
    - Publication ID (as embedding or categorical)
    - Historical CTR statistics (mean, std, count)
    - Content tags (multi-hot encoding)
    - Number of unique campaigns served
    - Audience size indicators (avg opens)

    #### Campaign Features:
    - Campaign ID (as embedding or categorical)
    - Target gender preference
    - Promoted item type
    - Number of income targets
    - Number of age targets
    - Historical CTR across publications
    - Advertiser embedding (hierarchical)

    #### Temporal Features:
    - Day of week
    - Hour of day
    - Month (seasonality)
    - Time since campaign start

    #### Interaction Features:
    - Publication-Campaign historical performance (if available)
    - Tag-Item type matching score
    - Audience-Target alignment score

    ### 4. Modeling Recommendations

    1. **Baseline Models**:
       - Publication average CTR
       - Campaign average CTR
       - Global average

    2. **Candidate Models**:
       - **Gradient Boosting** (XGBoost, LightGBM): Good for tabular data with interactions
       - **Factorization Machines**: Natural for pub-campaign interactions
       - **Two-Tower Neural Network**: Separate encoders for publication and campaign
       - **Beta Regression**: Handles bounded [0,1] target naturally

    3. **Handling Aggregated Data**:
       - Weight samples by number of opens (more reliable CTR estimates)
       - Consider using weighted loss functions
       - May want to filter low-volume combinations

    4. **Evaluation Strategy**:
       - Use time-based splits (train on older, test on newer)
       - Weight evaluation by opens volume
       - Consider both MAE and AUC (for binary click/not-click framing)
    """

    mo.md(_recommendations)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 14. Next Steps

    1. **Feature Engineering Pipeline**: Create features based on the analysis above
    2. **Embedding Analysis**: If using post_embeddings.csv, analyze how content embeddings relate to CTR
    3. **Model Training**: Implement baseline and candidate models
    4. **Cross-Validation**: Set up proper temporal validation strategy
    5. **Production Considerations**: Define inference pipeline for real-time predictions
    """)
    return


if __name__ == "__main__":
    app.run()
