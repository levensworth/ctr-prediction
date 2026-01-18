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
    from typing import Tuple, Optional
    from dataclasses import dataclass

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
def _(np, pl, plt, sns):
    """Helper functions for data loading, processing and visualization."""

    DATA_PATH = "../data/"

    def load_all_datasets() -> dict:
        """Load all datasets from the data directory."""
        return {
            "placements": pl.read_csv(DATA_PATH + "placements.csv"),
            "campaigns": pl.read_csv(DATA_PATH + "campaigns.csv"),
            "advertisers": pl.read_csv(DATA_PATH + "advertisers.csv"),
            "publications": pl.read_csv(DATA_PATH + "publication_mets.csv"),
            "publication_tags": pl.read_csv(DATA_PATH + "publication_tags.csv"),
        }

    def compute_ctr(df: pl.DataFrame) -> pl.DataFrame:
        """Add CTR column and datetime column to placements dataframe."""
        return df.with_columns([
            (pl.col("approved_clicks") / pl.col("approved_opens")).alias("ctr"),
            pl.col("post_send_at").str.to_datetime().alias("send_datetime")
        ]).filter(
            pl.col("approved_opens") > 0
        )

    def add_temporal_columns(df: pl.DataFrame) -> pl.DataFrame:
        """Add year, month, weekday, and hour columns from send_datetime."""
        return df.with_columns([
            pl.col("send_datetime").dt.year().alias("year"),
            pl.col("send_datetime").dt.month().alias("month"),
            pl.col("send_datetime").dt.weekday().alias("weekday"),
            pl.col("send_datetime").dt.hour().alias("hour")
        ])

    def plot_histogram(
        data: np.ndarray,
        ax: plt.Axes,
        bins: int = 50,
        color: str = '#2E86AB',
        xlabel: str = '',
        ylabel: str = 'Frequency',
        title: str = '',
        log_scale: bool = False,
        median_line: bool = False,
        mean_line: bool = False
    ) -> None:
        """Plot a histogram with optional median/mean lines."""
        ax.hist(data, bins=bins, edgecolor='black', alpha=0.7, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log_scale:
            ax.set_yscale('log')
        if median_line:
            ax.axvline(np.median(data), color='red', linestyle='--', 
                      label=f'Median: {np.median(data):.4f}')
        if mean_line:
            ax.axvline(np.mean(data), color='green', linestyle='--', 
                      label=f'Mean: {np.mean(data):.4f}')
        if median_line or mean_line:
            ax.legend()

    def plot_bar_chart(
        labels: list,
        values: np.ndarray,
        ax: plt.Axes,
        color: str = '#2E86AB',
        xlabel: str = '',
        ylabel: str = '',
        title: str = '',
        rotation: int = 45,
        counts: np.ndarray = None,
        horizontal: bool = False
    ) -> None:
        """Plot a bar chart with optional count annotations."""
        if horizontal:
            y_pos = np.arange(len(labels))
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
            ax.barh(y_pos, values, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([str(l)[:40] for l in labels], fontsize=8)
            ax.set_xlabel(xlabel)
            ax.set_title(title)
            ax.invert_yaxis()
        else:
            colors = sns.color_palette("husl", len(labels)) if len(labels) <= 20 else color
            bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.8)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=rotation, ha='right')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            if counts is not None:
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    ax.annotate(f'n={count:,}', 
                               xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               ha='center', va='bottom', fontsize=8)

    def plot_scatter(
        x: np.ndarray,
        y: np.ndarray,
        ax: plt.Axes,
        color: str = '#8338EC',
        alpha: float = 0.3,
        size: int = 5,
        xlabel: str = '',
        ylabel: str = '',
        title: str = '',
        xlim: tuple = None,
        ylim: tuple = None
    ) -> None:
        """Plot a scatter plot."""
        ax.scatter(x, y, alpha=alpha, s=size, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

    def compute_distribution_stats(data: np.ndarray) -> dict:
        """Compute common distribution statistics."""
        return {
            "count": len(data),
            "min": np.min(data),
            "max": np.max(data),
            "mean": np.mean(data),
            "median": np.median(data),
            "std": np.std(data),
            "p25": np.percentile(data, 25),
            "p75": np.percentile(data, 75),
            "p95": np.percentile(data, 95),
        }

    def print_stats(stats_dict: dict, title: str = "Statistics") -> None:
        """Print statistics in a formatted way."""
        print(f"\n=== {title} ===")
        for key, value in stats_dict.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value:,}")

    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size between two groups."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

    return (
        add_temporal_columns,
        cohens_d,
        compute_ctr,
        compute_distribution_stats,
        load_all_datasets,
        plot_bar_chart,
        plot_histogram,
        plot_scatter,
        print_stats,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1. Data Loading

    Let's load all the datasets and examine their structure.
    """)
    return


@app.cell
def _(compute_ctr, load_all_datasets):
    # Load all datasets
    datasets = load_all_datasets()

    placements = datasets["placements"]
    campaigns = datasets["campaigns"]
    advertisers = datasets["advertisers"]
    publications = datasets["publications"]
    publication_tags = datasets["publication_tags"]

    # Compute CTR for placements
    placements_with_ctr = compute_ctr(placements)

    print(f"Placements: {placements.shape[0]:,} rows, {placements.shape[1]} columns")
    print(f"Campaigns: {campaigns.shape[0]:,} rows, {campaigns.shape[1]} columns")
    print(f"Advertisers: {advertisers.shape[0]:,} rows, {advertisers.shape[1]} columns")
    print(f"Publications: {publications.shape[0]:,} rows, {publications.shape[1]} columns")
    print(f"Publication Tags: {publication_tags.shape[0]:,} rows, {publication_tags.shape[1]} columns")
    print(f"Placements with CTR (after filtering): {placements_with_ctr.shape[0]:,} rows")
    return campaigns, placements, placements_with_ctr, publication_tags


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
def _(placements_with_ctr):
    # Display placements schema and sample
    print("Placements Schema:")
    print(placements_with_ctr.schema)
    print("\nSample Data:")
    placements_with_ctr.head(10)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Distribution Analysis: Posts per Publication

    Understanding how posts are distributed across publications helps us identify
    high-volume publishers vs. long-tail publishers.
    """)
    return


@app.cell
def _(
    compute_distribution_stats,
    np,
    pl,
    placements_with_ctr,
    plot_histogram,
    plt,
    print_stats,
):
    def analyze_posts_per_publication(df: pl.DataFrame) -> pl.DataFrame:
        """Analyze the distribution of posts per publication."""
        return df.group_by("publication_id").agg([
            pl.col("post_id").n_unique().alias("post_count"),
            pl.col("ctr").mean().alias("mean_ctr"),
            pl.col("approved_opens").sum().alias("total_opens"),
        ]).sort("post_count", descending=True)

    def analyze_avg_time_between_posts(df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate the average time between posts for each publication.
        Returns DataFrame with publication_id and avg_days_between_posts.
        """
        # Sort by publication and datetime, then calculate time diffs
        sorted_df = df.sort(["publication_id", "send_datetime"])

        # Calculate time difference between consecutive posts per publication
        with_diff = sorted_df.with_columns([
            pl.col("send_datetime").diff().over("publication_id").alias("time_diff")
        ])

        # Convert to days and compute average per publication
        return with_diff.filter(
            pl.col("time_diff").is_not_null()
        ).with_columns([
            (pl.col("time_diff").dt.total_hours() / 24.0).alias("days_between")
        ]).group_by("publication_id").agg([
            pl.col("days_between").mean().alias("avg_days_between_posts"),
            pl.col("days_between").median().alias("median_days_between_posts"),
            pl.col("days_between").std().alias("std_days_between_posts"),
            pl.col("days_between").min().alias("min_days_between_posts"),
            pl.col("days_between").max().alias("max_days_between_posts"),
            pl.count().alias("num_intervals"),
        ]).filter(pl.col("num_intervals") >= 2)  # Need at least 2 intervals for meaningful avg

    posts_per_pub = analyze_posts_per_publication(placements_with_ctr)
    post_counts = posts_per_pub.select("post_count").to_numpy().flatten()

    time_between_posts = analyze_avg_time_between_posts(placements_with_ctr)
    avg_days_between = time_between_posts.select("avg_days_between_posts").to_numpy().flatten()
    median_days_between = time_between_posts.select("median_days_between_posts").to_numpy().flatten()

    # Create 2x3 subplot grid
    fig_posts_pub, axes_posts_pub = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: Post count distributions
    # Distribution histogram
    plot_histogram(
        post_counts[post_counts <= np.percentile(post_counts, 95)],
        axes_posts_pub[0, 0],
        bins=50,
        color='#2E86AB',
        xlabel='Number of Posts',
        ylabel='Number of Publications',
        title='Posts per Publication Distribution (up to 95th percentile)',
        median_line=True
    )

    # Log-scale distribution
    plot_histogram(
        np.log10(post_counts + 1),
        axes_posts_pub[0, 1],
        bins=50,
        color='#A23B72',
        xlabel='log10(Post Count + 1)',
        ylabel='Number of Publications',
        title='Posts per Publication (Log Scale)',
        median_line=True
    )

    # Cumulative distribution
    sorted_counts = np.sort(post_counts)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    axes_posts_pub[0, 2].plot(sorted_counts, cumulative, color='#F18F01', linewidth=2)
    axes_posts_pub[0, 2].set_xlabel('Number of Posts')
    axes_posts_pub[0, 2].set_ylabel('Cumulative Proportion')
    axes_posts_pub[0, 2].set_title('CDF: Posts per Publication')
    axes_posts_pub[0, 2].set_xscale('log')
    axes_posts_pub[0, 2].grid(True, alpha=0.3)

    # Row 2: Time between posts distributions
    # Filter out extreme outliers for better visualization
    avg_days_filtered = avg_days_between[
        (avg_days_between >= 0) & (avg_days_between <= np.percentile(avg_days_between, 95))
    ]

    # Average days between posts histogram
    plot_histogram(
        avg_days_filtered,
        axes_posts_pub[1, 0],
        bins=50,
        color='#3A86FF',
        xlabel='Average Days Between Posts',
        ylabel='Number of Publications',
        title='Avg Time Between Posts Distribution (up to 95th pctl)',
        median_line=True
    )

    # Log-scale distribution of average days
    avg_days_positive = avg_days_between[avg_days_between > 0]
    plot_histogram(
        np.log10(avg_days_positive + 0.1),
        axes_posts_pub[1, 1],
        bins=50,
        color='#FF006E',
        xlabel='log10(Avg Days Between Posts)',
        ylabel='Number of Publications',
        title='Avg Time Between Posts (Log Scale)',
        median_line=True
    )

    # Scatter: Post count vs Avg days between posts
    merged_stats = posts_per_pub.join(time_between_posts, on="publication_id", how="inner")
    scatter_post_counts = merged_stats.select("post_count").to_numpy().flatten()
    scatter_avg_days = merged_stats.select("avg_days_between_posts").to_numpy().flatten()

    axes_posts_pub[1, 2].scatter(
        np.log10(scatter_post_counts + 1),
        np.log10(scatter_avg_days + 0.1),
        alpha=0.4, s=10, color='#8338EC'
    )
    axes_posts_pub[1, 2].set_xlabel('log10(Post Count)')
    axes_posts_pub[1, 2].set_ylabel('log10(Avg Days Between Posts)')
    axes_posts_pub[1, 2].set_title('Post Frequency vs Volume')
    axes_posts_pub[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print statistics
    stats_posts_pub = compute_distribution_stats(post_counts)
    print_stats(stats_posts_pub, "Posts per Publication Statistics")

    stats_time_between = compute_distribution_stats(avg_days_between)
    print_stats(stats_time_between, "Avg Days Between Posts Statistics")

    print(f"\nTop 10 publications by post count:")
    print(posts_per_pub.head(10))

    print(f"\nMost frequent posters (lowest avg days between posts):")
    print(time_between_posts.sort("avg_days_between_posts").head(10))

    print(f"\nLeast frequent posters (highest avg days between posts, min 2 intervals):")
    print(time_between_posts.sort("avg_days_between_posts", descending=True).head(10))
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Distribution Analysis: Placements per Campaign

    Understanding how placements are distributed across campaigns reveals
    which campaigns have the most reach.
    """)
    return


@app.cell
def _(
    compute_distribution_stats,
    np,
    pl,
    placements_with_ctr,
    plot_histogram,
    plt,
    print_stats,
):
    def analyze_placements_per_campaign(df: pl.DataFrame) -> pl.DataFrame:
        """Analyze the distribution of placements per campaign."""
        return df.group_by("campaign_id").agg([
            pl.count().alias("placement_count"),
            pl.col("publication_id").n_unique().alias("unique_publications"),
            pl.col("ctr").mean().alias("mean_ctr"),
            pl.col("approved_opens").sum().alias("total_opens"),
            pl.col("approved_clicks").sum().alias("total_clicks"),
        ]).with_columns(
            (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
        ).sort("placement_count", descending=True)

    placements_per_campaign = analyze_placements_per_campaign(placements_with_ctr)
    placement_counts = placements_per_campaign.select("placement_count").to_numpy().flatten()

    fig_place_camp, axes_place_camp = plt.subplots(1, 3, figsize=(16, 5))

    # Distribution histogram
    plot_histogram(
        placement_counts[placement_counts <= np.percentile(placement_counts, 95)],
        axes_place_camp[0],
        bins=50,
        color='#3A86FF',
        xlabel='Number of Placements',
        ylabel='Number of Campaigns',
        title='Placements per Campaign Distribution (up to 95th percentile)',
        median_line=True
    )

    # Log-scale distribution
    plot_histogram(
        np.log10(placement_counts + 1),
        axes_place_camp[1],
        bins=50,
        color='#FF006E',
        xlabel='log10(Placement Count + 1)',
        ylabel='Number of Campaigns',
        title='Placements per Campaign (Log Scale)',
        median_line=True
    )

    # Scatter: placements vs unique publications
    unique_pubs = placements_per_campaign.select("unique_publications").to_numpy().flatten()
    axes_place_camp[2].scatter(
        np.log10(placement_counts + 1),
        np.log10(unique_pubs + 1),
        alpha=0.5, s=10, color='#8338EC'
    )
    axes_place_camp[2].set_xlabel('log10(Placement Count)')
    axes_place_camp[2].set_ylabel('log10(Unique Publications)')
    axes_place_camp[2].set_title('Placements vs Unique Publications per Campaign')
    axes_place_camp[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    stats_place_camp = compute_distribution_stats(placement_counts)
    print_stats(stats_place_camp, "Placements per Campaign Statistics")

    print(f"\nTop 10 campaigns by placement count:")
    print(placements_per_campaign.head(10))
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Historical Analysis: Publications Running Campaigns

    Let's analyze how many unique publications have run campaigns over time
    and see if this number is growing.
    """)
    return


@app.cell
def _(add_temporal_columns, np, pl, placements_with_ctr, plt):
    def analyze_publication_adoption_over_time(df: pl.DataFrame) -> tuple:
        """
        Analyze how many publications have run campaigns historically.
        Returns cumulative new publications and monthly active publications.
        """
        df_temporal = add_temporal_columns(df)

        # Get first campaign date for each publication
        first_campaign_by_pub = df_temporal.group_by("publication_id").agg([
            pl.col("send_datetime").min().alias("first_campaign_date")
        ]).with_columns([
            pl.col("first_campaign_date").dt.year().alias("first_year"),
            pl.col("first_campaign_date").dt.month().alias("first_month"),
        ])

        # Cumulative count of publications over time
        cumulative_pubs = first_campaign_by_pub.group_by(["first_year", "first_month"]).agg([
            pl.count().alias("new_publications")
        ]).sort(["first_year", "first_month"]).with_columns([
            pl.col("new_publications").cum_sum().alias("cumulative_publications")
        ])

        # Monthly active publications
        monthly_active = df_temporal.group_by(["year", "month"]).agg([
            pl.col("publication_id").n_unique().alias("active_publications"),
            pl.col("campaign_id").n_unique().alias("active_campaigns"),
            pl.count().alias("total_placements"),
        ]).sort(["year", "month"])

        return cumulative_pubs, monthly_active, first_campaign_by_pub

    cumulative_pubs, monthly_active, first_campaign_by_pub = analyze_publication_adoption_over_time(placements_with_ctr)

    fig_adoption, axes_adoption = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Cumulative publications over time
    ax = axes_adoption[0, 0]
    months_cumul = cumulative_pubs.select(
        pl.concat_str([
            pl.col("first_year").cast(pl.Utf8),
            pl.lit("-"),
            pl.col("first_month").cast(pl.Utf8).str.zfill(2)
        ])
    ).to_numpy().flatten()
    cumul_values = cumulative_pubs.select("cumulative_publications").to_numpy().flatten()
    ax.plot(range(len(months_cumul)), cumul_values, marker='o', linewidth=2, color='#2E86AB')
    ax.set_xticks(range(0, len(months_cumul), max(1, len(months_cumul)//10)))
    ax.set_xticklabels([months_cumul[i] for i in range(0, len(months_cumul), max(1, len(months_cumul)//10))], 
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Cumulative Publications')
    ax.set_title('Cumulative Publications Running Campaigns Over Time')
    ax.grid(True, alpha=0.3)

    # 2. New publications per month
    ax = axes_adoption[0, 1]
    new_pubs_values = cumulative_pubs.select("new_publications").to_numpy().flatten()
    ax.bar(range(len(months_cumul)), new_pubs_values, color='#A23B72', alpha=0.7)
    ax.set_xticks(range(0, len(months_cumul), max(1, len(months_cumul)//10)))
    ax.set_xticklabels([months_cumul[i] for i in range(0, len(months_cumul), max(1, len(months_cumul)//10))], 
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('New Publications')
    ax.set_title('New Publications Joining Each Month')

    # 3. Monthly active publications
    ax = axes_adoption[1, 0]
    months_active = monthly_active.select(
        pl.concat_str([
            pl.col("year").cast(pl.Utf8),
            pl.lit("-"),
            pl.col("month").cast(pl.Utf8).str.zfill(2)
        ])
    ).to_numpy().flatten()
    active_pubs_values = monthly_active.select("active_publications").to_numpy().flatten()
    ax.plot(range(len(months_active)), active_pubs_values, marker='o', linewidth=2, color='#F18F01')
    ax.set_xticks(range(0, len(months_active), max(1, len(months_active)//10)))
    ax.set_xticklabels([months_active[i] for i in range(0, len(months_active), max(1, len(months_active)//10))], 
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Active Publications')
    ax.set_title('Monthly Active Publications')
    ax.grid(True, alpha=0.3)

    # 4. Monthly active campaigns
    ax = axes_adoption[1, 1]
    active_camps_values = monthly_active.select("active_campaigns").to_numpy().flatten()
    ax.plot(range(len(months_active)), active_camps_values, marker='s', linewidth=2, color='#8338EC')
    ax.set_xticks(range(0, len(months_active), max(1, len(months_active)//10)))
    ax.set_xticklabels([months_active[i] for i in range(0, len(months_active), max(1, len(months_active)//10))], 
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Active Campaigns')
    ax.set_title('Monthly Active Campaigns')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Calculate growth metrics
    total_unique_pubs = first_campaign_by_pub.shape[0]
    print(f"\n=== Publication Adoption Statistics ===")
    print(f"Total unique publications that have run campaigns: {total_unique_pubs:,}")
    print(f"\nCumulative growth by month:")
    print(cumulative_pubs)
    print(f"\nMonthly activity summary:")
    print(monthly_active)

    # Calculate month-over-month growth
    if len(new_pubs_values) > 1:
        avg_new_per_month = np.mean(new_pubs_values)
        recent_avg = np.mean(new_pubs_values[-6:]) if len(new_pubs_values) >= 6 else np.mean(new_pubs_values)
        print(f"\nAverage new publications per month: {avg_new_per_month:.1f}")
        print(f"Recent 6-month average: {recent_avg:.1f}")
        growth_trend = "Growing" if recent_avg > avg_new_per_month else "Stable/Declining"
        print(f"Trend: {growth_trend}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Historical Average CTR per Month

    Let's analyze how the average CTR has evolved over time.
    """)
    return


@app.cell
def _(
    add_temporal_columns,
    compute_distribution_stats,
    np,
    pl,
    placements_with_ctr,
    plt,
    print_stats,
):
    def analyze_monthly_ctr_trend(df: pl.DataFrame) -> pl.DataFrame:
        """Analyze average CTR per month over time."""
        df_temporal = add_temporal_columns(df)

        return df_temporal.group_by(["year", "month"]).agg([
            pl.col("ctr").mean().alias("mean_ctr"),
            pl.col("ctr").median().alias("median_ctr"),
            pl.col("ctr").std().alias("std_ctr"),
            pl.count().alias("placement_count"),
            pl.col("approved_opens").sum().alias("total_opens"),
            pl.col("approved_clicks").sum().alias("total_clicks"),
        ]).with_columns([
            (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
        ]).sort(["year", "month"])

    monthly_ctr = analyze_monthly_ctr_trend(placements_with_ctr)

    fig_monthly_ctr, axes_monthly_ctr = plt.subplots(2, 2, figsize=(14, 10))

    months_ctr = monthly_ctr.select(
        pl.concat_str([
            pl.col("year").cast(pl.Utf8),
            pl.lit("-"),
            pl.col("month").cast(pl.Utf8).str.zfill(2)
        ])
    ).to_numpy().flatten()

    # 1. Weighted CTR trend
    ax = axes_monthly_ctr[0, 0]
    weighted_ctr_values = monthly_ctr.select("weighted_ctr").to_numpy().flatten() * 100
    ax.plot(range(len(months_ctr)), weighted_ctr_values, marker='o', linewidth=2, color='#2E86AB')
    ax.fill_between(range(len(months_ctr)), weighted_ctr_values, alpha=0.3, color='#2E86AB')
    ax.set_xticks(range(0, len(months_ctr), max(1, len(months_ctr)//10)))
    ax.set_xticklabels([months_ctr[i] for i in range(0, len(months_ctr), max(1, len(months_ctr)//10))], 
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('CTR (%)')
    ax.set_title('Weighted Average CTR per Month (Impressions-weighted)')
    ax.grid(True, alpha=0.3)
    # Add trend line
    z = np.polyfit(range(len(weighted_ctr_values)), weighted_ctr_values, 1)
    p = np.poly1d(z)
    ax.plot(range(len(months_ctr)), p(range(len(months_ctr))), 
           "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.4f})')
    ax.legend()

    # 2. Mean vs Median CTR
    ax = axes_monthly_ctr[0, 1]
    mean_ctr_values = monthly_ctr.select("mean_ctr").to_numpy().flatten() * 100
    median_ctr_values = monthly_ctr.select("median_ctr").to_numpy().flatten() * 100
    ax.plot(range(len(months_ctr)), mean_ctr_values, marker='o', linewidth=2, 
           color='#A23B72', label='Mean CTR')
    ax.plot(range(len(months_ctr)), median_ctr_values, marker='s', linewidth=2, 
           color='#F18F01', label='Median CTR')
    ax.set_xticks(range(0, len(months_ctr), max(1, len(months_ctr)//10)))
    ax.set_xticklabels([months_ctr[i] for i in range(0, len(months_ctr), max(1, len(months_ctr)//10))], 
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('CTR (%)')
    ax.set_title('Mean vs Median CTR per Month')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. CTR variability (std dev)
    ax = axes_monthly_ctr[1, 0]
    std_ctr_values = monthly_ctr.select("std_ctr").to_numpy().flatten() * 100
    ax.bar(range(len(months_ctr)), std_ctr_values, color='#8338EC', alpha=0.7)
    ax.set_xticks(range(0, len(months_ctr), max(1, len(months_ctr)//10)))
    ax.set_xticklabels([months_ctr[i] for i in range(0, len(months_ctr), max(1, len(months_ctr)//10))], 
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('CTR Std Dev (%)')
    ax.set_title('CTR Standard Deviation per Month')

    # 4. Placement volume
    ax = axes_monthly_ctr[1, 1]
    placement_counts_monthly = monthly_ctr.select("placement_count").to_numpy().flatten()
    ax.bar(range(len(months_ctr)), placement_counts_monthly, color='#3A86FF', alpha=0.7)
    ax.set_xticks(range(0, len(months_ctr), max(1, len(months_ctr)//10)))
    ax.set_xticklabels([months_ctr[i] for i in range(0, len(months_ctr), max(1, len(months_ctr)//10))], 
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Number of Placements')
    ax.set_title('Placement Volume per Month')

    plt.tight_layout()
    plt.show()

    # Statistics summary
    print("\n=== Monthly CTR Statistics ===")
    print(monthly_ctr)

    weighted_ctr_stats = compute_distribution_stats(weighted_ctr_values)
    print_stats(weighted_ctr_stats, "Weighted CTR (%) Distribution Across Months")

    # Trend analysis
    print(f"\nCTR Trend Analysis:")
    print(f"  First month weighted CTR: {weighted_ctr_values[0]:.4f}%")
    print(f"  Last month weighted CTR: {weighted_ctr_values[-1]:.4f}%")
    print(f"  Change: {weighted_ctr_values[-1] - weighted_ctr_values[0]:.4f}%")
    print(f"  Linear trend slope: {z[0]:.6f}% per month")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. CTR Distribution Analysis

    Let's understand the distribution of our target variable (CTR).
    """)
    return


@app.cell
def _(
    compute_distribution_stats,
    np,
    placements_with_ctr,
    plot_histogram,
    plt,
    print_stats,
):
    def analyze_ctr_distribution(df):
        """Analyze the overall CTR distribution."""
        ctr_values = df.select("ctr").to_numpy().flatten()
        ctr_filtered = ctr_values[(ctr_values >= 0) & (ctr_values <= 0.2)]
        ctr_positive = ctr_values[ctr_values > 0]
        return ctr_values, ctr_filtered, ctr_positive

    ctr_all, ctr_filtered, ctr_positive = analyze_ctr_distribution(placements_with_ctr)

    fig_ctr, axes_ctr = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of CTR (filtered)
    plot_histogram(
        ctr_filtered,
        axes_ctr[0, 0],
        bins=100,
        color='#2E86AB',
        xlabel='CTR',
        ylabel='Frequency',
        title='CTR Distribution (0-20%)',
        median_line=True,
        mean_line=True
    )

    # 2. Log-scale histogram
    plot_histogram(
        np.log10(ctr_positive + 1e-10),
        axes_ctr[0, 1],
        bins=100,
        color='#A23B72',
        xlabel='log10(CTR)',
        ylabel='Frequency',
        title='CTR Distribution (Log Scale)'
    )

    # 3. Box plot
    box_data = ctr_all[(ctr_all >= 0) & (ctr_all <= 0.1)]
    axes_ctr[1, 0].boxplot(box_data, vert=True)
    axes_ctr[1, 0].set_ylabel('CTR')
    axes_ctr[1, 0].set_title('CTR Box Plot (0-10%)')

    # 4. CDF
    sorted_ctr = np.sort(ctr_filtered)
    cdf = np.arange(1, len(sorted_ctr) + 1) / len(sorted_ctr)
    axes_ctr[1, 1].plot(sorted_ctr, cdf, color='#F18F01', linewidth=2)
    axes_ctr[1, 1].set_xlabel('CTR')
    axes_ctr[1, 1].set_ylabel('Cumulative Probability')
    axes_ctr[1, 1].set_title('CTR Cumulative Distribution Function')
    axes_ctr[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Statistics
    ctr_stats = compute_distribution_stats(ctr_all)
    ctr_stats["zero_count"] = int(np.sum(ctr_all == 0))
    ctr_stats["zero_pct"] = 100 * np.sum(ctr_all == 0) / len(ctr_all)
    print_stats(ctr_stats, "CTR Statistics")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Volume Analysis - Clicks and Opens Distribution

    Understanding the scale of impressions (opens) and clicks helps us weigh our CTR calculations.
    """)
    return


@app.cell
def _(np, placements_with_ctr, plot_histogram, plot_scatter, plt):
    def get_volume_data(df):
        """Extract opens and clicks data."""
        opens = df.select("approved_opens").to_numpy().flatten()
        clicks = df.select("approved_clicks").to_numpy().flatten()
        ctr = df.select("ctr").to_numpy().flatten()
        return opens, clicks, ctr

    opens_data, clicks_data, ctr_data = get_volume_data(placements_with_ctr)
    sample_idx = np.random.choice(len(opens_data), min(10000, len(opens_data)), replace=False)

    fig_vol, axes_vol = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Opens distribution
    plot_histogram(
        np.log10(opens_data + 1),
        axes_vol[0, 0],
        bins=100,
        color='#3A86FF',
        xlabel='log10(Opens + 1)',
        ylabel='Frequency',
        title='Distribution of Email Opens (Log Scale)',
        median_line=True
    )

    # 2. Clicks distribution
    plot_histogram(
        np.log10(clicks_data + 1),
        axes_vol[0, 1],
        bins=100,
        color='#FF006E',
        xlabel='log10(Clicks + 1)',
        ylabel='Frequency',
        title='Distribution of Clicks (Log Scale)',
        median_line=True
    )

    # 3. Opens vs Clicks scatter
    plot_scatter(
        np.log10(opens_data[sample_idx] + 1),
        np.log10(clicks_data[sample_idx] + 1),
        axes_vol[1, 0],
        color='#8338EC',
        xlabel='log10(Opens + 1)',
        ylabel='log10(Clicks + 1)',
        title='Opens vs Clicks (10K sample)'
    )

    # 4. CTR vs Opens volume
    plot_scatter(
        np.log10(opens_data[sample_idx] + 1),
        ctr_data[sample_idx],
        axes_vol[1, 1],
        color='#FB5607',
        xlabel='log10(Opens + 1)',
        ylabel='CTR',
        title='CTR vs Volume (10K sample)',
        ylim=(0, 0.1)
    )

    plt.tight_layout()
    plt.show()

    print("\n=== Volume Statistics ===")
    print(f"Total Opens: {opens_data.sum():,}")
    print(f"Total Clicks: {clicks_data.sum():,}")
    print(f"Overall CTR: {clicks_data.sum()/opens_data.sum():.6f}")
    print(f"\nOpens per placement: Median={np.median(opens_data):,.0f}, Mean={np.mean(opens_data):,.0f}")
    print(f"Clicks per placement: Median={np.median(clicks_data):,.0f}, Mean={np.mean(clicks_data):,.0f}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Campaign Analysis

    Let's analyze the campaigns dataset and understand how campaign attributes relate to CTR.
    """)
    return


@app.cell
def _(campaigns, pl):
    # Explore campaigns dataset
    print("Campaigns Schema:")
    print(campaigns.schema)
    print("\nSample Data:")

    # Analyze campaign attributes
    print("\n=== Campaign Targeting Analysis ===")
    gender_dist = campaigns.group_by("target_gender").agg(pl.count().alias("count"))
    print("\nTarget Gender Distribution:")
    print(gender_dist)

    item_dist = campaigns.group_by("promoted_item").agg(pl.count().alias("count")).sort("count", descending=True)
    print("\nPromoted Item Distribution:")
    print(item_dist)

    campaigns.head(10)
    return


@app.cell
def _(campaigns, pl, placements_with_ctr, plot_bar_chart, plt):
    def analyze_ctr_by_campaign_attributes(placements_df, campaigns_df):
        """Analyze CTR by campaign attributes."""
        merged = placements_df.join(campaigns_df, on="campaign_id", how="left")

        ctr_by_gender = merged.group_by("target_gender").agg([
            pl.col("ctr").mean().alias("mean_ctr"),
            pl.col("ctr").median().alias("median_ctr"),
            pl.col("ctr").std().alias("std_ctr"),
            pl.count().alias("count"),
            pl.col("approved_opens").sum().alias("total_opens"),
            pl.col("approved_clicks").sum().alias("total_clicks")
        ]).with_columns(
            (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
        ).sort("count", descending=True)

        ctr_by_item = merged.group_by("promoted_item").agg([
            pl.col("ctr").mean().alias("mean_ctr"),
            pl.col("ctr").median().alias("median_ctr"),
            pl.col("ctr").std().alias("std_ctr"),
            pl.count().alias("count"),
            pl.col("approved_opens").sum().alias("total_opens"),
            pl.col("approved_clicks").sum().alias("total_clicks")
        ]).with_columns(
            (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
        ).sort("count", descending=True)

        return merged, ctr_by_gender, ctr_by_item

    placements_campaigns, ctr_by_gender, ctr_by_item = analyze_ctr_by_campaign_attributes(
        placements_with_ctr, campaigns
    )

    print("CTR by Target Gender:")
    print(ctr_by_gender)
    print("\nCTR by Promoted Item:")
    print(ctr_by_item)

    fig_camp, axes_camp = plt.subplots(1, 2, figsize=(14, 5))

    # Gender plot
    gender_data = ctr_by_gender.filter(pl.col("target_gender").is_not_null())
    if len(gender_data) > 0:
        genders = gender_data.select("target_gender").to_numpy().flatten()
        gender_ctrs = gender_data.select("weighted_ctr").to_numpy().flatten() * 100
        gender_counts = gender_data.select("count").to_numpy().flatten()
        plot_bar_chart(genders, gender_ctrs, axes_camp[0],
                      ylabel='CTR (%)', title='CTR by Target Gender',
                      counts=gender_counts)

    # Item type plot
    item_data = ctr_by_item.filter(pl.col("promoted_item").is_not_null()).head(10)
    if len(item_data) > 0:
        items = item_data.select("promoted_item").to_numpy().flatten()
        item_ctrs = item_data.select("weighted_ctr").to_numpy().flatten() * 100
        item_counts = item_data.select("count").to_numpy().flatten()
        plot_bar_chart(items, item_ctrs, axes_camp[1],
                      ylabel='CTR (%)', title='CTR by Promoted Item Type',
                      counts=item_counts)

    plt.tight_layout()
    plt.show()
    return (placements_campaigns,)


@app.cell
def _(mo):
    mo.md("""
    ## 10. Publication Analysis

    Let's analyze how different publications perform in terms of CTR.
    """)
    return


@app.cell
def _(np, pl, placements_with_ctr, plot_histogram, plot_scatter, plt):
    def analyze_ctr_by_publication(df):
        """Analyze CTR metrics by publication."""
        return df.group_by("publication_id").agg([
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

    ctr_by_publication = analyze_ctr_by_publication(placements_with_ctr)

    print(f"Total unique publications: {len(ctr_by_publication):,}")
    print("\nTop 20 publications by placement count:")
    print(ctr_by_publication.head(20))

    pub_ctrs = ctr_by_publication.select("weighted_ctr").to_numpy().flatten()
    pub_counts = ctr_by_publication.select("placement_count").to_numpy().flatten()
    pub_std = ctr_by_publication.filter(pl.col("placement_count") >= 10).select("std_ctr").to_numpy().flatten()

    fig_pub, axes_pub = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distribution of publication CTRs
    plot_histogram(
        pub_ctrs[pub_ctrs < 0.1],
        axes_pub[0, 0],
        bins=50,
        color='#2E86AB',
        xlabel='Publication CTR',
        ylabel='Number of Publications',
        title='Distribution of Publication-level CTR',
        median_line=True
    )

    # 2. Publication volume distribution
    plot_histogram(
        np.log10(pub_counts + 1),
        axes_pub[0, 1],
        bins=50,
        color='#A23B72',
        xlabel='log10(Placement Count)',
        ylabel='Number of Publications',
        title='Distribution of Publication Activity'
    )

    # 3. CTR vs Volume for publications
    plot_scatter(
        np.log10(pub_counts + 1),
        pub_ctrs,
        axes_pub[1, 0],
        color='#F18F01',
        alpha=0.5,
        size=10,
        xlabel='log10(Placement Count)',
        ylabel='CTR',
        title='Publication CTR vs Activity Level',
        ylim=(0, 0.15)
    )

    # 4. Variance in CTR across publications (stability)
    plot_histogram(
        pub_std[~np.isnan(pub_std)],
        axes_pub[1, 1],
        bins=50,
        color='#8338EC',
        xlabel='CTR Standard Deviation',
        ylabel='Number of Publications',
        title='CTR Variance by Publication (min 10 placements)'
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 11. Publication Tags Analysis

    Let's analyze how content categories (tags) relate to CTR.
    """)
    return


@app.cell
def _(pl, placements_with_ctr, plot_bar_chart, plt, publication_tags):
    def analyze_ctr_by_tags(placements_df, tags_df):
        """Analyze CTR by publication tags."""
        merged = placements_df.join(tags_df, on="publication_id", how="left")

        return merged.group_by("tags").agg([
            pl.col("ctr").mean().alias("mean_ctr"),
            pl.col("ctr").median().alias("median_ctr"),
            pl.count().alias("count"),
            pl.col("approved_opens").sum().alias("total_opens"),
            pl.col("approved_clicks").sum().alias("total_clicks")
        ]).with_columns(
            (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
        ).filter(
            pl.col("count") >= 100
        ).sort("count", descending=True)

    tag_stats = analyze_ctr_by_tags(placements_with_ctr, publication_tags)

    print("CTR by Publication Tags (min 100 placements):")
    print(tag_stats.head(30))

    fig_tags, axes_tags = plt.subplots(1, 2, figsize=(16, 6))

    top_tags = tag_stats.head(20)
    tags_names = top_tags.select("tags").to_numpy().flatten()
    tags_ctrs = top_tags.select("weighted_ctr").to_numpy().flatten() * 100
    tags_counts = top_tags.select("count").to_numpy().flatten()

    # 1. CTR by tag (horizontal)
    plot_bar_chart(
        tags_names, tags_ctrs, axes_tags[0],
        xlabel='CTR (%)',
        title='CTR by Publication Tag Category (Top 20 by volume)',
        horizontal=True
    )

    # 2. Volume by tag (horizontal)
    plot_bar_chart(
        tags_names, tags_counts, axes_tags[1],
        xlabel='Number of Placements',
        title='Placement Volume by Tag Category',
        horizontal=True
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 12. Temporal Analysis

    Let's analyze how CTR varies by day of week and hour of day.
    """)
    return


@app.cell
def _(add_temporal_columns, pl, placements_with_ctr, plot_bar_chart, plt):
    def analyze_ctr_by_time_features(df):
        """Analyze CTR by temporal features (weekday, hour)."""
        df_temporal = add_temporal_columns(df)

        by_weekday = df_temporal.group_by("weekday").agg([
            pl.col("ctr").mean().alias("mean_ctr"),
            pl.count().alias("count"),
            pl.col("approved_opens").sum().alias("total_opens"),
            pl.col("approved_clicks").sum().alias("total_clicks")
        ]).with_columns(
            (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
        ).sort("weekday")

        by_hour = df_temporal.group_by("hour").agg([
            pl.col("ctr").mean().alias("mean_ctr"),
            pl.count().alias("count"),
            pl.col("approved_opens").sum().alias("total_opens"),
            pl.col("approved_clicks").sum().alias("total_clicks")
        ]).with_columns(
            (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
        ).sort("hour")

        return by_weekday, by_hour

    ctr_by_weekday, ctr_by_hour = analyze_ctr_by_time_features(placements_with_ctr)

    fig_time, axes_time = plt.subplots(1, 2, figsize=(14, 5))

    # CTR by day of week
    weekdays = ['Everyday', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekday_ctrs = ctr_by_weekday.select("weighted_ctr").to_numpy().flatten() * 100
    plot_bar_chart(weekdays[:len(weekday_ctrs)], weekday_ctrs, axes_time[0],
                  ylabel='CTR (%)', title='CTR by Day of Week')

    # CTR by hour
    hours = ctr_by_hour.select("hour").to_numpy().flatten()
    hourly_ctrs = ctr_by_hour.select("weighted_ctr").to_numpy().flatten() * 100
    axes_time[1].plot(hours, hourly_ctrs, marker='o', linewidth=2, color='#8338EC')
    axes_time[1].set_xlabel('Hour of Day')
    axes_time[1].set_ylabel('CTR (%)')
    axes_time[1].set_title('CTR by Hour of Day')
    axes_time[1].set_xticks(range(0, 24, 2))
    axes_time[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("CTR by Weekday:")
    print(ctr_by_weekday)
    print("\nCTR by Hour:")
    print(ctr_by_hour)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 13. Publication-Campaign Interaction Analysis

    Let's understand how the combination of publication and campaign affects CTR.
    """)
    return


@app.cell
def _(np, pl, placements_with_ctr, plot_histogram, plot_scatter, plt):
    def analyze_pub_campaign_interactions(df):
        """Analyze publication-campaign interaction effects on CTR."""
        combo_stats = df.group_by(["publication_id", "campaign_id"]).agg([
            pl.col("ctr").mean().alias("mean_ctr"),
            pl.count().alias("placement_count"),
            pl.col("approved_opens").sum().alias("total_opens"),
            pl.col("approved_clicks").sum().alias("total_clicks")
        ]).with_columns(
            (pl.col("total_clicks") / pl.col("total_opens")).alias("weighted_ctr")
        )

        # Variance across campaigns for each publication
        pub_variance = combo_stats.filter(
            pl.col("placement_count") >= 3
        ).group_by("publication_id").agg([
            pl.col("weighted_ctr").mean().alias("mean_ctr_across_campaigns"),
            pl.col("weighted_ctr").std().alias("std_ctr_across_campaigns"),
            pl.col("campaign_id").n_unique().alias("num_campaigns"),
        ]).filter(pl.col("num_campaigns") >= 3)

        # Variance across publications for each campaign
        camp_variance = combo_stats.filter(
            pl.col("placement_count") >= 3
        ).group_by("campaign_id").agg([
            pl.col("weighted_ctr").mean().alias("mean_ctr_across_pubs"),
            pl.col("weighted_ctr").std().alias("std_ctr_across_pubs"),
            pl.col("publication_id").n_unique().alias("num_publications"),
        ]).filter(pl.col("num_publications") >= 3)

        return combo_stats, pub_variance, camp_variance

    combo_stats, pub_variance, camp_variance = analyze_pub_campaign_interactions(placements_with_ctr)

    print(f"Unique publication-campaign combinations: {len(combo_stats):,}")

    fig_interact, axes_interact = plt.subplots(2, 2, figsize=(14, 10))

    # 1. CTR variance across campaigns (for each publication)
    pub_variance_values = pub_variance.select("std_ctr_across_campaigns").to_numpy().flatten()
    pub_variance_values = pub_variance_values[~np.isnan(pub_variance_values)]
    plot_histogram(
        pub_variance_values,
        axes_interact[0, 0],
        bins=50,
        color='#2E86AB',
        xlabel='Std Dev of CTR across Campaigns',
        ylabel='Number of Publications',
        title='How much does CTR vary for a publication across campaigns?',
        median_line=True
    )

    # 2. CTR variance across publications (for each campaign)
    camp_variance_values = camp_variance.select("std_ctr_across_pubs").to_numpy().flatten()
    camp_variance_values = camp_variance_values[~np.isnan(camp_variance_values)]
    plot_histogram(
        camp_variance_values,
        axes_interact[0, 1],
        bins=50,
        color='#A23B72',
        xlabel='Std Dev of CTR across Publications',
        ylabel='Number of Campaigns',
        title='How much does CTR vary for a campaign across publications?',
        median_line=True
    )

    # 3. Mean vs Std for publications
    pub_means = pub_variance.select("mean_ctr_across_campaigns").to_numpy().flatten()
    pub_stds = pub_variance.select("std_ctr_across_campaigns").to_numpy().flatten()
    plot_scatter(
        pub_means, pub_stds, axes_interact[1, 0],
        color='#F18F01',
        alpha=0.5,
        size=10,
        xlabel='Mean CTR',
        ylabel='Std Dev CTR',
        title='Publication: Mean vs Variance in CTR across Campaigns',
        xlim=(0, 0.1),
        ylim=(0, 0.05)
    )

    # 4. Mean vs Std for campaigns
    camp_means = camp_variance.select("mean_ctr_across_pubs").to_numpy().flatten()
    camp_stds = camp_variance.select("std_ctr_across_pubs").to_numpy().flatten()
    plot_scatter(
        camp_means, camp_stds, axes_interact[1, 1],
        color='#8338EC',
        alpha=0.5,
        size=10,
        xlabel='Mean CTR',
        ylabel='Std Dev CTR',
        title='Campaign: Mean vs Variance in CTR across Publications',
        xlim=(0, 0.1),
        ylim=(0, 0.05)
    )

    plt.tight_layout()
    plt.show()

    print(f"\n=== Variance Analysis ===")
    print(f"Publications with 3+ campaigns: {len(pub_variance):,}")
    print(f"  Median CTR std across campaigns: {np.median(pub_variance_values):.4f}")
    print(f"Campaigns with 3+ publications: {len(camp_variance):,}")
    print(f"  Median CTR std across publications: {np.median(camp_variance_values):.4f}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 14. Correlation Analysis

    Let's compute correlations between features and CTR to identify the most predictive features.
    """)
    return


@app.cell
def _(
    add_temporal_columns,
    campaigns,
    np,
    pl,
    placements_with_ctr,
    plt,
    publication_tags,
    sns,
):
    def prepare_correlation_data(placements_df, campaigns_df, tags_df):
        """Prepare feature matrix for correlation analysis."""
        # Prepare campaigns with encoded features
        gender_map = {"unknown": 0, "no_preference": 1, "balanced": 2, 
                     "predominantly_male": 3, "predominantly_female": 4}
        item_map = {"unknown": 0, "": 0, "product": 1, "service": 2, 
                   "newsletter": 3, "knowledge_product": 4, "event": 5, "other": 6}

        campaigns_features = campaigns_df.with_columns([
            pl.col("target_gender").fill_null("unknown"),
            pl.col("promoted_item").fill_null("unknown")
        ]).with_columns([
            pl.col("target_gender").replace(gender_map, default=0).alias("gender_encoded"),
            pl.col("promoted_item").replace(item_map, default=0).alias("item_encoded"),
            pl.col("target_incomes").str.count_matches(r"range_").alias("num_income_targets"),
            pl.col("target_ages").str.count_matches(r"range_").alias("num_age_targets")
        ])

        placements_temporal = add_temporal_columns(placements_df)

        full_data = placements_temporal.join(
            campaigns_features.select(["campaign_id", "gender_encoded", "item_encoded", 
                                       "num_income_targets", "num_age_targets"]),
            on="campaign_id",
            how="left"
        ).join(
            tags_df.select(["publication_id", "tag_ids"]),
            on="publication_id",
            how="left"
        ).with_columns([
            pl.col("tag_ids").str.count_matches(r"\d+").fill_null(0).alias("num_tags")
        ])

        return full_data.select([
            "ctr", "approved_opens", "approved_clicks", "year", "month",
            "weekday", "hour", "gender_encoded", "item_encoded",
            "num_income_targets", "num_age_targets", "num_tags"
        ]).drop_nulls()

    correlation_data = prepare_correlation_data(placements_with_ctr, campaigns, publication_tags)
    feature_names = correlation_data.columns
    data_matrix = correlation_data.to_numpy()
    corr_matrix = np.corrcoef(data_matrix.T)

    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, 
                mask=mask,
                xticklabels=feature_names, 
                yticklabels=feature_names, 
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                square=True,
                ax=ax_corr)
    ax_corr.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

    # Print correlations with CTR
    print("\n=== Correlations with CTR ===")
    ctr_idx = feature_names.index("ctr")
    correlations_with_ctr = [(feature_names[i], corr_matrix[ctr_idx, i]) 
                            for i in range(len(feature_names)) if i != ctr_idx]
    correlations_with_ctr.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in correlations_with_ctr:
        print(f"  {feat}: {corr:.4f}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 15. Variance Decomposition Analysis

    Let's understand how much variance in CTR is explained by publications vs campaigns.
    """)
    return


@app.cell
def _(np, pl, placements_with_ctr, plot_histogram, plt):
    def compute_variance_decomposition(df):
        """
        Decompose CTR variance by publication, campaign, and combination.
        Uses proper Sum of Squares approach with group size weighting.
        
        R-squared represents the proportion of variance explained by using
        group means as predictions instead of the overall mean.
        """
        ctr_values = df.select("ctr").to_numpy().flatten()
        overall_mean = np.mean(ctr_values)
        n_total = len(ctr_values)
        
        # Total Sum of Squares
        ss_total = np.sum((ctr_values - overall_mean) ** 2)
        overall_var = ss_total / (n_total - 1)
        
        # Publication effect: SS_between for publications
        pub_stats = df.group_by("publication_id").agg([
            pl.col("ctr").mean().alias("pub_mean_ctr"),
            pl.count().alias("pub_count")
        ])
        pub_means = pub_stats.select("pub_mean_ctr").to_numpy().flatten()
        pub_counts = pub_stats.select("pub_count").to_numpy().flatten()
        ss_between_pubs = np.sum(pub_counts * (pub_means - overall_mean) ** 2)
        r2_pubs = ss_between_pubs / ss_total
        
        # Campaign effect: SS_between for campaigns
        camp_stats = df.group_by("campaign_id").agg([
            pl.col("ctr").mean().alias("camp_mean_ctr"),
            pl.count().alias("camp_count")
        ])
        camp_means = camp_stats.select("camp_mean_ctr").to_numpy().flatten()
        camp_counts = camp_stats.select("camp_count").to_numpy().flatten()
        ss_between_camps = np.sum(camp_counts * (camp_means - overall_mean) ** 2)
        r2_camps = ss_between_camps / ss_total
        
        # Publication-Campaign combination effect
        combo_stats = df.group_by(["publication_id", "campaign_id"]).agg([
            pl.col("ctr").mean().alias("combo_mean_ctr"),
            pl.count().alias("combo_count")
        ])
        combo_means = combo_stats.select("combo_mean_ctr").to_numpy().flatten()
        combo_counts = combo_stats.select("combo_count").to_numpy().flatten()
        ss_between_combos = np.sum(combo_counts * (combo_means - overall_mean) ** 2)
        r2_combos = ss_between_combos / ss_total
        
        return {
            "overall_mean": overall_mean,
            "overall_var": overall_var,
            "ss_total": ss_total,
            "ss_between_pubs": ss_between_pubs,
            "ss_between_camps": ss_between_camps,
            "ss_between_combos": ss_between_combos,
            "r2_pubs": r2_pubs,
            "r2_camps": r2_camps,
            "r2_combos": r2_combos,
            "pub_stats": pub_stats,
            "camp_stats": camp_stats,
            "combo_stats": combo_stats,
            "n_publications": len(pub_means),
            "n_campaigns": len(camp_means),
            "n_combinations": len(combo_means),
        }

    variance_decomp = compute_variance_decomposition(placements_with_ctr)

    overall_mean = variance_decomp["overall_mean"]
    overall_var = variance_decomp["overall_var"]
    r2_pubs = variance_decomp["r2_pubs"]
    r2_camps = variance_decomp["r2_camps"]
    r2_combos = variance_decomp["r2_combos"]

    print(f"Overall CTR mean: {overall_mean:.6f}")
    print(f"Overall CTR variance: {overall_var:.8f}")
    print(f"Total observations: {placements_with_ctr.shape[0]:,}")
    print(f"\n=== Variance Decomposition (R-squared) ===")
    print(f"Variance explained by publication_id: {100*r2_pubs:.1f}%")
    print(f"  (Using {variance_decomp['n_publications']:,} unique publications)")
    print(f"Variance explained by campaign_id: {100*r2_camps:.1f}%")
    print(f"  (Using {variance_decomp['n_campaigns']:,} unique campaigns)")
    print(f"Variance explained by pub-campaign combo: {100*r2_combos:.1f}%")
    print(f"  (Using {variance_decomp['n_combinations']:,} unique combinations)")
    print(f"\nNote: These are independent R² values, not additive components.")

    fig_var, axes_var = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Publication means distribution
    pub_ctr_values = variance_decomp["pub_stats"].select("pub_mean_ctr").to_numpy().flatten()
    plot_histogram(
        pub_ctr_values[pub_ctr_values < 0.1],
        axes_var[0],
        bins=50,
        color='#2E86AB',
        xlabel='Mean CTR',
        ylabel='Number of Publications',
        title=f'Distribution of Publication Mean CTR\nR² = {100*r2_pubs:.1f}%'
    )
    axes_var[0].axvline(overall_mean, color='red', linestyle='--', linewidth=2, label=f'Overall: {overall_mean:.4f}')
    axes_var[0].legend()

    # 2. Campaign means distribution
    camp_ctr_values = variance_decomp["camp_stats"].select("camp_mean_ctr").to_numpy().flatten()
    plot_histogram(
        camp_ctr_values[camp_ctr_values < 0.1],
        axes_var[1],
        bins=50,
        color='#A23B72',
        xlabel='Mean CTR',
        ylabel='Number of Campaigns',
        title=f'Distribution of Campaign Mean CTR\nR² = {100*r2_camps:.1f}%'
    )
    axes_var[1].axvline(overall_mean, color='red', linestyle='--', linewidth=2, label=f'Overall: {overall_mean:.4f}')
    axes_var[1].legend()

    # 3. Combination means distribution
    combo_ctr_values = variance_decomp["combo_stats"].select("combo_mean_ctr").to_numpy().flatten()
    plot_histogram(
        combo_ctr_values[combo_ctr_values < 0.1],
        axes_var[2],
        bins=50,
        color='#F18F01',
        xlabel='Mean CTR',
        ylabel='Number of Combinations',
        title=f'Distribution of Pub-Campaign Mean CTR\nR² = {100*r2_combos:.1f}%'
    )
    axes_var[2].axvline(overall_mean, color='red', linestyle='--', linewidth=2, label=f'Overall: {overall_mean:.4f}')
    axes_var[2].legend()

    plt.tight_layout()
    plt.show()
    return overall_var, r2_camps, r2_combos, r2_pubs


@app.cell
def _(mo):
    mo.md("""
    ## 16. Statistical Tests for Feature Importance

    Let's perform statistical tests to validate the significance of feature relationships.
    """)
    return


@app.cell
def _(cohens_d, np, pl, placements_campaigns, stats):
    def perform_statistical_tests(df):
        """Perform ANOVA and effect size calculations for categorical features."""
        results = {}

        # ANOVA for target_gender
        gender_groups = df.filter(
            pl.col("target_gender").is_not_null()
        ).group_by("target_gender").agg(
            pl.col("ctr").alias("ctr_values")
        )
        gender_group_dict = {row["target_gender"]: np.array(row["ctr_values"]) 
                           for row in gender_groups.iter_rows(named=True)}

        if len(gender_group_dict) >= 2:
            f_stat, p_value = stats.f_oneway(*gender_group_dict.values())
            results["gender_anova"] = {"f_stat": f_stat, "p_value": p_value}

        # ANOVA for promoted_item
        item_groups = df.filter(
            pl.col("promoted_item").is_not_null() & (pl.col("promoted_item") != "")
        ).group_by("promoted_item").agg(
            pl.col("ctr").alias("ctr_values")
        )
        item_group_dict = {row["promoted_item"]: np.array(row["ctr_values"]) 
                         for row in item_groups.iter_rows(named=True)}

        if len(item_group_dict) >= 2:
            f_stat, p_value = stats.f_oneway(*item_group_dict.values())
            results["item_anova"] = {"f_stat": f_stat, "p_value": p_value}

        # Effect sizes
        if 'predominantly_male' in gender_group_dict and 'predominantly_female' in gender_group_dict:
            results["gender_effect_size"] = cohens_d(
                gender_group_dict['predominantly_male'],
                gender_group_dict['predominantly_female']
            )

        if 'product' in item_group_dict and 'service' in item_group_dict:
            results["item_effect_size"] = cohens_d(
                item_group_dict['product'],
                item_group_dict['service']
            )

        return results

    stat_test_results = perform_statistical_tests(placements_campaigns)

    if "gender_anova" in stat_test_results:
        print("ANOVA: Target Gender effect on CTR")
        print(f"  F-statistic: {stat_test_results['gender_anova']['f_stat']:.4f}")
        print(f"  p-value: {stat_test_results['gender_anova']['p_value']:.2e}")
        print(f"  Significant: {'Yes' if stat_test_results['gender_anova']['p_value'] < 0.05 else 'No'}")

    if "item_anova" in stat_test_results:
        print("\nANOVA: Promoted Item effect on CTR")
        print(f"  F-statistic: {stat_test_results['item_anova']['f_stat']:.4f}")
        print(f"  p-value: {stat_test_results['item_anova']['p_value']:.2e}")
        print(f"  Significant: {'Yes' if stat_test_results['item_anova']['p_value'] < 0.05 else 'No'}")

    print("\n=== Effect Sizes (Cohen's d) ===")
    if "gender_effect_size" in stat_test_results:
        print(f"Male vs Female targeting: d = {stat_test_results['gender_effect_size']:.4f}")
    if "item_effect_size" in stat_test_results:
        print(f"Product vs Service: d = {stat_test_results['item_effect_size']:.4f}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 17. Feature Engineering Recommendations

    Based on our analysis, here are the recommended features for CTR prediction.
    """)
    return


@app.cell
def _(
    mo,
    r2_camps,
    r2_combos,
    r2_pubs,
):
    recommendations = f"""
    ## Summary of Findings

    ### 1. Target Variable (CTR) Characteristics
    - CTR follows a highly skewed distribution with many zeros
    - Consider using log-transform or beta distribution modeling
    - Median CTR is typically much lower than mean due to outliers

    ### 2. Variance Explainability (R-squared)
    - **Publication effect**: {100*r2_pubs:.1f}% of total variance
    - **Campaign effect**: {100*r2_camps:.1f}% of total variance  
    - **Combined effect**: {100*r2_combos:.1f}% of total variance

    The pub-campaign combination explains most variance, suggesting strong interaction effects.
    Note: These are independent R² values measuring how much variance each grouping explains.

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

    mo.md(recommendations)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 18. Next Steps

    1. **Feature Engineering Pipeline**: Create features based on the analysis above
    2. **Embedding Analysis**: If using post_embeddings.csv, analyze how content embeddings relate to CTR
    3. **Model Training**: Implement baseline and candidate models
    4. **Cross-Validation**: Set up proper temporal validation strategy
    5. **Production Considerations**: Define inference pipeline for real-time predictions
    """)
    return


if __name__ == "__main__":
    app.run()
