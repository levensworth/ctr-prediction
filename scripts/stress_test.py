#!/usr/bin/env python3
"""
Stress test script for the CTR Prediction API.

Measures performance metrics including:
- Mean execution time per request
- Average per-row inference time
- Latency distribution (p50, p75, p90, p95, p99)
- Concurrent request handling capabilities

Usage:
    python scripts/stress_test.py --url http://localhost:8000 --requests 100 --concurrency 10 --batch-size 5
"""

import argparse
import asyncio
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import httpx
except ImportError:
    print("Error: httpx is required. Install with: pip install httpx")
    sys.exit(1)


@dataclass(frozen=True)
class RequestResult:
    """Result of a single API request."""
    
    success: bool
    latency_ms: float
    rows_in_request: int
    server_execution_time_ms: Optional[float]
    error_message: Optional[str] = None


@dataclass
class StressTestConfig:
    """Configuration for the stress test."""
    
    base_url: str
    total_requests: int
    concurrency: int
    batch_size: int
    timeout_seconds: float
    use_feature_store_ids: bool
    feature_store_path: Optional[Path]


def load_sample_ids_from_feature_store(feature_store_path: Path) -> tuple[list[str], list[str]]:
    """Load sample publication and campaign IDs from the feature store."""
    try:
        import polars as pl
        
        test_features_path = feature_store_path / "test_features" / "features.parquet"
        if not test_features_path.exists():
            return [], []
        
        df = pl.read_parquet(test_features_path)
        
        publication_ids: list[str] = []
        campaign_ids: list[str] = []
        
        if "publication_id" in df.columns:
            publication_ids = df["publication_id"].unique().cast(pl.Utf8).to_list()
        
        if "campaign_id" in df.columns:
            campaign_ids = df["campaign_id"].unique().cast(pl.Utf8).to_list()
        
        return publication_ids, campaign_ids
    except Exception as e:
        print(f"Warning: Could not load feature store data: {e}")
        return [], []


def generate_request_payload(
    batch_size: int,
    publication_ids: list[str],
    campaign_ids: list[str],
) -> dict:
    """Generate a request payload with publication-campaign pairs."""
    import random
    
    data = []
    for _ in range(batch_size):
        if publication_ids and campaign_ids:
            pub_id = random.choice(publication_ids)
            camp_id = random.choice(campaign_ids)
        else:
            pub_id = str(uuid.uuid4())
            camp_id = str(uuid.uuid4())
        
        data.append({
            "publication_id": pub_id,
            "campaign_id": camp_id,
        })
    
    return {"data": data}


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    timeout: float,
) -> RequestResult:
    """Send a single request to the API and measure latency."""
    start_time = time.perf_counter()
    rows_in_request = len(payload["data"])
    
    try:
        response = await client.post(
            url,
            json=payload,
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        if response.status_code == 200:
            response_data = response.json()
            server_execution_time = response_data.get("execution_time_ms")
            return RequestResult(
                success=True,
                latency_ms=latency_ms,
                rows_in_request=rows_in_request,
                server_execution_time_ms=server_execution_time,
            )
        else:
            return RequestResult(
                success=False,
                latency_ms=latency_ms,
                rows_in_request=rows_in_request,
                server_execution_time_ms=None,
                error_message=f"HTTP {response.status_code}: {response.text[:200]}",
            )
    except httpx.TimeoutException:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return RequestResult(
            success=False,
            latency_ms=latency_ms,
            rows_in_request=rows_in_request,
            server_execution_time_ms=None,
            error_message="Request timeout",
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return RequestResult(
            success=False,
            latency_ms=latency_ms,
            rows_in_request=rows_in_request,
            server_execution_time_ms=None,
            error_message=str(e),
        )


async def run_stress_test(config: StressTestConfig) -> list[RequestResult]:
    """Execute the stress test with the given configuration."""
    
    publication_ids: list[str] = []
    campaign_ids: list[str] = []
    
    if config.use_feature_store_ids and config.feature_store_path:
        print("Loading sample IDs from feature store...")
        publication_ids, campaign_ids = load_sample_ids_from_feature_store(
            config.feature_store_path
        )
        if publication_ids and campaign_ids:
            print(f"  Loaded {len(publication_ids)} publication IDs and {len(campaign_ids)} campaign IDs")
        else:
            print("  No IDs found, will use random UUIDs")
    else:
        print("Using random UUIDs for testing")
    
    payloads = [
        generate_request_payload(config.batch_size, publication_ids, campaign_ids)
        for _ in range(config.total_requests)
    ]
    
    invoke_url = f"{config.base_url.rstrip('/')}/invoke"
    
    print(f"\nStarting stress test:")
    print(f"  URL: {invoke_url}")
    print(f"  Total requests: {config.total_requests}")
    print(f"  Concurrency: {config.concurrency}")
    print(f"  Batch size: {config.batch_size} rows/request")
    print(f"  Timeout: {config.timeout_seconds}s")
    print()
    
    results: list[RequestResult] = []
    semaphore = asyncio.Semaphore(config.concurrency)
    
    async def bounded_request(payload: dict) -> RequestResult:
        async with semaphore:
            return await send_request(
                client, invoke_url, payload, config.timeout_seconds
            )
    
    async with httpx.AsyncClient() as client:
        # Warm-up request
        print("Sending warm-up request...")
        warmup_result = await send_request(
            client, invoke_url, payloads[0], config.timeout_seconds
        )
        if not warmup_result.success:
            print(f"Warning: Warm-up request failed: {warmup_result.error_message}")
        else:
            print(f"Warm-up complete (latency: {warmup_result.latency_ms:.2f}ms)")
        
        print("\nRunning stress test...")
        start_time = time.perf_counter()
        
        tasks = [bounded_request(payload) for payload in payloads]
        
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            
            if completed % max(1, config.total_requests // 10) == 0:
                progress = (completed / config.total_requests) * 100
                print(f"  Progress: {completed}/{config.total_requests} ({progress:.0f}%)")
        
        total_time = time.perf_counter() - start_time
    
    print(f"\nTest completed in {total_time:.2f}s")
    print(f"Throughput: {config.total_requests / total_time:.2f} requests/second")
    
    return results


def calculate_percentile(values: list[float], percentile: float) -> float:
    """Calculate the given percentile of a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    index = min(index, len(sorted_values) - 1)
    return sorted_values[index]


def print_distribution_histogram(values: list[float], title: str, bins: int = 10) -> None:
    """Print an ASCII histogram of the distribution."""
    if not values:
        print(f"\n{title}: No data")
        return
    
    min_val = min(values)
    max_val = max(values)
    
    if min_val == max_val:
        print(f"\n{title}: All values are {min_val:.2f}ms")
        return
    
    bin_width = (max_val - min_val) / bins
    histogram: list[int] = [0] * bins
    
    for val in values:
        bin_index = min(int((val - min_val) / bin_width), bins - 1)
        histogram[bin_index] += 1
    
    max_count = max(histogram)
    bar_width = 40
    
    print(f"\n{title}")
    print("-" * 70)
    
    for i, count in enumerate(histogram):
        lower = min_val + i * bin_width
        upper = min_val + (i + 1) * bin_width
        bar_length = int((count / max_count) * bar_width) if max_count > 0 else 0
        bar = "█" * bar_length
        print(f"  {lower:8.2f} - {upper:8.2f}ms | {bar:<{bar_width}} | {count:4d}")


def analyze_results(results: list[RequestResult], batch_size: int) -> None:
    """Analyze and print the stress test results."""
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print("\n" + "=" * 70)
    print("STRESS TEST RESULTS")
    print("=" * 70)
    
    # Summary
    print(f"\n{'SUMMARY':^70}")
    print("-" * 70)
    print(f"  Total Requests:     {len(results)}")
    print(f"  Successful:         {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"  Failed:             {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    print(f"  Batch Size:         {batch_size} rows/request")
    print(f"  Total Rows:         {sum(r.rows_in_request for r in results)}")
    
    if not successful:
        print("\nNo successful requests to analyze.")
        if failed:
            print("\nSample errors:")
            for error in set(r.error_message for r in failed[:5] if r.error_message):
                print(f"  - {error}")
        return
    
    # Latency statistics (client-side)
    latencies = [r.latency_ms for r in successful]
    
    print(f"\n{'CLIENT-SIDE LATENCY (end-to-end)':^70}")
    print("-" * 70)
    print(f"  Mean:               {statistics.mean(latencies):.2f}ms")
    print(f"  Std Dev:            {statistics.stdev(latencies) if len(latencies) > 1 else 0:.2f}ms")
    print(f"  Min:                {min(latencies):.2f}ms")
    print(f"  Max:                {max(latencies):.2f}ms")
    print(f"  Median (p50):       {calculate_percentile(latencies, 50):.2f}ms")
    print(f"  p75:                {calculate_percentile(latencies, 75):.2f}ms")
    print(f"  p90:                {calculate_percentile(latencies, 90):.2f}ms")
    print(f"  p95:                {calculate_percentile(latencies, 95):.2f}ms")
    print(f"  p99:                {calculate_percentile(latencies, 99):.2f}ms")
    
    # Server-side execution time
    server_times = [
        r.server_execution_time_ms for r in successful 
        if r.server_execution_time_ms is not None
    ]
    
    if server_times:
        print(f"\n{'SERVER-SIDE EXECUTION TIME (from API response)':^70}")
        print("-" * 70)
        print(f"  Mean:               {statistics.mean(server_times):.2f}ms")
        print(f"  Std Dev:            {statistics.stdev(server_times) if len(server_times) > 1 else 0:.2f}ms")
        print(f"  Min:                {min(server_times):.2f}ms")
        print(f"  Max:                {max(server_times):.2f}ms")
        print(f"  Median (p50):       {calculate_percentile(server_times, 50):.2f}ms")
        print(f"  p75:                {calculate_percentile(server_times, 75):.2f}ms")
        print(f"  p90:                {calculate_percentile(server_times, 90):.2f}ms")
        print(f"  p95:                {calculate_percentile(server_times, 95):.2f}ms")
        print(f"  p99:                {calculate_percentile(server_times, 99):.2f}ms")
        
        # Per-row inference time
        per_row_times = [t / batch_size for t in server_times]
        
        print(f"\n{'PER-ROW INFERENCE TIME (server execution / batch size)':^70}")
        print("-" * 70)
        print(f"  Mean:               {statistics.mean(per_row_times):.4f}ms/row")
        print(f"  Std Dev:            {statistics.stdev(per_row_times) if len(per_row_times) > 1 else 0:.4f}ms/row")
        print(f"  Min:                {min(per_row_times):.4f}ms/row")
        print(f"  Max:                {max(per_row_times):.4f}ms/row")
        print(f"  p95:                {calculate_percentile(per_row_times, 95):.4f}ms/row")
    
    # Network overhead
    if server_times and len(server_times) == len(latencies):
        network_overhead = [
            latencies[i] - server_times[i] 
            for i in range(len(server_times))
        ]
        print(f"\n{'NETWORK OVERHEAD (client latency - server time)':^70}")
        print("-" * 70)
        print(f"  Mean:               {statistics.mean(network_overhead):.2f}ms")
        print(f"  p95:                {calculate_percentile(network_overhead, 95):.2f}ms")
    
    # Distribution histograms
    print_distribution_histogram(latencies, "CLIENT LATENCY DISTRIBUTION")
    
    if server_times:
        print_distribution_histogram(server_times, "SERVER EXECUTION TIME DISTRIBUTION")
    
    # Errors summary
    if failed:
        print(f"\n{'ERRORS':^70}")
        print("-" * 70)
        error_counts: dict[str, int] = {}
        for r in failed:
            error = r.error_message or "Unknown error"
            error_counts[error] = error_counts.get(error, 0) + 1
        
        for error, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {count:4d}x  {error[:60]}")
    
    print("\n" + "=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stress test the CTR Prediction API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test with 100 requests, 10 concurrent
  python scripts/stress_test.py --url http://localhost:8000 -n 100 -c 10

  # High load test with larger batches
  python scripts/stress_test.py --url http://localhost:8000 -n 500 -c 50 -b 20

  # Use real IDs from feature store
  python scripts/stress_test.py --url http://localhost:8000 -n 100 -c 10 --use-feature-store

  # Quick test with custom timeout
  python scripts/stress_test.py --url http://localhost:8000 -n 50 -c 5 --timeout 30
        """,
    )
    
    parser.add_argument(
        "--url", "-u",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)",
    )
    
    parser.add_argument(
        "--requests", "-n",
        type=int,
        default=100,
        help="Total number of requests to send (default: 100)",
    )
    
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=10,
        help="Maximum number of concurrent requests (default: 10)",
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=5,
        help="Number of rows per request (default: 5)",
    )
    
    parser.add_argument(
        "--timeout", "-t",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60)",
    )
    
    parser.add_argument(
        "--use-feature-store",
        action="store_true",
        help="Use real IDs from the feature store instead of random UUIDs",
    )
    
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/features",
        help="Path to the feature store artifacts (default: artifacts/features)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for the stress test script."""
    args = parse_args()
    
    config = StressTestConfig(
        base_url=args.url,
        total_requests=args.requests,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout,
        use_feature_store_ids=args.use_feature_store,
        feature_store_path=Path(args.artifacts_dir) if args.use_feature_store else None,
    )
    
    print("CTR Prediction API - Stress Test")
    print("=" * 70)
    
    results = asyncio.run(run_stress_test(config))
    analyze_results(results, config.batch_size)


if __name__ == "__main__":
    main()
