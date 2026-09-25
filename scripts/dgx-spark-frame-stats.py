"""Summarize the server's sampled frame timing logs from stdin."""

import math
import re
import statistics
import sys


def percentile(values, fraction):
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def main():
    pattern = re.compile(r"frame=(\d+) total=([0-9.]+)ms")
    samples = [(int(match.group(1)), float(match.group(2)))
               for line in sys.stdin if (match := pattern.search(line))]
    if not samples:
        raise SystemExit("No sampled frame timings found")
    values = sorted(value for _, value in samples)
    print(f"Sampled frames: {len(values)} (server logs one in every 50 frames)")
    print(f"Frame numbers: {samples[0][0]} through {samples[-1][0]}")
    print(f"p50: {percentile(values, 0.50):.1f} ms")
    print(f"p95: {percentile(values, 0.95):.1f} ms")
    print(f"p99: {percentile(values, 0.99):.1f} ms")
    print(f"max: {max(values):.1f} ms")
    print(f"Sampled frames over 80 ms: {sum(value > 80 for value in values)}")
    print("Unlogged frames were not measured by this report.")


if __name__ == "__main__":
    main()
