"""Command line entry point: ``python3 -m uradata <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .analysis import (absorption, assess_offer, largest_gap, monthly_medians,
                       privatisation_event_study, project_benchmark, since_month,
                       size_cohort)
from .client import URAClient, URAError
from .hedonic import HedonicModel
from .model import flatten, month_index, year_month
from .stats import median, percentile

DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ura_cache.json")


def load(cache_path: str, refresh: bool = False):
    """Return the raw project payload, downloading it if needed."""
    if refresh or not os.path.exists(cache_path):
        client = URAClient()
        def progress(batch, count):
            print(f"  batch {batch}: {count} projects", file=sys.stderr)
        print("Fetching PMI_Resi_Transaction ...", file=sys.stderr)
        projects = client.transactions(progress=progress)
        with open(cache_path, "w") as handle:
            json.dump(projects, handle)
        print(f"Cached {len(projects)} projects -> {cache_path}", file=sys.stderr)
        return projects
    with open(cache_path) as handle:
        return json.load(handle)


def _fmt(value, width=11):
    return f"{value:>{width},.0f}" if value is not None else " " * width


def cmd_fetch(args):
    projects = load(args.cache, refresh=True)
    rows = flatten(projects)
    print(f"{len(rows):,} transactions across {len(projects):,} projects")
    months = sorted({r.year_month for r in rows if r.month})
    print(f"Period covered: {months[0]} .. {months[-1]}")
    return 0


def cmd_benchmark(args):
    rows = flatten(load(args.cache))
    project_rows = project_benchmark(rows, args.project, resale_only=not args.include_new)
    if not project_rows:
        print(f"No transactions found for {args.project!r}")
        return 1
    if args.since:
        project_rows = since_month(project_rows, args.since)
        if not project_rows:
            print("No transactions in the requested window.")
            return 1
    first = project_rows[0]
    print(f"{first.project} — {first.street}, D{first.district} ({first.segment})")
    print(f"{first.property_type}, {first.tenure}")
    print(f"{len(project_rows)} transactions\n")
    print(f"{'month':>8} {'type':>8} {'sqft':>7} {'price':>11} {'psf':>7}  floor")
    for t in project_rows:
        print(f"{t.year_month:>8} {t.sale_type_name:>8} {t.sqft:>7,.0f} "
              f"{t.price:>11,.0f} {t.psf:>7,.0f}  {t.floor_range}")
    print("\npsf by month:")
    for m, value in sorted(monthly_medians(project_rows).items()):
        count = sum(1 for t in project_rows if t.month == m)
        print(f"  {m // 12:04d}-{m % 12 + 1:02d}  n={count:>3}  median psf {value:>7,.0f}")
    return 0


GAP_WARNING_MONTHS = 3


def _window(project_rows, since):
    """Apply --since, and warn when a dormant period is still in the sample."""
    gap = largest_gap(project_rows)
    if since:
        project_rows = since_month(project_rows, since)
        print(f"Window: {since} onwards ({len(project_rows)} transactions)")
    elif gap and gap[0] > GAP_WARNING_MONTHS:
        resume = f"{gap[1] // 12:04d}-{gap[1] % 12 + 1:02d}"
        print(f"WARNING: {gap[0]} quiet months before {resume} are still in the sample.")
        print(f"         A time trend fitted across that gap will overstate drift.")
        print(f"         Re-run with --since {resume} for the post-gap market.\n")
    return project_rows


def cmd_offer(args):
    rows = flatten(load(args.cache))
    project_rows = project_benchmark(rows, args.project)
    if not project_rows:
        print(f"No resale transactions found for {args.project!r}")
        return 1
    project_rows = _window(project_rows, args.since)
    if not project_rows:
        print("No transactions in the requested window.")
        return 1

    fair_value = None
    if args.floor is not None:
        try:
            model = HedonicModel(project_rows)
            month = args.month or max(t.month for t in project_rows)
            valuation = model.value(args.sqft, args.floor, month)
            fair_value = valuation.price
            print(f"Hedonic model  n={valuation.n}  R2={valuation.r2:.3f}  "
                  f"residual sd={valuation.residual_sd * 100:.1f}%")
            print(f"  size elasticity {model.size_elasticity:.3f} | "
                  f"floor {model.floor_premium * 100:+.2f}%/storey | "
                  f"drift {model.monthly_drift * 100:+.2f}%/month")
            print(f"  fair value for {args.sqft:,.0f} sqft on floor {args.floor:.0f}: "
                  f"{valuation.price:,.0f} ({valuation.psf:,.0f} psf)")
            print(f"  one-sd range: {valuation.low:,.0f} .. {valuation.high:,.0f}")
            print(f"  offer sits {valuation.assess(args.price)}\n")
        except ValueError as exc:
            print(f"(hedonic model skipped: {exc})\n")

    result = assess_offer(project_rows, args.price, args.sqft, fair_value=fair_value)
    print(f"Offer {result.offer:,.0f} for {result.sqft:,.0f} sqft = {result.offer_psf:,.0f} psf")
    print(f"  comparable cohort (+/-3% size): n={result.cohort_n}"
          + (f", median {result.cohort_median_price:,.0f} "
             f"({result.cohort_median_psf:,.0f} psf)" if result.cohort_n else ""))
    print(f"  project median psf: {result.project_median_psf:,.0f}")
    print(f"  offer psf percentile: {result.psf_percentile:.0f}th")
    print(f"  offer price percentile: {result.price_percentile:.0f}th")
    print(f"  VERDICT: {result.verdict}")

    cohort = size_cohort(project_rows, args.sqft)
    if cohort:
        print(f"\n  comparable sales:")
        for t in sorted(cohort, key=lambda t: t.month or 0):
            delta = (args.price / t.price - 1) * 100
            print(f"    {t.year_month}  {t.sqft:>6,.0f}sqft {t.price:>11,.0f} "
                  f"{t.psf:>6,.0f}psf  fl{t.floor_range}   offer {delta:+.1f}%")
    return 0


def cmd_eventstudy(args):
    rows = flatten(load(args.cache))
    print("EC privatisation event study — sensitivity to the TOP estimate\n")
    print(f"{'TOP proxy':>12} {'min post':>9} {'events':>7} {'median':>9} {'mean':>8} {'positive':>10}")
    for lag in args.lags:
        for min_post in args.min_post:
            result = privatisation_event_study(rows, lease_to_top=lag, min_post=min_post)
            if result["n_events"] < 4:
                continue
            print(f"{'lease+' + str(lag):>12} {min_post:>9} {result['n_events']:>7} "
                  f"{result['median_lift'] * 100:>+8.1f}% {result['mean_lift'] * 100:>+7.1f}% "
                  f"{result['positive']:>6}/{result['n_events']}")
    detail = privatisation_event_study(rows, lease_to_top=args.lags[0], min_post=args.min_post[0])
    print(f"\nPer-project detail (TOP = lease+{args.lags[0]}, min post {args.min_post[0]}):")
    for event in detail["events"]:
        print(f"  {event['project'][:34]:34s} priv~{event['privatisation_year']}  "
              f"n={event['n_pre']:>3}/{event['n_post']:<3} "
              f"{event['relative_psf_before']:.3f} -> {event['relative_psf_after']:.3f}  "
              f"({event['lift'] * 100:+.1f}%)")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="uradata", description=__doc__)
    parser.add_argument("--cache", default=os.path.abspath(DEFAULT_CACHE))
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download and cache all transaction batches")
    fetch.set_defaults(func=cmd_fetch)

    bench = sub.add_parser("benchmark", help="list a project's transactions and psf trend")
    bench.add_argument("project")
    bench.add_argument("--include-new", action="store_true", help="include developer sales")
    bench.add_argument("--since", help="only transactions from this YYYY-MM onwards")
    bench.set_defaults(func=cmd_benchmark)

    offer = sub.add_parser("offer", help="score an offer against a project's history")
    offer.add_argument("project")
    offer.add_argument("--price", type=float, required=True)
    offer.add_argument("--sqft", type=float, required=True)
    offer.add_argument("--floor", type=float, help="storey; enables the hedonic model")
    offer.add_argument("--month", type=int, help="month index to value at (default: latest)")
    offer.add_argument("--since", help="only fit on transactions from this YYYY-MM onwards")
    offer.set_defaults(func=cmd_offer)

    study = sub.add_parser("eventstudy", help="test the EC privatisation premium")
    study.add_argument("--lags", type=int, nargs="+", default=[3, 4, 5])
    study.add_argument("--min-post", type=int, nargs="+", default=[8, 20, 30])
    study.set_defaults(func=cmd_eventstudy)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except URAError as exc:
        print(f"URA error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
