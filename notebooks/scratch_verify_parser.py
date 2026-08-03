# notebooks/scratch_verify_parser.py

from collections import Counter
from pathlib import Path

from ebs_tft.data.parsers import ebs_csv
from ebs_tft.data.repositories import raw_file

path = Path("data/raw/2024/20240102-EBS_LVL2_EUR_USD_0.csv.gz")

# Test quote parsing
print("=== QUOTES ===")
quotes = list(ebs_csv.parse_quotes(path=path))

print(f"Total quote rows parsed: {len(quotes)}")
print(f"First quote row:  {quotes[0]}")
print(f"Second quote row: {quotes[1]}")

# Check that None prices are handled
none_prices = [row for row in quotes if row.price is None]
print(f"Quote rows with no price (init rows): {len(none_prices)}")

# Check level distribution — should show levels 1–10 all roughly equal
level_counts = Counter(row.level for row in quotes)
print(f"Level distribution: {dict(sorted(level_counts.items()))}")

# Check side distribution — should be roughly 50/50 bid vs ask
side_counts = Counter(row.side for row in quotes)
print(f"Side distribution (0=bid, 1=ask): {dict(sorted(side_counts.items()))}")

# Test deal parsing
print("\n=== DEALS ===")
deals = list(ebs_csv.parse_deals(path=path))

print(f"Total deal rows parsed: {len(deals)}")
print(f"First deal row: {deals[0]}")

# Check side distribution
# 1=buy-initiated (highest paid), 0=sell-initiated (lowest given)
deal_side_counts = Counter(deal.side for deal in deals)
print(
    f"Deal side distribution (0=sell, 1=buy): {dict(sorted(deal_side_counts.items()))}"
)

# Check how many deal rows have no price (quiet periods)
no_deal_price = [d for d in deals if d.deal_price is None]
print(f"Deal rows with no price: {len(no_deal_price)}")


# Test file finding
files = list(
    raw_file.find_raw_files(
        data_dir=Path("data/raw"),
        instruments=raw_file.KNOWN_INSTRUMENTS,
        years=[2024],
    )
)
print("\n=== FILE SCANNER ===")
print(f"Total files found: {len(files)}")
for f in files[:5]:
    print(f"  {f.instrument}  {f.trading_date}  {f.path.name}")
