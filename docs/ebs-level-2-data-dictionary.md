# EBS Level 2 source contract

This document separates vendor statements from conclusions verified against the
licensed files in this project. It is the semantic boundary for parsing and bar
construction; model code must not reinterpret these fields.

## Authoritative product semantics

CME describes EBS Spot history as an order book created on a time-slice basis.
Price records contain the market best prices at the end of a slice; deal records
contain the highest paid and lowest given prices during the slice. Level 2 uses
100-millisecond slices and contains up to five or ten price levels, including bid
and offer prices and sizes.

- Product catalog: <https://datamine.new.cmegroup.com/catalog?category=CA07>
- EBS Spot data page: <https://www.cmegroup.com/market-data/browse-data/catalog/ebs-spot-fx.html>
- EBS trading hours: <https://www.cmegroup.com/trading-hours.html>
- EBS value-date calendar: <https://www.cmegroup.com/content/dam/cmegroup/documents/ebs_value-date-calendar.pdf>
- EBS deal terminology: <https://www.cmegroup.com/tools-information/webhelp/ebs-workstation-quick-guide/Content/EBSDealsCard.html>

The standard spot trading date rolls at 17:00 New York, adjusted for daylight
saving time. The project files confirm that the stored timestamp is UTC: the
2024-01-02 file spans 22:00 UTC to 21:59 UTC, while the 2024-03-11 file spans
21:00 UTC to 20:59 UTC immediately after the New York DST change. The filename
date is therefore the EBS trading date, not necessarily the row's UTC date.

## Common columns

| Position | Field | Type/unit | Accepted state | Meaning and validation |
|---:|---|---|---|---|
| 0 | UTC date | `YYYY/MM/DD` | trading date or preceding calendar date | UTC calendar component of the observation. |
| 1 | UTC time | `HH:MM:SS.mmm` | valid millisecond time | UTC time component; records must be non-decreasing. |
| 2 | symbol | text | configured slash-delimited pair | EBS instrument; must agree with the filename. |
| 3 | record marker | enum | `Q` or `D` | Routes the row to the price- or deal-record parser. |
| 4 | side | integer enum | `0` or `1` | For Q: 0 bid, 1 offer. For D: 0 lowest given, 1 highest paid. |

## Price (`Q`) record columns

Price records contain nine columns. Observed rows arrive in blocks with one
timestamp and side. A normal block contains contiguous levels 1..10. At session
initialization, a one-row level-1 block with empty price and zero size/count
clears that side. A block replaces the complete side state; it is not interpreted
as an incremental update to one level.

| Position | Field | Type/unit | Accepted state | Meaning and validation |
|---:|---|---|---|---|
| 5 | level | integer | 1..10 | Rank away from top of book; bid prices decrease and offers increase. |
| 6 | price | positive finite decimal | price or empty reset | Quoted FX price at the end of the slice. |
| 7 | size | non-negative integer, base-currency units | positive with price; zero on reset | Aggregate displayed amount at the level. |
| 8 | count | non-negative integer | positive with price; zero on reset | Number of displayed orders contributing at the level. |

Bid and offer blocks can have different timestamps. Minute features consequently
use the latest causal state of each side and publish their maximum age. They do
not claim that the two source records were simultaneous. A side older than the
configured limit is missing, not forward-filled indefinitely.

## Deal (`D`) record columns

Deal records contain ten columns. “Paid” is aggressive buying and “given” is
aggressive selling. Both sides can occur in the same 100-millisecond slice.

| Position | Field | Type/unit | Accepted state | Meaning and validation |
|---:|---|---|---|---|
| 5 | level placeholder | empty text | empty only | Deals have no book-depth rank. |
| 6 | extremal deal price | positive finite decimal | required | Highest paid for side 1 or lowest given for side 0 in the slice. |
| 7 | extremal-price volume | positive integer, base-currency units | required | Volume represented at the reported extremal price. |
| 8 | deal count | positive integer | required | Number of deals represented by the side record. |
| 9 | total side volume | positive integer, base-currency units | at least column 7 | Total paid/given volume represented by the row across prices in the slice. |

Column 9 is used for directional paid/given volume. Because prices for the other
volume in the slice are not present, a true VWAP cannot be reconstructed. The
only persisted price aggregate is explicitly named
`extremal_price_weighted_mean` and weights column 6 by column 7; it must never be
reported as VWAP.

## Missingness and invariants

- Empty physical lines are counted, not treated as records.
- Every physical line reconciles to Q, D, empty, or an explicit parse error.
- Non-finite/negative values, malformed CSV, symbol mismatches, invalid levels,
  and out-of-order timestamps are errors.
- A valid reconstructed book has best bid below best offer, strictly decreasing
  bid depth, strictly increasing offer depth, and non-negative size/count.
- Reset, stale, crossed, or incomplete states yield an explicitly unobserved book;
  they are never silently retained or repaired with future information.
- Zero deal volume is emitted only when the minute has a valid observed book/feed.
  Otherwise deal fields remain missing.

## Current-corpus verification (2024 Q1)

On 2026-08-30, a full structural pass over all 234 configured files reported:

- 357,754,110 Q rows and 463,023 D rows;
- 179,407,369 bid and 178,346,741 offer rows;
- 225,714 given and 237,309 paid deal rows;
- 473 explicit quote-side reset rows;
- 21,727 rows where total volume differs from extremal-price volume;
- maximum observed depth 10;
- zero malformed structures, symbol/filename mismatches, timestamp reversals,
  duplicate/reversed levels, or level gaps.

Representative strict parser/reconstruction runs for 2024-01-02 produced exactly
1,440 UTC minute rows for each configured instrument with no parse errors or
negative quote ages. Invalid asynchronous crossed states are quarantined rather
than repaired (one bar in EUR/USD, none in EUR/JPY, one in USD/JPY for that date).
