# Munder Difflin – Reflection Report

---

## 1. Agent Workflow

The system uses **smolagents** with GPT-4o and five agents in an orchestrator–worker hierarchy.

**Orchestrator** receives every request, parses the date, and routes to the right specialist. It holds no tools — routing only.

**Inventory agent** checks stock levels, retrieves full inventory snapshots, and places supplier reorders. Reorders are recorded at the order date so the cash balance updates immediately.

**Quoting agent** looks up past quotes, calculates prices with bulk discounts (5% / 10% / 15% at 100 / 500 / 1000 units), and verifies availability before quoting.

**Ordering agent** records sales transactions, estimates delivery dates, checks cash balance, generates financial reports, and triggers restocking after a sale.

**Communications agent** receives the orchestrator's internal result and rewrites it into clean customer-facing language before it is returned. It has no tools — its only job is to strip internal error language (transaction IDs, "Unknown item" errors, system references) and ensure successful orders always include itemised pricing, discount justification, and delivery dates.

Quoting and ordering are intentionally separate so that pricing logic never touches the transaction ledger. The communications agent is kept separate from the orchestrator so the rewrite rules can be updated independently without changing routing logic.

---

## 2. Evaluation Results

All 20 requests from `quote_requests_sample.csv` were tested. Full output is in `test_results.csv`.

| | |
|---|---|
| Requests tested | 20 |
| Fulfilled (full or partial) | 3 (requests 3, 7, 14) |
| Failed | 17 |
| Starting cash | $45,059.70 |
| Ending cash | $44,809.70 |

### Cash balance changes

Three distinct cash balance changes are visible in `test_results.csv`:

- **Request 3 (Apr 04)** — A4 paper sale recorded; cash dropped from $45,059.70 to $44,559.70
- **Request 7 (Apr 07)** — supplier reorder costs deducted; cash held at $44,559.70
- **Request 14 (Apr 09)** — A4 paper sale revenue added; cash rose to $44,809.70

### Successfully fulfilled requests

- **Request 3 (Apr 04)** — 10,000 sheets of A4 paper at $0.05/sheet, $500.00 total, delivery April 11
- **Request 7 (Apr 07)** — glossy A4, matte A3, poster boards, and heavyweight cardstock confirmed; $320.00 total, delivery April 11 ahead of the April 15 deadline
- **Request 14 (Apr 09)** — 5,000 sheets of A4 paper with 15% bulk discount (orders over 1,000 units), $212.50 total, delivery April 16

### Unfulfilled requests and reasons

17 requests failed. The dominant cause was informal customer language not matching exact catalog entries — for example "glossy A4 paper", "recycled cardstock", "poster board", and "washi tape" are not catalog entries. A secondary cause was items existing in the catalog but not being seeded into the inventory table at startup (only ~40% of catalog items are loaded).

**Strengths:** Agent routing worked reliably across all 20 runs. Bulk discounts were calculated and communicated correctly where items were found. The communications agent successfully removed internal error language and placeholder text from all responses, producing clean customer-facing output throughout.

---

## 3. Suggested Improvements

**Fuzzy name matching** — map informal customer descriptions to catalog entries using string similarity (e.g. `difflib.get_close_matches`) so "glossy A4 paper" resolves to "Glossy paper". This would resolve the majority of the 17 failures.

**Full catalog seeding** — currently only ~40% of catalog items are loaded into inventory at startup. Seeding all items (with zero stock where needed) would let the system acknowledge items exist and offer reorder estimates instead of reporting them as unavailable.