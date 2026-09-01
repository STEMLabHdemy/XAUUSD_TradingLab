# Paper comparison V1

Every completed M1 candle produces **one** inference from the selected model.
The resulting `signal_id` is the completed-bar UTC timestamp in milliseconds and
is fanned out to four independent virtual ledgers with identical tick, spread,
cost, capital and size inputs.

| ID | Strategy | Difference from control |
| --- | --- | --- |
| A | Baseline | Current burst entries and immediate probability reversal. |
| B | Anti-burst | One open position; after a stop, the same side stays blocked until two non-directional completed M1 bars and a later return of that side. |
| C | Smart SHORT | LONG reversals close immediately; the first consecutive SHORT reversal enables protection, the second closes. |
| D | Combined | B and C together. |

For Smart SHORT, configurable defaults are: net break-even at +2, lock +2 at
+4, and trailing distance 2 after +6. Stops can only tighten and every update
is recorded as a `SMART_PROTECT` event.

The older Paper state remains in `data/live/paper/` as an archived ledger. The
new independent ledgers live under `data/live/paper/comparison_v1/`.
