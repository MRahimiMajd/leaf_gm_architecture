# Ticket 0006 — Sensible limits for number inputs

**Status:** stub — detail before starting (drafted 2026-10-14)

## Problem

The number fields on the pages (e.g. "Ensemble Size") accept any value a
visitor types. A single visitor entering an extreme value (say 100,000) starts
a calculation that by itself can exhaust the app's memory.

## Proposed fix

Give the number fields sensible maximum values (and double-check the minimums)
so that extreme inputs are impossible. The app then simply refuses to compute
with unrealistic settings, and no single visitor can bring it down.

Consider raising the smallest allowed ensemble size from 1 to 3:
with fewer repeats, the error columns are empty or unreliable. (Discuss
this with Milad beforehand.)

## Found during ticket 0001 (2026-10-14)

While writing the coverage tests we found a related crash: `RF_with_split`
(functions.py) raises an `UnboundLocalError` instead of returning the NaN
result when the "not enough data" branch fires, because `model` is never
assigned there (`RF_with_train_and_test_data` sets `model = None`). A visitor
on the Cross Validation page triggers it by setting "Minimum Rows" above the
rows available for the chosen PFT group and traits. Fixing the minimum
values for the number fields here should also set `model = None` in that
branch (and ideally show the existing friendly error message instead of an
exception page).
