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
