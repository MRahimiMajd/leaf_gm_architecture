# Ticket 0003 — Stored results must expire

**Status:** stub — detail before starting (drafted 2026-10-14)

## Problem

To avoid repeating slow calculations, the app remembers every result forever
("caching"). It does this for every combination of settings any visitor ever
tried: every plant group, every trait selection, every ensemble size. The app
has been running for over two years, so this collection grows without end.
This unbounded collection might be the reason that the app eventually ran out
of memory.

## Proposed fix (numbers agreed 2026-10-14)

Every stored result expires after 6 hours, and at most 16 results are kept per
calculation type. Worst case for a visitor: a rarely used combination is
calculated again, which takes roughly 1-5 minutes.

## Plan (rough)

- Add a small automatic check that every cached calculation in the code
  declares an expiry and a limit.
- Add the expiry/limit to all cached functions (data loading included).
- Reuse the memory-measurement script from Ticket 0005 to show that memory now
  stays bounded while many different settings are used.

## Glossary

- **Cache / caching:** keeping calculation results in memory so they do not
  have to be computed again on the next visit or click.
