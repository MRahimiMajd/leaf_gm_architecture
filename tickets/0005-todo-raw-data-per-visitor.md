# Ticket 0005 — One copy of the raw data, not one per visitor

**Status:** stub — detail before starting (drafted 2026-10-14)

## Problem

Each visitor's session causes the server to re-read the Excel file and
keep its own in-memory copy. The file is only used only to show the
min/max values of input fields on one page. With multiple visitors,
the same data is duplicated in memory.

## Proposed fix

Compute the small table of smallest/largest values once, remember that instead,
and stop keeping the full raw table per visitor.

## Required evidence

Before changing anything, write a small measurement script that
simulates a visitor session and shows how much memory the app uses today.
Run it again after the change to prove the improvement. The script reads the
memory use directly from the operating system, so no extra tools are needed.

## Glossary

- **Session:** one visit to the web app, from opening the page to leaving it.
  Memory that the server keeps tied to one visit is called *session state*.
