# Ticket 0004 — Stop filing away unused trained models

**Status:** stub — detail before starting (drafted 2026-10-14)

## Problem

Two pages ("Cross Prediction: global-PFT" and "Cross Prediction: PFT-PFT")
save not only the results table but also the last trained model (a random
forest of 100 decision trees, together with every cached result).
These models are never looked at again. One page uses the model only for
a yes/no check ("did the training succeed?").

## Proposed fix

Save only the results table plus a yes/no success flag, not the model.
This touches only the two page files; the mathematical core (`functions.py`)
stays untouched.
