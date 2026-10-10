#!/usr/bin/env python3
"""Generate a small PaySim-shaped CSV for smoke-testing the pipeline.

Fraud = takeover of a quiet account followed by a large TRANSFER/CASH_OUT to a fresh receiver.
Like real PaySim, fraudulent transfers are cancelled, so balances are not updated (leak).
Usage: python make_toy_data.py toy_paysim.csv 200000
"""
import sys

import numpy as np
import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "toy_paysim.csv"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
rng = np.random.default_rng(0)

step = np.sort(rng.integers(1, 745, n))
snd = rng.integers(0, n // 20, n)
rcv = rng.integers(0, n // 15, n)
typ = rng.choice(["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"], n, p=[.35, .22, .35, .06, .02])
amount = np.round(rng.lognormal(7.0, 1.2, n), 2)

fraud = (rng.random(n) < 0.004) & np.isin(typ, ["TRANSFER", "CASH_OUT"])
amount[fraud] = np.round(rng.lognormal(10.5, 0.6, fraud.sum()), 2)
rcv[fraud] = rng.integers(n // 15, n // 15 + 3000, fraud.sum())   # fresh receivers
snd[fraud] = rng.integers(0, n // 400, fraud.sum())                # a few targeted senders (repeat victims)

old_o = np.round(rng.lognormal(9, 1.5, n), 2)
new_o = np.where(typ == "CASH_IN", old_o + amount, np.maximum(old_o - amount, 0))
old_d = np.round(rng.lognormal(9, 1.5, n), 2)
new_d = old_d + amount
new_o[fraud] = old_o[fraud] - 0                                    # cancelled: origin balance not debited
new_d[fraud] = 0.0                                                 # cancelled: destination not credited
old_o[fraud] = amount[fraud]                                       # account emptied by the attack

pd.DataFrame({
    "step": step, "type": typ, "amount": amount,
    "nameOrig": ["C" + str(i) for i in snd], "oldbalanceOrg": old_o, "newbalanceOrig": new_o,
    "nameDest": ["M" + str(i) if t == "PAYMENT" else "C" + str(i) for i, t in zip(rcv, typ)],
    "oldbalanceDest": old_d, "newbalanceDest": new_d,
    "isFraud": fraud.astype(int), "isFlaggedFraud": 0,
}).to_csv(out, index=False)
print(f"wrote {out}: {n:,} rows, {fraud.sum():,} frauds")
