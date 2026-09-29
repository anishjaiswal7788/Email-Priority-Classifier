"""Shared feature engineering (used by training AND the app)."""
import re
import numpy as np
import pandas as pd

URGENT = r"urgent|asap|immediately|critical|escalat|deadline|action required|failed|down|overdue|emergency|now"
DEADLINE = r"today|tomorrow|tonight|eod|by \d|within \d+ ?(hour|hr)|this morning"
PROMO = r"sale|% off|discount|offer|unsubscribe|newsletter|webinar|subscribe|deal|coupon|follow"
VIP_DOMAINS = ("ceo.", "client", "boss", "hr.", "university.edu", "bank")

def prepare(df):
    df = df.copy()
    df["text"] = df["subject"].fillna("") + " . " + df["body"].fillna("")
    return df

def meta_features(df):
    t = (df["subject"].fillna("") + " " + df["body"].fillna(""))
    tl = t.str.lower()
    caps = t.apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))
    vip = df["sender"].fillna("").str.lower().apply(lambda s: int(any(d in s for d in VIP_DOMAINS)))
    return np.c_[
        tl.str.count(URGENT), tl.str.count(DEADLINE), tl.str.count(PROMO),
        t.str.count("!"), caps, vip, t.str.contains(r"\?").astype(int),
        np.log1p(t.str.len()),
    ]
