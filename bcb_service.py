from bcb import currency
from datetime import datetime, timedelta

def get_currency(before_days):
    start= datetime.now() - timedelta(before_days)
    start = start.date()
    df = currency.get(["USD"], start=f"{start}", end=f"{datetime.now().date()}", side="ask")
    return df