import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# Данные о клиентах
customers = pd.DataFrame(
    {
        "customer_id": range(1, 1001),
        "signup_date": pd.date_range(start="2022-01-01", periods=1000, freq="D"),
        "country": np.random.choice(["USA", "UK", "Germany"], 1000),
        "channel": np.random.choice(["Organic", "Paid", "Referral"], 1000),
    }
)

# Данные о транзакциях
transactions = pd.DataFrame(
    {
        "customer_id": np.random.randint(1, 1001, 5000),
        "transaction_date": pd.date_range(start="2022-01-01", periods=5000, freq="h"),
        "amount": np.random.uniform(10, 100, 5000),
    }
)

# Данные о маркетинговых затратах
marketing_costs = pd.DataFrame(
    {
        "channel": np.random.choice(["Organic", "Paid", "Referral"], 100),
        "date": pd.date_range(start="2022-01-01", periods=100, freq="D"),
        "cost": np.random.uniform(100, 1000, 100),
    }
)

# Суммирование затрат по каналам
marketing_costs = marketing_costs.groupby("channel", as_index=False)["cost"].sum()

# Когортный анализ
customers["signup_month"] = customers["signup_date"].dt.to_period("M")
data = pd.merge(transactions, customers, on="customer_id")

cohorts = (
    data.groupby(["signup_month", "transaction_date"])
    .agg({"customer_id": "nunique"})
    .reset_index()
)

# Исправленный расчет номера периода
cohorts["period_number"] = (
    cohorts["transaction_date"] - cohorts["signup_month"].dt.start_time
).dt.days // 30

cohort_pivot = cohorts.pivot_table(
    index="signup_month", columns="period_number", values="customer_id"
)
cohort_pivot = cohort_pivot.div(cohort_pivot.iloc[:, 0], axis=0).fillna(0)
