import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Подготовка данных: создаем синтетические данные
np.random.seed(42)

# Данные о клиентах
customers = pd.DataFrame({
    'customer_id': range(1, 1001),
    'signup_date': pd.date_range(start='2022-01-01', periods=1000, freq='D'),
    'country': np.random.choice(['USA', 'UK', 'Germany'], 1000),
    'channel': np.random.choice(['Organic', 'Paid', 'Referral'], 1000)
})

# Данные о транзакциях
transactions = pd.DataFrame({
    'customer_id': np.random.randint(1, 1001, 5000),
    'transaction_date': pd.date_range(start='2022-01-01', periods=5000, freq='H'),
    'amount': np.random.uniform(10, 100, 5000)
})

# Данные о маркетинговых затратах
marketing_costs = pd.DataFrame({
    'channel': np.random.choice(['Organic', 'Paid', 'Referral'], 100),
    'date': pd.date_range(start='2022-01-01', periods=100, freq='D'),
    'cost': np.random.uniform(100, 1000, 100)
})

# Когортный анализ и добавляем месяц регистрации
customers['signup_month'] = customers['signup_date'].dt.to_period('M')

# Объединяем данные о транзакциях и клиентах
data = pd.merge(transactions, customers, on='customer_id')

# Создаем когорты
cohorts = data.groupby(['signup_month', 'transaction_date']).agg({'customer_id': 'nunique'}).reset_index()
cohorts['period_number'] = (cohorts['transaction_date'].dt.to_period('M') - cohorts['signup_month']).apply(lambda x: x.n)

# Pivot-таблица для Retention Rate
cohort_pivot = cohorts.pivot_table(index='signup_month', columns='period_number', values='customer_id')
cohort_pivot = cohort_pivot.divide(cohort_pivot.iloc[:, 0], axis=0)  # Нормализация

# Расчет LTV (Lifetime Value)

# Средний доход на клиента
avg_revenue_per_user = data.groupby('customer_id')['amount'].sum().mean()

# Churn Rate
churn_rate = 1 - cohort_pivot.iloc[:, 1:].mean(axis=1).mean()  # Средний Churn Rate
ltv = avg_revenue_per_user * (1 / churn_rate)
print(f"LTV: {ltv:.2f}")

# Расчет CAC (Customer Acquisition Cost)

# Общие затраты на маркетинг
total_marketing_costs = marketing_costs['cost'].sum()

# Количество привлеченных клиентов
acquired_customers = customers['customer_id'].nunique()

# CAC
cac = total_marketing_costs / acquired_customers
print(f"CAC: {cac:.2f}")

# Анализ юнит-экономики

# Средний чек
avg_order_value = transactions['amount'].mean()

# Маржинальность (предположим, что себестоимость 50%)
margin = 0.5
contribution_margin = avg_order_value * margin

# Окупаемость CAC
payback_period = cac / contribution_margin
print(f"Payback Period: {payback_period:.2f} месяцев")

# Факторный анализ отклонений выручки/EBITDA

# Пример факторного анализа
revenue = transactions['amount'].sum()
ebitda = revenue * margin - total_marketing_costs

# Факторы
factor_customers = (acquired_customers - 900) * avg_revenue_per_user  # Предположим, что ожидалось 900 клиентов
factor_avg_order = (avg_order_value - 50) * acquired_customers  # Предположим, что ожидался средний чек 50
factor_costs = total_marketing_costs - 50000  # Предположим, что ожидались затраты 50,000

print(f"Отклонение выручки: {factor_customers + factor_avg_order:.2f}")
print(f"Отклонение EBITDA: {factor_customers + factor_avg_order - factor_costs:.2f}")

# Визуализация результатов

# Retention Rate Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cohort_pivot, annot=True, fmt='.0%', cmap='Blues')
plt.title('Retention Rate по когортам')
plt.xlabel('Период')
plt.ylabel('Когорта')
plt.show()

# График LTV и CAC
plt.figure(figsize=(8, 5))
plt.bar(['LTV', 'CAC'], [ltv, cac], color=['blue', 'orange'])
plt.title('LTV и CAC')
plt.ylabel('Сумма')
plt.show()

# График отклонений
plt.figure(figsize=(8, 5))
plt.bar(['Отклонение выручки', 'Отклонение EBITDA'], [factor_customers + factor_avg_order, factor_customers + factor_avg_order - factor_costs], color=['green', 'red'])
plt.title('Факторный анализ отклонений')
plt.ylabel('Сумма')
plt.show()
