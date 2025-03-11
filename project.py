# Импорт библиотек
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 100)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sns.set(style="whitegrid", palette="pastel")

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Данные успешно загружены. Размер данных: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Ошибка загрузки данных: {e}")
        raise

def preprocess_data(df):
    try:
        # Фильтрация колонок
        df = df[['order_id', 'order_date', 'cust_id', 'Customer Since', 'qty_ordered', 'price', 'value', 'discount_amount', 'total']]
        
        # Коррекция количества заказанных товаров
        df['qty_ordered'] = df['qty_ordered'] - 1
        df = df.drop(columns=['price', 'value', 'discount_amount'])
        
        # Преобразование типов данных
        df['order_date'] = pd.to_datetime(df['order_date'], format='%Y-%m-%d')
        df['Customer Since'] = pd.to_datetime(df['Customer Since'], format='%m/%d/%Y')
        df['qty_ordered'] = df['qty_ordered'].astype('int64')
        df = df.drop_duplicates()
        df.columns = ['order_id', 'order_date', 'customer_id', 'customer_since', 'quantity', 'total_sales']
        
        logging.info("Предобработка данных завершена.")
        return df
    except Exception as e:
        logging.error(f"Ошибка предобработки данных: {e}")
        raise

def analyze_data(df):
    try:
        # Группировка данных
        da = pd.DataFrame(df.groupby(["order_date", "customer_id", "order_id"]).agg({
            'customer_since': min,
            'quantity': sum,
            'total_sales': sum
        })).reset_index()
        
        da['order_month'] = da['order_date'].dt.to_period('M')
        da['acquired_year'] = da['customer_since'].dt.to_period('Y')
        da['acquired_year'] = da['acquired_year'].astype('int64')
        da['acquired_year_bins'] = pd.cut(
            x=da['acquired_year'],
            bins=list(range(np.min(da['acquired_year']), np.max(da['acquired_year']) + 5, 5))
        )
        
        # Подсчет уникальных клиентов
        total_cust = pd.DataFrame(da.groupby('acquired_year_bins', as_index=False).agg({'customer_id': 'nunique'}))
        total_cust.columns = ['acquired_year_bins', 'total_cust']
        
        # Создание когортной матрицы
        cohort_matrix = pd.pivot_table(
            da,
            index='acquired_year_bins',
            columns='order_month',
            values='customer_id',
            aggfunc=pd.Series.nunique
        )
        
        # Расчет процента удержания
        cohort_matrix_percentage = cohort_matrix.div(total_cust.iloc[:, 1].values, axis=0)
        
        # Анализ количества товаров и продаж
        cohort_matrix_quantity = pd.pivot_table(
            da,
            index='acquired_year_bins',
            columns='order_month',
            values='quantity',
            aggfunc=pd.Series.mean
        )
        
        cohort_matrix_sales = pd.pivot_table(
            da,
            index='acquired_year_bins',
            columns='order_month',
            values='total_sales',
            aggfunc=pd.Series.median
        )
        
        logging.info("Анализ данных завершен.")
        return total_cust, cohort_matrix, cohort_matrix_percentage, cohort_matrix_quantity, cohort_matrix_sales
    except Exception as e:
        logging.error(f"Ошибка анализа данных: {e}")
        raise

# Визуализация данных
def visualize_data(total_cust, cohort_matrix, cohort_matrix_percentage, cohort_matrix_quantity, cohort_matrix_sales):
    try:
        # График количества клиентов по когортам
        plt.figure(figsize=(10, 6))
        sns.barplot(data=total_cust, x='acquired_year_bins', y='total_cust', palette='mako')
        plt.title('Number of Customers by Cohort', fontsize=12)
        plt.xlabel('Cohort Year')
        plt.ylabel('Number of Customers')
        plt.show()

        # Тепловая карта количества клиентов
        plt.figure(figsize=(10, 6))
        sns.heatmap(cohort_matrix, annot=True, annot_kws={"size": 7}, fmt=".0f", linewidths=.4, cmap="Blues", cbar_kws={'label': 'Number of Customers'})
        plt.title('Number of Customers by Cohort', fontsize=12)
        plt.xlabel('Month')
        plt.ylabel('Acquired Year')
        plt.show()

        # Тепловая карта процента удержания
        plt.figure(figsize=(10, 6))
        sns.heatmap(cohort_matrix_percentage, annot=True, annot_kws={"size": 7}, fmt=".2%", linewidths=.4, cmap="OrRd", cbar_kws={'label': 'Retention Rate'})
        plt.title('Retention Rate by Cohort', fontsize=12)
        plt.xlabel('Month')
        plt.ylabel('Acquired Year')
        plt.show()

        # Тепловая карта среднего количества товаров
        plt.figure(figsize=(10, 6))
        sns.heatmap(cohort_matrix_quantity, annot=True, annot_kws={"size": 7}, fmt=".2f", linewidths=.4, cmap="Purples", cbar_kws={'label': 'Average Items'})
        plt.title('Average Order Items per Cohort Over Time', fontsize=12)
        plt.xlabel('Month')
        plt.ylabel('Acquired Year')
        plt.show()

        # Тепловая карта медианных продаж
        plt.figure(figsize=(10, 6))
        sns.heatmap(cohort_matrix_sales, annot=True, annot_kws={"size": 7}, fmt=".2f", linewidths=.4, cmap="Greens", cbar_kws={'label': 'Median Sales Amount'})
        plt.title('Median Order Sales Amount per Cohort Over Time', fontsize=12)
        plt.xlabel('Month')
        plt.ylabel('Acquired Year')
        plt.show()
        
        logging.info("Визуализация данных завершена.")
    except Exception as e:
        logging.error(f"Ошибка визуализации данных: {e}")
        raise

# Основной блок
if __name__ == "__main__":
    try:
        # Загрузка данных
        df = load_data('sales.csv')
        
        # Предобработка данных
        df = preprocess_data(df)
        
        # Анализ данных
        total_cust, cohort_matrix, cohort_matrix_percentage, cohort_matrix_quantity, cohort_matrix_sales = analyze_data(df)
        
        # Визуализация данных
        visualize_data(total_cust, cohort_matrix, cohort_matrix_percentage, cohort_matrix_quantity, cohort_matrix_sales)
    except Exception as e:
        logging.error(f"Ошибка выполнения программы: {e}")
