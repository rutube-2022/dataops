from airflow import DAG
from airflow.models import Variable
from datetime import datetime
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
import sqlite3
import pandas as pd
import numpy as np
import sqlalchemy
import multiprocessing as mp 
from math import sqrt
import optuna                                                                                                
import logging
from hide_warnings import hide_warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, median_absolute_error
import mlflow
import os
from mlflow.models import infer_signature
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.preprocessing import StandardScaler

count = Variable.set('count', 135169)
engine = sqlalchemy.create_engine("sqlite:////opt/airflow/variant1DB.db")

dag = DAG (
    'tusur-de')
    #schedule_interval='*/15 * * * *',
    #start_date=datetime(2025, 1, 1),
    #catchup=False)

def read_data(task_instance):
    count = int(Variable.get('count'))
    part = int(Variable.get('part', default_var=1000))
    if part == 1000:
        Variable.set('part', 0)
        part = 0 
    if part > 9:
        part = 0
        Variable.set('part', part)
    df = pd.read_sql(f'SELECT "store_id", "order_id", "product_id", "price", "profit", "delivery_distance", "date_create", "order_start_prepare", "planned_prep_time", "order_ready" FROM  joined_table LIMIT {count} OFFSET {part * count}', 
                 engine)
    task_instance.xcom_push(key="df_data", value=df)    

def pre_cleaning(task_instance):
    df=task_instance.xcom_pull(key="df_data", task_ids="read_data_task")
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df.replace('', None, inplace=True)
    df.replace(0, None, inplace=True)
    df['profit'] = pd.to_numeric(df['profit'], errors='coerce')
    df['delivery_distance'] = pd.to_numeric(df['delivery_distance'], errors='coerce')
    df['date_create'] = pd.to_datetime(df['date_create'], errors='coerce')
    df['order_start_prepare'] = pd.to_datetime(df['order_start_prepare'], errors='coerce')
    df['order_ready'] = pd.to_datetime(df['order_ready'], errors='coerce')
    df=df.dropna()
    task_instance.xcom_push(key="df_pre_cl", value=df)

def formation_columns(task_instance):
    df=task_instance.xcom_pull(key="df_pre_cl", task_ids="pre_cleaning_task")
    dfs = df.groupby('order_id')['price'].sum().reset_index().rename(columns={'price': 'order_price'})
    dfu=df['order_id'].value_counts()
    df = df.drop_duplicates(subset=['order_id'])
    df = df.reset_index(drop=True)
    df = df.merge(dfs, on='order_id').merge(dfu, on='order_id')
    del dfu
    del dfs
    df = df.rename(columns={'count': 'items_count'})
    df['prep_minutes'] = (df['order_ready'] - df['order_start_prepare']).dt.total_seconds() / 60
    df = df.drop(columns=['price', 'product_id', 'order_id', 'order_ready', 'planned_prep_time', 'order_start_prepare'])
    task_instance.xcom_push(key="df_form_col", value=df)

def data_processing(task_instance):
    df=task_instance.xcom_pull(key="df_form_col", task_ids="formation_columns_task")
    def iqr_search_spdf(data, feature, threshold = 1.5):                                           #16
        IQR = data[feature].quantile(0.75) - data[feature].quantile(0.25)
        low = data[feature].quantile(0.25) - (IQR * threshold)
        up = data[feature].quantile(0.75) + (IQR * threshold)
        outliers = (data[feature] < low) | (data[feature] > up)
        print (f'информация по столбцу: {feature}')
        print (f'нижняя граница: {low}, верхняя граница: {up}')
        print(f'Количество выбросов в данных: {outliers.sum()}')
        print(f'Доля выбросов: {outliers.sum()/len(data[feature])}')
        print()
        return low, up
    p_l, p_u = iqr_search_spdf (df, 'profit')
    d_l, d_u = iqr_search_spdf (df, 'delivery_distance')
    o_l, o_u = iqr_search_spdf (df, 'order_price')
    i_l, i_u = iqr_search_spdf (df, 'items_count')
    m_l, m_u = iqr_search_spdf (df, 'prep_minutes')
    print (f'размер таблицы до начала удаления выбросов: {df.shape}')
    print()
    df = df[(df['profit'] >= p_l) & (df['profit'] <= p_u)]
    print (f'размер таблицы после удаления выбросов из колонки Profit: {df.shape}')
    print()
    df = df[(df['delivery_distance'] >= d_l) & (df['delivery_distance'] <= d_u)]
    print (f'размер таблицы после удаления выбросов из колонки Delivery Distance: {df.shape}')
    print()
    df = df[(df['order_price'] >= o_l) & (df['order_price'] <= o_u)]
    print (f'размер таблицы после удаления выбросов из колонки Order Price: {df.shape}')
    print()
    df = df[(df['items_count'] >= i_l) & (df['items_count'] <= i_u)]
    print (f'размер таблицы после удаления выбросов из колонки Items Count: {df.shape}')
    print()
    df = df[(df['prep_minutes'] >= m_l) & (df['prep_minutes'] <= m_u)]
    print (f'размер таблицы после удаления выбросов из колонки Prep_minutes: {df.shape}')
    del p_l, p_u, d_l, d_u, o_l, o_u, i_l, i_u, m_l, m_u

    def cyclic_encode(value, max_value):                                                           #17
        sine = np.sin(2 * np.pi * value / max_value)
        cosine = np.cos(2 * np.pi * value / max_value)
        return sine, cosine
    df['month_cr_sin'], df['month_cr_cos'] = cyclic_encode(df['date_create'].dt.month, 12)
    df['day_cr_sin'], df['day_cr_cos'] = cyclic_encode(df['date_create'].dt.day, 31)
    df['hour_cr_sin'], df['hour_cr_cos'] = cyclic_encode(df['date_create'].dt.hour, 24)
    df['minute_cr_sin'], df['minute_cr_cos'] = cyclic_encode(df['date_create'].dt.minute, 60)
    df['second_cr_sin'], df['second_cr_cos'] = cyclic_encode(df['date_create'].dt.second, 60)
    df['year_cr'] = df['date_create'].dt.year
    df = df.drop(['date_create'], axis=1)

    task_instance.xcom_push(key="df_data_proc", value=df)

def pearson(task_instance):
    df=task_instance.xcom_pull(key="df_data_proc", task_ids="data_processing_task")
    #num_chunks = mp.cpu_count()
    #chunks = np.array_split(df, num_chunks)
    #def process_data (df_chunk):
    #    pearson_corr = df_chunk.corr(method='pearson')
    #    return pearson_corr
    #with mp.Pool(processes=num_chunks) as pool:
    #    chunk_results = pool.map(process_data, chunks)
    #combined_corr = pd.DataFrame(np.zeros((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns)
    #for chunk_corr in chunk_results:
    #    combined_corr += chunk_corr
    #combined_corr /= num_chunks
    filtered_corr_ps = df.corr(method='pearson') #combined_corr[(combined_corr > 0) & (combined_corr != 1.0)]
    filtered_corr_ps = filtered_corr_ps.dropna(how='all').dropna(axis=1, how='all')
    filtered_corr_ps = filtered_corr_ps.loc[['prep_minutes']].dropna(axis=1)
    task_instance.xcom_push(key="pearson", value=filtered_corr_ps)

def spearman(task_instance):
    df=task_instance.xcom_pull(key="df_data_proc", task_ids="data_processing_task")
    #num_chunks = mp.cpu_count()
    #chunks = np.array_split(df, num_chunks)
    #def process_data (df_chunk):
    #    pearson_corr = df_chunk.corr(method='spearman')
    #    return pearson_corr
    #with mp.Pool(processes=num_chunks) as pool:
    #    chunk_results = pool.map(process_data, chunks)
    #combined_corr = pd.DataFrame(np.zeros((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns)
    #for chunk_corr in chunk_results:
    #    combined_corr += chunk_corr
    #combined_corr /= num_chunks
    filtered_corr_sp = df.corr(method='spearman') #combined_corr[(combined_corr > 0) & (combined_corr != 1.0)]
    filtered_corr_sp = filtered_corr_sp.dropna(how='all').dropna(axis=1, how='all')
    filtered_corr_sp = filtered_corr_sp.loc[['prep_minutes']].dropna(axis=1)
    task_instance.xcom_push(key="spearman", value=filtered_corr_sp)

def kendall(task_instance):
    df=task_instance.xcom_pull(key="df_data_proc", task_ids="data_processing_task")
    #num_chunks = mp.cpu_count()
    #chunks = np.array_split(df, num_chunks)
    #def process_data (df_chunk):
    #    pearson_corr = df_chunk.corr(method='kendall')
    #    return pearson_corr
    #with mp.Pool(processes=num_chunks) as pool:
    #    chunk_results = pool.map(process_data, chunks)
    #combined_corr = pd.DataFrame(np.zeros((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns)
    #for chunk_corr in chunk_results:
    #    combined_corr += chunk_corr
    #combined_corr /= num_chunks
    filtered_corr_kn = df.corr(method='kendall') #combined_corr[(combined_corr > 0) & (combined_corr != 1.0)]
    filtered_corr_kn = filtered_corr_kn.dropna(how='all').dropna(axis=1, how='all')
    filtered_corr_kn = filtered_corr_kn.loc[['prep_minutes']].dropna(axis=1)
    task_instance.xcom_push(key="kendall", value=filtered_corr_kn)

def final_data_prepare(task_instance):
    part = int(Variable.get('part'))
    if part == 0:
        pearson=task_instance.xcom_pull(key="pearson", task_ids="pearson_task")
        spearman=task_instance.xcom_pull(key="spearman", task_ids="spearman_task")
        kendall=task_instance.xcom_pull(key="kendall", task_ids="kendall_task")
        cor_columns = set(pearson.columns) | set(spearman.columns) | set(kendall.columns)
        cor_columns.add('prep_minutes')
        cor_columns = list(cor_columns)
        Variable.set('cor_columns', cor_columns)
    else:
        cor_columns = eval(Variable.get('cor_columns'))
    print (cor_columns)
    df=task_instance.xcom_pull(key="df_data_proc", task_ids="data_processing_task")
    df = df[cor_columns]
    task_instance.xcom_push(key="df_final_data_prep", value=df)

def ML(task_instance):
    df=task_instance.xcom_pull(key="df_final_data_prep", task_ids="final_data_prepare_task")
    print (df.dtypes)
    X = df.drop(columns=['prep_minutes'])                                                                         #23
    y = df['prep_minutes']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4,random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.25, random_state=42)
    scalerS = StandardScaler()
    X_trainS = scalerS.fit_transform(X_train)
    X_testS = scalerS.transform(X_test)
    X_valS = scalerS.transform(X_val)

    mlflow.set_tracking_uri(uri="http://ml.fatal-error.ru:8181")
    if mlflow.get_experiment_by_name("Tusur-DE") is None:
        mlflow.create_experiment("Tusur-DE", artifact_location="s3://kserve/mlflow")
    mlflow.set_experiment("Tusur-DE")

    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_DEFAULT_REGION = os.environ.get('AWS_DEFAULT_REGION')
    AWS_ENDPOINT_URL = os.environ.get('AWS_ENDPOINT_URL')
    
    mlflow.start_run(run_name=f"PAR")
    for scor in ['neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_median_absolute_error']:
        @hide_warnings (out = False)
        def PAR(trial):
            C = trial.suggest_float('C', 1e-3, 1e+2)
            fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
            max_iter = trial.suggest_int('max_iter', 100, 10000)
            tol = trial.suggest_float('tol', 1e-6, 1e-1)
            early_stopping = trial.suggest_categorical('early_stopping', [True, False])
            #validation_fraction = trial.suggest_float('validation_fraction', 0.1, 1.0) if early_stopping==True else 0.1
            n_iter_no_change = trial.suggest_int('n_iter_no_change', 1, 10)
            shuffle = trial.suggest_categorical('shuffle', [True, False])
            loss = trial.suggest_categorical('loss', ['epsilon_insensitive', 'squared_epsilon_insensitive'])
            epsilon = trial.suggest_float('epsilon', 0.0, 1.0)

            model = PassiveAggressiveRegressor(
                C=C, 
                fit_intercept=fit_intercept, 
                max_iter=max_iter, 
                tol=tol, 
                early_stopping=early_stopping, 
                #validation_fraction=validation_fraction, 
                n_iter_no_change=n_iter_no_change, 
                shuffle=shuffle, 
                loss=loss, 
                epsilon=epsilon,
                random_state=42
                )
            
            # Кросс-валидация модели и вычисление средней точности
            score = cross_val_score(model, X_valS , y_val, cv=5, scoring=scor, n_jobs=-1).mean()
                
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(PAR, n_trials=100)
        print(f'Начинается подбор гипер параметров с использованеим метрики {scor.upper()}')
        print()
        print(f"Лучшие гиперпараметры для обучения при использовании метрики {scor}: {study.best_params}")
        model = PassiveAggressiveRegressor(
            C=study.best_params.get('C'), 
            fit_intercept=study.best_params.get('fit_intercept'), 
            max_iter=study.best_params.get('max_iter'), 
            tol=study.best_params.get('tol'), 
            early_stopping=study.best_params.get('early_stopping'), 
            #validation_fraction=study.best_params.get('validation_fraction'), 
            n_iter_no_change=study.best_params.get('n_iter_no_change'), 
            shuffle=study.best_params.get('shuffle'), 
            loss=study.best_params.get('loss'), 
            epsilon=study.best_params.get('epsilon'),
            random_state=42
            )
        
        model.fit(X_trainS, y_train)
        
        y_pred = model.predict(X_testS)
        mse = mean_squared_error (y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
                        
        mlflow.start_run(run_name=f"par_with_{scor}", nested=True)
        mlflow.log_params(study.best_params)
        mlflow.log_metrics({'rmse': sqrt(mse), 'mape': mape})
        signature = infer_signature(X_testS, y_pred)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"par_with_{scor}",
            signature=signature,
            input_example=X_testS,
            registered_model_name=f"par_with_{scor}")
        mlflow.end_run()

        print()
        print(f'При обучении модели с вышеуказанными гипрепараметрами достигнуты такие результаты:') 
        print(f'RMSE: {sqrt(mse)}')
        print(f'MAPE: {mape}')
        print()
        print('*'*120)
        print()
    mlflow.end_run()

    part = int(Variable.get('part'))
    part += 1
    Variable.set('part', part)

def further_train(task_instance):
    df=task_instance.xcom_pull(key="df_final_data_prep", task_ids="final_data_prepare_task")
    print (df.dtypes)
    X = df.drop(columns=['prep_minutes'])
    y = df['prep_minutes']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4,random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.25, random_state=42)
    scalerS = StandardScaler()
    X_trainS = scalerS.fit_transform(X_train)
    X_testS = scalerS.transform(X_test)
    X_valS = scalerS.transform(X_val)

    mlflow.set_tracking_uri(uri="http://ml.fatal-error.ru:8181")
    mlflow.set_experiment("Tusur-DE")

    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_DEFAULT_REGION = os.environ.get('AWS_DEFAULT_REGION')
    AWS_ENDPOINT_URL = os.environ.get('AWS_ENDPOINT_URL')

    mlflow.start_run(run_name="PAR")
    for ml in mlflow.search_registered_models(filter_string="name LIKE '%par%'"):
        for data in ml.latest_versions:
            model = mlflow.sklearn.load_model(f"models:/{data.name}/{data.version}")
            model.partial_fit(X_trainS, y_train)
        
            y_pred = model.predict(X_testS)
            mse = mean_squared_error (y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
                       
            mlflow.start_run(run_id=data.run_id, nested=True)
            mlflow.log_metrics({'rmse': sqrt(mse), 'mape': mape})
            signature = infer_signature(X_testS, y_pred)
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=data.name,
                signature=signature,
                input_example=X_testS,
                registered_model_name=data.name)
            mlflow.end_run()

            print()
            print(f'При дообучении модели {data.name} достигнуты такие результаты:') 
            print(f'RMSE: {sqrt(mse)}')
            print(f'MAPE: {mape}')
            print()
            print('*'*120)
            print()
    mlflow.end_run()

    part = int(Variable.get('part'))
    part += 1
    Variable.set('part', part)

def choose_task():
    condition = int(Variable.get("part"))
    if condition == 0:
        return 'ML_task'
    else:
        return 'further_train_task'

def choose_cor():
    condition = int(Variable.get("part"))
    if condition == 0:
        return ['pearson_task', 'spearman_task', 'kendall_task']
    else:
        return 'final_data_prepare_task'
    
read_data_task = PythonOperator(
    task_id="read_data_task",
    python_callable=read_data,
    dag=dag,
)

pre_cleaning_task = PythonOperator(
    task_id="pre_cleaning_task",
    python_callable=pre_cleaning,
    dag=dag,
)
formation_columns_task = PythonOperator(
    task_id="formation_columns_task",
    python_callable=formation_columns,
    dag=dag,
)
data_processing_task = PythonOperator(
    task_id="data_processing_task",
    python_callable=data_processing,
    dag=dag,
)
pearson_task = PythonOperator(
    task_id="pearson_task",
    python_callable=pearson,
    dag=dag,
)
spearman_task = PythonOperator(
    task_id="spearman_task",
    python_callable=spearman,
    dag=dag,
)
kendall_task = PythonOperator(
    task_id="kendall_task",
    python_callable=kendall,
    dag=dag,
)
final_data_prepare_task = PythonOperator(
    task_id="final_data_prepare_task",
    python_callable=final_data_prepare,
    dag=dag,
    trigger_rule='none_failed'
)
ML_task = PythonOperator(
    task_id="ML_task",
    python_callable=ML,
    dag=dag,
)
further_train_task = PythonOperator(
    task_id="further_train_task",
    python_callable=further_train,
    dag=dag,
)
branching = BranchPythonOperator(
    task_id="branching",
    python_callable=choose_task,
    dag=dag
    )
choose_cor = BranchPythonOperator(
    task_id="choose_cor",
    python_callable=choose_cor,
    dag=dag
    )

read_data_task >> pre_cleaning_task >> formation_columns_task >> data_processing_task >> choose_cor >> [pearson_task, spearman_task, kendall_task] >> final_data_prepare_task >> branching >> [ML_task, further_train_task]
