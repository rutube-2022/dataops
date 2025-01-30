from airflow import DAG
from airflow.operators.python_operator import PythonOperator
#import sqlalchemy
import psycopg2
  
dag = DAG ('dataops')

def check():
    con = psycopg2.connect(host="psql-postgresql.psql.svc",
                           database="airflow",
                           user="airflow1",
                           password="airflow")
    nncur = con.cursor()
    nncur.execute ("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    nncur.close()
    con.close()
check_task = PythonOperator(
    task_id="check_task",
    python_callable=check,
    dag=dag,
)

