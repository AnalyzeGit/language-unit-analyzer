#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sqlalchemy import create_engine
import io
import pyodbc

def load_database(data,table_name):

    # CUBRID DB 연결을 설정합니다.
    dsn = 'my_cubrid_dsn'  # ODBC 데이터 소스 관리자에서 설정한 DSN 이름
    username = 'dba'
    password = '!^admin1234^!'

    # CUBRID ODBC 연결 문자열
    connection_string = f"DSN={dsn};UID={username};PWD={password};CHARSET=UTF-8"

    column_definitions = {
        'STAT_NGRAM_NO': 'BIGINT',
        'CORPUS_CL': 'VARCHAR(50)',
        'ANALS_YEAR': 'CHAR(4)',
        'ANALS_TOPIC': 'VARCHAR(100)',
        'MORPHEME': 'VARCHAR(20)',
        'NGRAM_TY': 'INTEGER',
        'NGRAM_ORDR': 'INTEGER',
        'NGRAM_FREQUENCY': 'INTEGER',
        'NGRAM_VAL': 'VARCHAR(100)'
        }

    def get_sql_type(column_name):
        return column_definitions.get(column_name, 'VARCHAR(255)')
        
    # 데이터 프레임을 CUBRID DB에 적재합니다.
    try:
        print("Connecting to the database...")
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        # 테이블 존재 여부 확인
        check_table_query = f"SELECT class_name FROM db_class WHERE class_name = '{table_name.upper()}'"
        cursor.execute(check_table_query)
        table_exists = cursor.fetchone()
        
        if not table_exists:
            print(f"Creating table {table_name}...")
            # 데이터 프레임의 스키마에 따라 테이블 생성
            columns = ', '.join([f"{col} {get_sql_type(col)}" for col in data.columns])
            create_table_query = f"CREATE TABLE {table_name} ({columns})"
            cursor.execute(create_table_query)
            print(f"Table '{table_name}' created.")

        
        # DataFrame 'data'가 비어있는지 확인
        if not data.empty:
            # 데이터 프레임의 각 행을 삽입
            print(f"Inserting data into {table_name}...")
            for index, row in data.iterrows():
                columns = ', '.join(row.index)
                values = ', '.join(["'{}'".format(str(value).replace("'", "''")) for value in row.values])
                insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"
                cursor.execute(insert_query)
            conn.commit()
            print(f'DataFrame has been successfully loaded into {table_name} table in the database.')
        else:
            # 빈 데이터 프레임인 경우
            print("The DataFrame is empty. No data was loaded into the table.")
    except Exception as e:
        # 예외 발생
        print(f"Failed to load data: {e}")
    finally:
        cursor.close()
        conn.close()
        
