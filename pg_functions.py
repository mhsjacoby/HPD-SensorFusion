"""
pg_functions.py
Author: Maggie Jacoby
Last updated: September 7, 2020

Functions for interacting with a PostgreSQL database from python

"""


import sys
import psycopg2
import psycopg2.extras as extras

import pandas as pd

connection_parameters = {
    'host': 'localhost',
    'db':'hpd_mobile',
    'usr':'maggie',
    'pw':'arpa-e'
}




class Structs():
    def __init__(self):
        
        self.inference_table = """
            CREATE TABLE curr_schema.tab (
                entry_id integer PRIMARY KEY,
                day date NOT NULL,
                hr_min_sec time without time zone NOT NULL,
                hub character(3) NOT NULL,
                img integer,
                audio integer,
                env integer,
                occupied integer NOT NULL
            )
            """

        self.probability_table = """
            CREATE TABLE curr_schema.tab (
                entry_id integer PRIMARY KEY,
                day date NOT NULL,
                hr_min_sec time without time zone NOT NULL,
                hub character(3) NOT NULL,
                img numeric(4,3),
                audio numeric(4,3),
                temp numeric(4,3),
                rh numeric(4,3),
                light numeric(4,3),
                co2eq numeric(4,3),
                occupied integer NOT NULL
            )
            """
        
        self.drop = """
            DROP TABLE curr_schema.tab ;
            """


class PostgreSQL(Structs):
    def __init__(self, home_parameters, connection_params=connection_parameters, schema="sixhourfill"):
        self.P = connection_params
        self.home = home_parameters["home"]
        self.schema = schema
        Structs.__init__(self)


    def connect(self):
        conn = None
        
        try:
            conn = psycopg2.connect(
                host=self.P["host"], 
                database=self.P["db"], 
                user=self.P["usr"], 
                password=self.P["pw"])
        except (Exception, psycopg2.DatabaseError) as error:
            print(f'Error with connect: {error}')

        print(f'\n>Connecting to database: {self.P["db"]}')
        return conn
    
    
    def PG_connect(self, execute_statement, sucess_statement):
        try:
            conn = self.connect()
            cur = conn.cursor()         
            cur.execute(execute_statement)
            conn.commit() 
            cur.close()
            print(sucess_statement) 

        except (Exception, psycopg2.DatabaseError) as error:
            print(f'Error with PG_connect: {error}')

        finally:
            if conn is not None:
                conn.close()
                print('>Database connection closed.')  


    
    def create_inf_table(self, table_name=None):
        schema = self.schema
        print(f'Creating inference table: {table_name}')      
        ex = self.inference_table.replace("tab", table_name).replace("curr_schema", schema)
        self.PG_connect(ex, f'Table {table_name} created sucessfully!')

    def create_prob_table(self, table_name=None):
        schema = self.schema
        print(f'Creating probability table: {table_name}')    
        ex = self.probability_table.replace("tab", table_name).replace("curr_schema", schema)
        self.PG_connect(ex, f'Table {table_name} created sucessfully!')

        
    def drop_table(self, table_name):
        schema = self.schema
        print(f'Dropping table: {table_name}.')
        ex = self.drop.replace("tab", table_name).replace("curr_schema", schema)
        self.PG_connect(ex, f'Table {table_name} sucessfully dropped.')

    
    def insert_table(self, df, table):
        schema = self.schema
        conn = self.connect()
        table = f'{schema}.{table}'
        print(f'Table to insert: {table}')
        try:
            tuples = [tuple(x) for x in df.to_numpy()]
            cols = ",".join(list(df.columns))
            query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
            cur = conn.cursor()
            try:
                extras.execute_values(cur, query, tuples)
                conn.commit()
            except (Exception, psycopg2.DatabaseError) as error:
                print(f'Error with INSERT: {error}') 
                conn.rollback()
                cur.close()

            print(f'Table {table} sucesfully inserted from pandas df.')
            cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(f'Error with insert_table: {error}')
        
        finally:
            if conn is not None:
                conn.close()
                print('>Database connection closed.')



    def query_db(self, query, conn=None):
        conn = self.connect()
        cur = conn.cursor()
        table = pd.read_sql_query(query, conn)
        return table
