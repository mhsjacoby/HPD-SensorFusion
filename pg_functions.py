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
        
        self.drop = """
            DROP TABLE curr_schema.tab ;
            """
        
        # self.select_query = """
        #     SELECT hr_min_sec, hub, audio, env, occupancy
        #     FROM h2_red_inference
        #     """


class PostgreSQL(Structs):
    def __init__(self, home_parameters, connection_params=connection_parameters,):
        self.P = connection_params
        self.home = home_parameters['home']
        Structs.__init__(self)
        self.inf_table=None


    def connect(self):
        conn = None
        
        try:
            conn = psycopg2.connect(
                host=self.P['host'], 
                database=self.P['db'], 
                user=self.P['usr'], 
                password=self.P['pw'])
        except (Exception, psycopg2.DatabaseError) as error:
            print(f'error with connect: {error}')

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
            print(f'Error with connection: {error}')

        finally:
            if conn is not None:
                conn.close()
                print('>Database connection closed.')  
                
    def create_table(self, schema='public', t_name=None):
        table_name = self.home + '_inference' if not t_name else self.home + t_name
        print(f'Creating table: {table_name}')
        self.inf_table = f'{schema}.{table_name}'
        ex = self.inference_table.replace("tab", table_name).replace("curr_schema", schema)
        self.PG_connect(ex, f'Table {table_name} created sucessfully!')
        
        
    def drop_table(self, table_name, schema="public"):
        print(f'Dropping table: {table_name}.')
        ex = self.drop.replace("tab", table_name).replace("curr_schema", schema)
        self.PG_connect(ex, f'Table {table_name} sucessfully dropped.')
    
    
    def insert_table(self, df, table=None, schema="public"):
        conn = self.connect()
        table = self.inf_table if not table else f'{schema}.{table}'
        print(table)
        try:
            tuples = [tuple(x) for x in df.to_numpy()]
            cols = ','.join(list(df.columns))
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
            print(f'Error with connection: {error}')
        
        finally:
            if conn is not None:
                conn.close()
                print('>Database connection closed.')



    def query_db(self, query, conn=None):
        conn = self.connect()
        cur = conn.cursor()
        table = pd.read_sql_query(query, conn)
        return table
