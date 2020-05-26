import sys
import time

def create_table_sql(table_name, args):
    return f"""
        CREATE TABLE {table_name}(
            timestamp float PRIMARY KEY,
            task text NOT NULL,
            parametrization text NOT NULL,
            nepoch INT NOT NULL,
            acc float8 NOT NULL,
            loss float8 NOT NULL,
            exec_time float8 NOT NULL,
            TP INT NOT NULL,
            FN INT NOT NULL,
            FP INT NOT NULL,
            TN INT NOT NULL,
            AVG_TPC float8 NOT NULL,
            AVG_FNC float8 NOT NULL,
            AVG_FPC float8 NOT NULL,
            AVG_TNC float8 NOT NULL,
            sensitivity float8 NOT NULL,
            miss_rate float8 NOT NULL,
            specificity float8 NOT NULL,
            fallout float8 NOT NULL,
            precision float8 NOT NULL,
            NPV float8 NOT NULL,
            F1 float8 NOT NULL
        );
        """ 
    
def insert_row(table_name, args, task, parametrization, nepoch=0, timestamp=0, m=None):
    return f"""
    INSERT INTO {table_name}(
        timestamp,task,parametrization, nepoch, acc, loss, exec_time, TP, FN, FP, TN, AVG_TPC, AVG_FNC, AVG_FPC, AVG_TNC, sensitivity, miss_rate, specificity, fallout, precision, NPV, F1
    )
    VALUES(
         {timestamp},'{task}','{parametrization}',{nepoch},{m["acc"]},{m["loss"]},{m["exec_time"]},{m["TP"]}, {m["FN"]}, {m["FP"]}, {m["TN"]}, {m["AVG_TPC"]}, {m["AVG_FNC"]}, {m["AVG_FPC"]},{m["AVG_TNC"]}, {m["sensitivity"]}, {m["miss_rate"]}, {m["specificity"]}, {m["fallout"]}, {m["precision"]}, {m["NPV"]}, {m["F1"]}
    );
    """

def make_sure_table_exist(args, conn, cur, table_name):
    cur.execute("select exists(select * from information_schema.tables where table_name=%s)", (table_name,))
    table_exists = cur.fetchone()[0]
    if not table_exists:
        psql = create_table_sql(table_name,args)
        cur.execute(psql)
        conn.commit()