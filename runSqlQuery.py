#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 09:12:11 2017

@author: keith.landry

This function connects to the Genius Sports SQL server.
You must be connected through the VPN. The server address 
and info are contained in the files freetds.conf, odbc.ini,
and odbcinst.ini. The first attribute in the string used in
the connect function is the name given to the desired server 
in these files. (name will be in square brackets)

See https://github.com/mkleehammer/pyodbc/wiki/Connecting-to-SQL-Server-from-Mac-OSX
for more details on connecting to SQL server with pyodbc on mac

"""
import pyodbc
import pandas as pd

def run_sql_query(qry):
    conn = pyodbc.connect('DSN=MYMSSQL;UID=reader;PWD=Dj1bout1')
    df = pd.read_sql(qry, conn)
    conn.close()
    print('shape of data = ', df.shape)
    print('column names:\n', list(df.columns))
    return df