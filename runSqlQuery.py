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


def run_sql_query(qry, profile='MYMSSQL', os='mac', verbose=True, database='AppNexus'):

    # Database is only required for connecting on windows.
    # I'm not sure the windows implementation works or is the best approach.
    # In fact I'm pretty sure it won't work well on windows anymore.
    # profile is the name of the profile in freetds.conf

    sql_info = pd.read_csv('~/.aws/sql-info.csv')

    if profile not in sql_info.prof_name.values:
        raise ValueError('UID and PWD for {} not found'.format(profile))

    profile_info = sql_info[sql_info.prof_name == profile]

    if os == 'mac':
        connstring = 'DSN={pro};UID={uid};PWD={pwd}' \
            .format(pro=profile,
                    uid=profile_info.UID.iloc[0],
                    pwd=profile_info.PWD.iloc[0])
        conn = pyodbc.connect(connstring)

    if os == 'win':
        server = ','.join([profile_info.address.iloc[0], str(profile_info.port.iloc[0])])

        connstring = (r'DRIVER={ODBC Driver 13 for SQL Server};'
            r'SERVER={ser};'
            r'DATABASE={dat}};'
            r'UID={uid};'
            r'PWD={pwd};') \
                .format(ser=server,
                        dat=database,
                        uid=profile_info.UID.iloc[0],
                        pwd=profile_info.PWD.iloc[0])

    df = pd.read_sql(qry, conn)
    conn.close()
    if verbose:
        print('shape of data = ', df.shape)
        print('column names:\n', list(df.columns))
    return df
