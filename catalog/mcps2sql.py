import numpy as np
import sqlite3 as lite

dbname = '/Users/bjohnson/DATA/magellanic/catalogs/LMC/mcps/mcps_lmc.db'
catalog_name = '/Users/bjohnson/DATA/magellanic/catalogs/LMC/mcps/table1.dat'

con = lite.connect(dbname)

with con:
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS mcps")
    cur.execute("CREATE TABLE mcps(Id INTEGER PRIMARY KEY, RAh REAL, Dec REAL, bessell_U REAL, bessell_U_unc REAL, bessell_B REAL, bessell_B_unc REAL, bessell_V REAL, bessell_V_unc REAL, bessell_I REAL, bessell_I_unc REAL, flag INT)")

    f = open(catalog_name, 'r')
    for i,row in enumerate(f):
        row  = [float(r) for r in row.split()]
        row[-1] = int(row[-1])
        row = tuple([i] + row)
        cur.execute("INSERT INTO mcps VALUES(?"+",?"*(len(row)-1) + ")", row)
    #    for row in f:
    f.close()

    
