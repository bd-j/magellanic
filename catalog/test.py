import sqlite3 as lite
import numpy as np

dbname = 'mcps_lmc.db'
con = lite.connect(dbname)
s2p = {'INT': '<i8', 'REAL': '<f8', 'INTEGER': '<i8'}
sc = ("WHERE bessell_V < 20 AND bessell_V > 0 " +
      "AND RAh > 5.5 AND RAh < 5.7 " +
      "AND Dec < -66.5 AND Dec > -67.5 " +
    "AND bessell_U > 0 AND bessell_B > 0 and bessell_I > 0")
with con:
    cur = con.cursor()
    cur.execute("SELECT * FROM mcps "+sc)
    data = cur.fetchall()
    cur.execute('PRAGMA table_info(mcps)')
    info = cur.fetchall()

names = []
formats = []
for i in info:
    names.append(i[1])
    formats.append(s2p[i[2]])
dt = dict(names = names, formats = formats)
cat = np.array(data, dtype = dt)
