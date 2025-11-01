import pandas as pd
import numpy as np

df = pd.DataFrame({
    "A": [1,2,np.nan,4],
    "B": [5,np.nan,np.nan,8],
    "C": [9,10,11,12]
})

print(df.isnull())

# Eliminar filas en las que haya nulos / Remove the rows that contain nulls

# df_del = df.dropna()

# Eliminar columnas en las que haya nulos / Remove the columns that contain nulls
# df_del = df.dropna(axis=1)

# Rellena los nulos con 0 / Fill the null with 0
df_relleno = df.fillna(value=0)
print(df_relleno)

# Rellena los nulos con el valor anterior de la columna / Fill the nulls with the previous value in the column
df_anterior = df.fillna(method="ffill")
print(df_anterior)

# Rellena los nulos con el siguiente valor en la columna / Fill the nulls with the next value in the column
df_siguiente = df.fillna(method="bfill")
print(df_siguiente)

# Rellena los nulos con la media de los valores de la columna / Fill the nulls with the mean of the column values
df_media = df.fillna(df.mean())
print(df_media)

# Rellena los nulos interponlando, estimando el valor desconocido con los conocidos / Estimate the unknwon value using the known values
df_inter = df.interpolate()
print(df_inter)

