import numpy as np
import joblib
import pandas as pd

def leastsquares_fit(p, a, b):
    # form a based on p
    x = np.hstack([a ** i for i in range(p+1)])
    
    w_opt = np.linalg.inv(x.T@x) @ x.T @b
    
    return w_opt

df = pd.read_csv('data_daily.csv')
df['# Date'] = pd.to_datetime(df['# Date'])
df.set_index('# Date', inplace=True)

dates_ordinal = np.array([d.toordinal() for d in df.index])
y = df['Receipt_Count'].to_numpy()
X = dates_ordinal.reshape(-1, 1)

w_opt_LS = leastsquares_fit(1, X, y)

# Save the model
joblib.dump(w_opt_LS, 'model.joblib')