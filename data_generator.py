import pandas as pd
import numpy as np

np.random.seed(42)
n_samples=2000

data={
    'Age': np.random.randint(22, 60, n_samples),
    'YearsAtCompany': np.random.randint(1, 20, n_samples),
    'MonthlyIncome': np.random.randint(3000, 15000, n_samples),
    'OverTime': np.random.choice(['Yes', 'No'], n_samples),
    'Attrition': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
}

df=pd.DataFrame(data)

condition_leave = (df['MonthlyIncome'] < 5000) & (df['OverTime'] == 'Yes')
df.loc[condition_leave, 'Attrition'] = 1

# 3. Add a secondary pattern (Young people with low pay)
condition_young_leave = (df['Age'] < 26) & (df['MonthlyIncome'] < 3500)
df.loc[condition_young_leave, 'Attrition'] = 1

df.to_csv('employee_data.csv',index=False)
print("✅ employee_data.csv created successfully!")