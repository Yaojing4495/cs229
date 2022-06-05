import numpy as np
import pandas as pd
from mf import MF
import matplotlib.pyplot as plt

# import util

customer_article_matrix = pd.read_csv("customer_article_array.csv")
customer_article_array = pd.DataFrame(customer_article_matrix).to_numpy()
#print(customer_article_array.shape)
customer_article_array = customer_article_array.astype(int)
#print(np.sum(customer_article_array))
customer_article_array = customer_article_array

k = [2,3,4,5]
mse = []
mae = []
rmse = []

for K in k:
    mf = MF(customer_article_array, K, alpha=0.01, beta=0.01, iterations=200)
    training_process = mf.train()
    mse.append(training_process[0])
    mae.append(training_process[1])
    rmse.append(training_process[2])
col1 = "K = 2"
col2 = "K = 3"
col3 = "K = 4"
col4 = "K = 5"
col5 = "iterations"
x=[]
for n in range(200):
    x.append(n+1)
data_mse = pd.DataFrame({col1:mse[0],col2:mse[1],col3:mse[2],col4:mse[3],col5:x})
data_mse.to_excel('mse.xlsx', sheet_name='1', index=False)
data_mae = pd.DataFrame({col1:mse[0],col2:mae[1],col3:mae[2],col4:mae[3],col5:x})
data_mae.to_excel('mae.xlsx', sheet_name='1', index=False)
data_rmse = pd.DataFrame({col1:rmse[0],col2:rmse[1],col3:rmse[2],col4:rmse[3],col5:x})
data_rmse.to_excel('rmse.xlsx', sheet_name='1', index=False)
"""

mf = MF(customer_article_array, 5, alpha=0.01, beta=0.01, iterations=150)
training_process = mf.train()
output = np.argsort(1-mf.full_matrix(), axis=0)
output_12 = np.delete(output, np.s_[12::], axis=0)
output_12T = output_12.T
customer_id = pd.read_csv("out_customer.csv")
customer_id = pd.DataFrame(customer_id)
result = pd.concat([customer_id, pd.DataFrame(output_12T)], axis=1)
article_id = pd.read_csv("out_article.csv").squeeze("columns")
result["article_id"] = result.iloc[:, 1::].applymap(lambda index: article_id[index]).values.tolist()
result = result[["customer_id", "article_id"]]
result.to_csv("result_svd.csv", sep=",", index=False, header=["customer_id", "article_id"])

# Show results
show = pd.read_csv("result_svd.csv", index_col="customer_id")
print(show)
"""



