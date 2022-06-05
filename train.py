import numpy as np
import pandas as pd
from mf import MF
# import util

customer_article_matrix = pd.read_csv("customer_article_array.csv")
customer_article_array = pd.DataFrame(customer_article_matrix).to_numpy()
#print(customer_article_array.shape)
customer_article_array = customer_article_array.astype(int)
#print(np.sum(customer_article_array))
customer_article_array = customer_article_array


mf = MF(customer_article_array, K=3, alpha=0.01, beta=0.01, iterations=200)
training_process = mf.train()
#print(mf.full_matrix().shape)
output = np.argsort(1-mf.full_matrix(), axis=0)
#print(mf.user)
#print(mf.item)
#print(mf.full_matrix())1
#print(output)
output_12 = np.delete(output, np.s_[12::], axis=0)
output_12T = output_12.T
#print(output_12T)
## convert your array into a dataframe
#df = pd.DataFrame (output)
## save to xlsx file
#df.to_excel('my_excel_file.xlsx', index=False)
customer_id = pd.read_csv("out_customer.csv")
customer_id = pd.DataFrame(customer_id)
result = pd.concat([customer_id, pd.DataFrame(output_12T)], axis=1)
article_id = pd.read_csv("out_article.csv").squeeze("columns")
result["article_id"] = result.iloc[:, 1::].applymap(lambda index: article_id[index]).values.tolist()
result = result[["customer_id", "article_id"]]
result.to_csv("result.csv", sep=",", index=False, header=["customer_id", "article_id"])

# Show results
show = pd.read_csv("result_train.csv", index_col="customer_id")
print(show)