import numpy as np
import pandas as pd
import openpyxl
class MF():

    def __init__(self, R, K, alpha, beta, iterations):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):

        self.user = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.item = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        #self.b_u = np.zeros(self.num_users)
        #self.b_i = np.zeros(self.num_items)
        #self.b = np.mean(self.R[np.where(self.R != 0)])


        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        training_process_rmse=[]
        training_process_mse = []
        training_process_mae = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse = self.rmse()/10
            mae = self.mae()
            mse = self.mse()
            training_process_rmse.append(rmse)
            training_process_mae.append(mae)
            training_process_mse.append(mse)
            if (i+1) % 10 == 0:
                print("Iteration: %d ; rmse_error = %.4f" % (i+1, rmse))
                print("Iteration: %d ; mae_error = %.4f" % (i + 1, mae))
                print("Iteration: %d ; mse_error = %.4f" % (i + 1, mse))
        #col1 = "rmse"
        #col2 = "mae"
        #col3 = "mse"
        #data1 = pd.DataFrame({col1:training_process_rmse,col2:training_process_mae,col3:training_process_mse})
        #data1.to_excel('sample_data1.xlsx', sheet_name='k = 2', index=False)
        return [training_process_mse,training_process_mae,training_process_rmse]

    def rmse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def mse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        a = xs.size
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return error/a

    def mae(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        a = xs.size
        for x, y in zip(xs, ys):
            error += np.absolute(self.R[x, y] - predicted[x, y])
        return error/a


    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            #self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            #self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            P_i = self.user[i, :][:]

            self.user[i, :] += self.alpha * (2* e * self.item[j, :] - 2*self.beta * self.user[i,:])
            self.item[j, :] += self.alpha * (2* e * P_i - self.beta * self.item[j,:])

    def get_rating(self, i, j):

        prediction = self.item[j, :].dot(self.user[i, :].T)
        return prediction

    def full_matrix(self):
        return self.user.dot(self.item.T)
    """

    def __init__(self, R, K, alpha, beta, iterations):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):

        self.user = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K))
        self.item = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K))

        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        training_process_rmse = []
        training_process_mse = []
        training_process_mae = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse = self.rmse() / 10
            mae = self.mae()
            mse = self.mse()
            training_process_rmse.append(rmse)
            training_process_mae.append(mae)
            training_process_mse.append(mse)
            if (i + 1) % 10 == 0:
                print("Iteration: %d ; rmse_error = %.4f" % (i + 1, rmse))
                print("Iteration: %d ; mae_error = %.4f" % (i + 1, mae))
                print("Iteration: %d ; mse_error = %.4f" % (i + 1, mse))
        # col1 = "rmse"
        # col2 = "mae"
        # col3 = "mse"
        # data1 = pd.DataFrame({col1:training_process_rmse,col2:training_process_mae,col3:training_process_mse})
        # data1.to_excel('sample_data1.xlsx', sheet_name='k = 2', index=False)
        return [training_process_mse, training_process_mae, training_process_rmse]

    def rmse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def mse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        a = xs.size
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return error / a

    def mae(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        a = xs.size
        for x, y in zip(xs, ys):
            error += np.absolute(self.R[x, y] - predicted[x, y])
        return error / a

    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            P_i = self.user[i, :][:]

            self.user[i, :] += self.alpha * (e * self.user[i, :] - self.beta * self.item[j, :])
            self.item[j, :] += self.alpha * (e * P_i - self.beta * self.user[i, :])

    def get_rating(self, i, j):

        prediction = self.b + self.b_u[i] + self.b_i[j] + self.item[j, :].dot(self.user[i, :].T)
        return prediction

    def full_matrix(self):
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.user.dot(self.item.T)
    """