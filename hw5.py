#!/usr/bin/env python
# coding: utf-8

# In[223]:


import numpy as np
import matplotlib.pyplot as plt


# In[224]:


monona = np.loadtxt("monona.txt")
mendota = np.loadtxt("mendota.txt")


# In[225]:


# Q1.1
plt.plot(monona[:,0], monona[:,1], label="monona")
plt.plot(mendota[:,0], mendota[:,1], label="mendota")
plt.legend()
plt.xlabel("year")
plt.ylabel("ice days")


# In[226]:


# Q1.1 part 2

diff = np.empty((0,2))
for year in range(1850,2020):
    days = 9999
    for row in monona:
        if row[0] == year:
            for row2 in mendota:
                if row2[0] == year:
                    days = row[1] - row2[1]
    if days != 9999:
        diff = np.append(diff, [[year, days]], axis=0)

plt.plot(diff[:,0], diff[:,1], label="monona - mendota")
plt.legend()
plt.xlabel("year")
plt.ylabel("ice days")


# In[227]:


# Q1.2

monona_train = np.empty((0,2))
monona_test = np.empty((0,2))

for row in monona:
    if row[0] <= 1970:
        monona_train = np.append(monona_train, [[row[0], row[1]]], axis=0)
    else:
        monona_test = np.append(monona_test, [[row[0], row[1]]], axis=0)

mendota_train = np.empty((0,2))
mendota_test = np.empty((0,2))

for row in mendota:
    if row[0] <= 1970:
        mendota_train = np.append(mendota_train, [[row[0], row[1]]], axis=0)
    else:
        mendota_test = np.append(mendota_test, [[row[0], row[1]]], axis=0)

print("monona mean:", np.mean(monona_train[:,1]))
print("monona std:", np.std(monona_train[:,1]))

print("mendota mean:", np.mean(mendota_train[:,1]))
print("mendota std:", np.std(mendota_train[:,1]))


# In[228]:


# restrict to years with data for both lakes

monona_train_ov = np.empty((0,2))
monona_test_ov = np.empty((0,2))
mendota_train_ov = np.empty((0,2))
mendota_test_ov = np.empty((0,2))

for year in range(1850,2020):
    for row1 in monona:
        if row1[0] == year:
            for row2 in mendota:
                if row2[0] == year:
                    if year <= 1970:
                        monona_train_ov = np.append(monona_train_ov, [[row1[0], row1[1]]], axis=0)
                        mendota_train_ov = np.append(mendota_train_ov, [[row2[0], row2[1]]], axis=0)
                    else:
                        monona_test_ov = np.append(monona_test_ov, [[row1[0], row1[1]]], axis=0)
                        mendota_test_ov = np.append(mendota_test_ov, [[row2[0], row2[1]]], axis=0)


# In[229]:


# Q1.3
# w = (XTX)^-1 XTy

X = np.empty((0,3))

row_num = 0
for row in monona_train_ov:
    X = np.append(X, [[1, mendota_train_ov[row_num, 0], row[1]]], axis=0)
    row_num += 1

beta = np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(X), X)), np.matmul(np.matrix.transpose(X), mendota_train_ov[:,1]))

print(beta)


# In[230]:


# Q1.4

mendota_est = np.empty((0,1))
for row1 in mendota_train_ov:
    year = row1[0]
    for row2 in monona_train_ov:
        if row2[0] == year:
            y = row2[1]
    mendota_est = np.append(mendota_est, [[beta[0] + (beta[1]*year) + (beta[2]*y)]], axis=0)

# MSE
MSE = 0
row_num = 0
for y in mendota_est:
    MSE += ((y.item() - mendota_train_ov[row_num, 1])**2)
    row_num += 1
MSE = MSE/(row_num+1)

print("MSE:", MSE)

# R squared
num = 0
denom = 0
row_num = 0
mendota_mean = np.mean(mendota_train[:,1])
for row in mendota_train_ov:
    num += ((row[1] - mendota_est[row_num])**2)
    denom += ((row[1] - mendota_mean)**2)
    row_num += 1
Rsq = 1 - (num/denom)
Rsq = Rsq.item()

print("R squared", Rsq)


# In[231]:


def get_grad(mendota, monona, chain, beta, mse=0):
    sum = 0
    row_num = 0
    for row1 in mendota:
        year = row1[0]
        y = monona[row_num,1].item()
        if chain == 1:
            chain_term = year
        if chain == 2:
            chain_term = y
        else:
            chain_term = 1
        if mse == 1:
            sum += (beta[0] + (beta[1]*year) + (beta[2]*y) - row1[1])**2
        else:
            sum += (beta[0] + (beta[1]*year) + (beta[2]*y) - row1[1]) * chain_term
        row_num += 1
    
    if (mse == 1):
        return sum/row_num
    else:
        return sum*2/row_num


# In[235]:


# Q1.6

beta_grad = [0,0,0]
beta_new = [0,0,0]
nu = .1

print("beta", beta_grad)
print("MSE", get_grad(mendota_train_ov, monona_train_ov, 0, beta_grad, 1))

for attempt in range (1,1000):
    beta_new[0] = -1 * nu * get_grad(mendota_train_ov, monona_train_ov, 0, beta_grad)
    beta_new[1] = -1 * nu * get_grad(mendota_train_ov, monona_train_ov, 1, beta_grad)
    beta_new[2] = -1 * nu * get_grad(mendota_train_ov, monona_train_ov, 2, beta_grad)
    
    beta_grad[0] += beta_new[0]
    beta_grad[1] += beta_new[1]
    beta_grad[2] += beta_new[2]
    
    print("attempt", attempt)
    print("beta", beta_grad)
    print("MSE", get_grad(mendota_train_ov, monona_train_ov, 0, beta_grad, 1))


# In[236]:


# BE CAREFUL THIS NORMALIZES THE INPUT DATA TOO
def normalize(data,noy):
    max0 = -9999999
    max1 = -9999999
    min0 = 9999999
    min1 = 9999999
    for row in data:
        if row[0] > max0:
            max0 = row[0]
        if row[0] < min0:
            min0 = row[0]
        if row[1] > max1:
            max1 = row[1]
        if row[1] < min1:
            min1 = row[1]
    row_num = 0
    norm_data = data
    for row in data:
        norm_data[row_num,0] = (row[0]-min0)/(max0-min0)
        if noy != 1:
            norm_data[row_num,1] = (row[1]-min1)/(max1-min1)
        row_num += 1
    return norm_data


# In[245]:


# Q1.7

norm_monona = normalize(monona_train_ov, 0)
norm_mendota = normalize(mendota_train_ov, 1)

beta_grad = [0,0,0]
beta_new = [0,0,0]
nu = 1

print("beta", beta_grad)
print("MSE", get_grad(norm_mendota, norm_monona, 0, beta_grad, 1))

for attempt in range (1,1000):
    beta_new[0] = -1 * nu * get_grad(norm_mendota, norm_monona, 0, beta_grad)
    beta_new[1] = -1 * nu * get_grad(norm_mendota, norm_monona, 1, beta_grad)
    beta_new[2] = -1 * nu * get_grad(norm_mendota, norm_monona, 2, beta_grad)
    
    beta_grad[0] += beta_new[0]
    beta_grad[1] += beta_new[1]
    beta_grad[2] += beta_new[2]
    
    print("attempt", attempt)
    print("beta", beta_grad)
    print("MSE", get_grad(norm_mendota, norm_monona, 0, beta_grad, 1))


# In[133]:


# Q1.8
# w = (XTX)^-1 XTy

X = np.empty((0,2))

row_num = 0
for row in mendota_train_ov:
    X = np.append(X, [[1, row[0]]], axis=0)

beta = np.matmul(np.linalg.inv(np.matmul(np.matrix.transpose(X), X)), np.matmul(np.matrix.transpose(X), mendota_train_ov[:,1]))

print(beta)


# In[138]:


# Q1.9
# w = (XTX-2A)^-1 XTy

X = np.empty((0,3))
neg2A = [[0,0,0],[0,-2,0],[0,0,-2]]

row_num = 0
for row in monona_train_ov:
    X = np.append(X, [[1, mendota_train_ov[row_num, 0], row[1]]], axis=0)
    row_num += 1

beta = np.matmul(np.linalg.inv(np.add(np.matmul(np.matrix.transpose(X), X),neg2A)), np.matmul(np.matrix.transpose(X), mendota_train_ov[:,1]))

print(beta)


# In[199]:


print(mendota_train_ov)


# In[ ]:




