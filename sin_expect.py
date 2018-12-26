
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
'''
sin波
'''
f = []
T = 1000
x = np.sin(np.arange(0, 2 * T + 1))
noise = 0.05 * np.random.uniform(low=-1.0, high=1.0, size=len(x))
f = [i + j for (i, j) in zip(x, noise)]

length_of_sequences = 2 * T
maxlen = 25  # ひとつの時系列データの長さ



li = list(f)
group_by = 1
Xl = [li[i:i + group_by] for i in range(0, len(li), group_by)]
Xl = np.array_split(Xl,250)
print(Xl)
input_layer = 1
hidden_layer = 10
output_layer = 1
input_hidden_weight = np.random.randn(hidden_layer,input_layer)  
hidden_hidden_weight = np.random.randn(hidden_layer,hidden_layer)
hidden_output_weight = np.random.randn(output_layer,hidden_layer)
#print(str(input_hidden_weight)+"\n"+str(hidden_hidden_weight)+"\n"+str(hidden_output_weight))


# In[2]:


def sigmoid(arr):
    return 1.0 / (1.0 + np.exp(-arr))

def tanh(arr):
    return np.tanh(arr)

def forward(x,z,p,q):
    r = hidden_hidden_weight.dot(z)
    z = input_hidden_weight.dot(x) + r
    p.append(z)
    z = sigmoid(z)
    #print(z)
    y = hidden_output_weight.dot(z)
    q.append(y)
    #y = sigmoid(y)
    return z, y, p, q

def forward_seq(X):
    z = np.zeros(hidden_layer)
    zs, ys = ([], [])
    p,q = ([], [])
    for x in X:
        z,y,p,q = forward(x,z,p,q)
        zs.append(z)
        ys.append(y)
    return zs, ys, p, q


# In[3]:


epsilon = 0.05
epoch = 100
loss = np.zeros(epoch)
f2 = open("loss.csv","w")
### train start ###
for epo in range(epoch):
    for X in np.random.permutation(Xl):
        zs,ys,p,q = forward_seq(X)
        #print("zs = \n" + str(zs) + "\n")
        #print("ys = \n" + str(ys) + "\n")
        hidden_delta = np.zeros(hidden_layer)
        hidden_dEdw = np.zeros(input_hidden_weight.shape)
        recurr_dEdw = np.zeros(hidden_hidden_weight.shape)
        output_dEdw = np.zeros(hidden_output_weight.shape)
        tau = X.shape[0]    
        for t in range(tau - 1)[::-1]:
            #fx = sigmoid(q[t])
            #print("t = " + str(t))
            #output_delta =  (1 / (np.cosh(q[t]) ** 2)) * (ys[t] - X[t+1, :]) 
            output_delta = (ys[t] - X[t+1,:]) 
            output_dEdw += output_delta.reshape(-1, 1) * zs[t]
            #print(output_dEdw)
            #print()

            fx = sigmoid(p[t])
            hidden_delta = (1 - fx) * fx * ((output_delta.dot(hidden_output_weight)) + hidden_delta.dot(hidden_hidden_weight))
            #hidden_delta = (1 / (np.cosh(p[t]) ** 2)) * ((output_delta.dot(hidden_output_weight)) + hidden_delta.dot(hidden_hidden_weight))
            hidden_dEdw += hidden_delta.reshape(-1, 1) * X[t,:]
            #print(hidden_dEdw)
            #print()

            zs_prev = zs[t - 1] if t > 0 else np.zeros(hidden_layer)
            recurr_dEdw += hidden_delta.reshape(-1, 1) * zs_prev
            #print(recurr_dEdw)
            #print()

            loss[epo] += 0.5 * (ys[t] - X[t + 1]).dot((ys[t] - X[t + 1]).reshape((-1, 1))) / (tau - 1)
            #print(loss)
            #print()
        
        hidden_output_weight -= epsilon * output_dEdw 
        input_hidden_weight -= epsilon * hidden_dEdw
        hidden_hidden_weight -= epsilon * recurr_dEdw
    f2.write(str(loss[epo]) + "\n")
    print("epoch: {" + str(epo) + "}:\t" + str(loss[epo]))
f2.close()


# In[4]:


plt.plot(np.arange(loss.size), loss)
plt.ylabel("E")
plt.xlabel("epoch")
plt.savefig("loss_sin.png")
plt.show()


# In[5]:


zs,ys,ps,qs = forward_seq(Xl[0][:2])
print(Xl[0][:2])
print(ys)
#print("ys = \n" + str(y_after) + "\n\nzs = \n" + str(z_after))
#print()
y, z = ys[-1], zs[-1]
p,q =([],[])
for t in range(100):
    z, y, p, q = forward(y, z, p, q)
    zs.append(z)
    ys.append(y)
print(ys)


# In[6]:


y_input = np.array(ys)[:100]
x = np.sin(np.arange(0, 100))

x = np.array(x)[:]
plt.plot(x, linestyle='dotted',color="Blue")
plt.ylim([-1.5, 1.5])
plt.plot(y_input,linestyle="solid")
plt.plot(y_input[:len(Xl[0][:2])],linestyle="dashed",color="Red")
plt.xlabel("t")
plt.ylabel("sin(t)")
plt.savefig("sin.png")
plt.show()


# In[ ]:




