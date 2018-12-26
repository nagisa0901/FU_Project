
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt

fname = input()
f = open("./midi/"+fname + ".txt", "r")
data1 = f.read()
f.close()
lines1 = data1.split('\n') # 改行で区切る(改行文字そのものは戻り値のデータには含まれない)
oto = []

ave = 0
a = 0
for line in lines1:
    if(a==0):
        max = int(line)
        min = int(line)
    a += 1
    ave += int(line)
    if(max<int(line)):
        max = int(line)
    if(min>int(line)):
        min = int(line)
ave /= a
b = 1
while(abs(max-min)>40):
    max /= 2
    min /= 2
    b *= 2
print(b)

for line in lines1:
    oto.append((int(line)-ave)/b)
    
group_by = 1
Xa = [oto[i:i + group_by] for i in range(0, len(oto), group_by)]
Xl = np.array_split(Xa,30)
print(Xl)


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


input_layer = 1
hidden_layer = 50
output_layer = 1
input_hidden_weight = np.random.randn(hidden_layer,input_layer)  
hidden_hidden_weight = np.random.randn(hidden_layer,hidden_layer)
hidden_output_weight = np.random.randn(output_layer,hidden_layer)

epsilon = 0.01
epoch = 2000
loss = np.zeros(epoch)
f2 = open(fname + "_loss.csv","w")
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

            loss[epo] += 0.5 * (ys[t] - X[t + 1]).dot((ys[t] - X[t + 1]).reshape((-1, 1)))
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
plt.savefig(fname + "_loss.png")
plt.show()


# In[5]:


zs,ys,ps,qs = forward_seq(Xl[0][:3])
print(Xl[0][:3])
print(ys)
zs = list(zs)
ys = list(ys)
#print("ys = \n" + str(y_after) + "\n\nzs = \n" + str(z_after))
print(ys)
y, z = ys[-1], zs[-1]
p,q =([],[])
for t in range(len(oto)-3):
    z, y, p, q = forward(y, z, p, q)
    zs.append(list(z))
    ys.append(list(y))
print(ys)


# In[6]:


fys=open(fname + "_output.txt","w")
ysi=0
for yss in ys:
    ys_list = [int(round(i*b+ave)) for i in yss]
    for w in ys_list:
        if(w != "[" or w != "]"):
            ysi+=1
            fys.write(str(w))
            if(ysi<len(ys)):
                fys.write("\n")
                
fys.close()


# In[7]:


#plt.plot(oto[:25])
plt.plot(ys,linestyle="solid")
plt.plot(Xa, linestyle='dotted',color="Blue")
plt.plot(ys[:len(Xl[0][:3])],linestyle="dashed",color="Red")
plt.ylabel("notenumber * 0.01")
plt.xlabel("Number")
plt.savefig(fname + "_compare.png")
plt.show()


# In[ ]:





# In[ ]:




