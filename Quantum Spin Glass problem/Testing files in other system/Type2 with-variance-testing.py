#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pennylane as qml
from matplotlib import pyplot as plt
import numpy as np
import scipy
import networkx as nx
import copy
import seaborn
import time


# In[2]:


qubit_number = 6
qubits = range(qubit_number)


# In[3]:


G=nx.complete_graph(qubit_number)


# In[4]:


for node in range(qubit_number):
    G.add_node(node)
    
G.nodes()


# In[5]:


for node in range(qubit_number):
    if node<qubit_number-1: 
        G.add_edge(node,node+1)
len(G.edges)


# In[6]:


nx.draw(G)


# In[7]:


def H_matrix(graph,edge_weights,no_of_qubits):
    # edge_weights= $\theta_1$ parameters for interaction terms
    # node_weights= $\theta_2$ parameters for single qubit terms
    
    full_matrix=np.zeros((2**no_of_qubits,2**no_of_qubits))
    
    # creating the two qubit interation terms:
    for i,edge in enumerate(graph.edges):
        
        zz_int=1
        
        for qubit in range(no_of_qubits):
            
            if qubit in edge:
                # for only ZZ interaction 
                zz_int=np.kron(zz_int,qml.PauliZ.matrix)
                
            else:
                zz_int=np.kron(zz_int,np.identity(2))
                
        full_matrix+=edge_weights[i]*zz_int*(-1)
        
        
    for i,edge in enumerate(graph.edges):
        
        xx_int=1
        
        for qubit in range(no_of_qubits):
            
            if qubit in edge:
                # for only XX interaction 
                xx_int=np.kron(xx_int,qml.PauliX.matrix)
                
            else:
                xx_int=np.kron(xx_int,np.identity(2))
                
        full_matrix+=edge_weights[i]*xx_int*(-1)
        
       
    for i,edge in enumerate(graph.edges):
        
        yy_int=1
        
        for qubit in range(no_of_qubits):
            
            if qubit in edge:
                # for only YY interaction 
                yy_int=np.kron(yy_int,qml.PauliY.matrix)
                
            else:
                yy_int=np.kron(yy_int,np.identity(2))
                
        full_matrix = full_matrix + edge_weights[i]*yy_int*(-1)
    
    return full_matrix


# In[8]:


# generating gaussian numbers from box muller transformation

def generategaussian(n,var,mean):
    arr=[]
    
    u1=np.random.rand(n)
    u2=np.random.rand(n)
    arr.append(np.sqrt(var)*np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)+mean)
    arr.append(np.sqrt(var)*np.sqrt(-2*np.log(u1))*np.sin(2*np.pi*u2)+mean)
        
    return arr


# In[9]:


Gaussians=generategaussian(10000,1,0)


# In[10]:


from pylab import show,hist,subplot,figure


# In[11]:


figure()
hist(Gaussians)


# Defining the three variables except the interaction terms in the below cell

# In[36]:


'''
Only the number of "runs" in the below cell should vary  
'''


# In[12]:


run=100 # the number of times you have to repeat the experiment
# no transverse field in this model
mean=1  # mean value of the gaussian interactions Jij
var=np.linspace(0.1,2,10)  # variance of the gaussian interactions Jij
var


# In[13]:


min_energy=[]
gaussian_arr=[]


# In[14]:


start_time = time.time() 
for v in var:
    for i in range(run):
        edge_weights =generategaussian(len(G.edges),v,mean)[0]
        gaussian_arr.append(edge_weights)
        Ham=H_matrix(G,edge_weights,qubit_number)
        min_energy.append(np.real_if_close(min(np.linalg.eig(Ham)[0])))
print("Total elements",len(min_energy))
end_time = time.time()
print('Total time taken: ', (end_time-start_time)/60, ' mins.')


# In[15]:


np.sum(min_energy[50:100])/len(min_energy[50:100])


# In[16]:


min_eng=[]
for j in range(len(var)):
    min_eng.append(np.sum(min_energy[run*j:run*(j+1)])/len(min_energy[run*j:run*(j+1)]))

(min_eng)


# In[17]:


plt.style.use("seaborn")
#x=np.linspace(1,100,100)
plt.plot(var,min_eng,"turquoise",label="Ground state energies using exact diagonalization")
plt.ylabel("Ground state energies", fontsize=18)
plt.xlabel("Variance", fontsize=18)
plt.tick_params(axis="both", colors='black',which="major", labelsize=16)   # helps to increase the size of the values in X and Y axis
plt.tick_params(axis="both", colors='black', which="minor", labelsize=16)
#plt.ylim(-7,-3)
plt.legend()
plt.show()


# In[18]:


#####################################################


# Try couple of times to make the above plot as much as smooth, if that doesn't work try to increase the "run by 20" 

# In[19]:


#####################################################


# In[20]:


print(f"Minimum average eigen energies:{min_eng}")


# In[21]:


dev = qml.device("lightning.qubit", wires= qubit_number) # 16 register qubits 


# In[22]:


wires=list(range(qubit_number)) # will be used below
print("length of wires",len(wires))
for i in wires:
    print("wire->",i)
print("length of edges",len(G.edges))


# In[23]:


'''
For a complete graph the below ansatz has to be defined separately
'''


# In[24]:


# Hardware efficient ansatz from paper "Certified variational quantum algorithms for eigenstate preparation"

def ansatz(l):
    
    if len(l)!=Rx_Rz_layers*qubit_number+qubit_number+1:
        raise ValueError("Number of parameters are not correct")  # or Rx_Rz_layers*len(G.nodes)+len(G.edges)+1 "+1" because the 
                                                                  # controlled Ry are operated in cyclic way 
    # single qubit gates Rx and Ry layers..............
    
    for j in range(Rx_Rz_layers):
        if j/2==0:
            for i in range(qubit_number):
                qml.RX(l[i+qubit_number*j],wires=i)
            
        
        if j/2!=0:
            for i in range(qubit_number):
                qml.RZ(l[i+qubit_number*j],wires=i)
        
    l=l[len(G.nodes)*Rx_Rz_layers:]
    
   # end...............................................

   # Two qubit controlled Ry...........................
    for w in wires:
        if w<qubit_number-1:
            qml.CRY(l[w],wires=[w,w+1])    
        else:
            qml.CRY(l[w],wires=[w,0])
            
   # end...............................................


# In[25]:


coeffs_toy = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
#qml.PauliZ(15)@qml.PauliZ(0),
obs_toy = [
    qml.PauliZ(0)@qml.PauliZ(1), qml.PauliZ(1)@qml.PauliZ(2), qml.PauliZ(2)@qml.PauliZ(3), qml.PauliZ(3)@qml.PauliZ(4),
     qml.PauliZ(4)@qml.PauliZ(5),qml.PauliZ(5)@qml.PauliZ(6),qml.PauliZ(6)@qml.PauliZ(7),qml.PauliZ(7)@qml.PauliZ(8),
    qml.PauliZ(8)@qml.PauliZ(9),qml.PauliZ(9)@qml.PauliZ(10),qml.PauliZ(10)@qml.PauliZ(11),qml.PauliZ(11)@qml.PauliZ(12),
    qml.PauliZ(12)@qml.PauliZ(13),qml.PauliZ(13)@qml.PauliZ(14),qml.PauliZ(14)@qml.PauliZ(15),
    qml.PauliX(0),qml.PauliX(1),qml.PauliX(2),qml.PauliX(3),qml.PauliX(4),qml.PauliX(5),qml.PauliX(6),qml.PauliX(7)
    ,qml.PauliX(8),qml.PauliX(9),qml.PauliX(10),qml.PauliX(11),qml.PauliX(12),qml.PauliX(13),qml.PauliX(14),qml.PauliX(15)

]
H_toy = qml.Hamiltonian(coeffs_toy, obs_toy)
print(H_toy)


# In[26]:


gaussian_arr[59]


# In[27]:


coeffs_arr=[]
for v1,v in enumerate(var):
    for i in range(run):
        coeffs =np.concatenate([-gaussian_arr[i+run*v1] for k in range(3)]) # [-1,-1,-1,-1,-1,-g,-g,-g,-g,-g,-g]
        coeffs_arr.append(coeffs)
(coeffs_arr)[59]


# The printed values in the above cell will be -$J_{ij}$. As in the hamiltonian below there is a -ve sign before these coefficients

# In[28]:


obs=[]
for i in range(qubit_number):
    for j in range(qubit_number):
        if i<j:
            obs.append(qml.PauliZ(i)@qml.PauliZ(j))

print(obs)


# In[29]:


H=[]

obs=[]
for i in range(qubit_number):
    for j in range(qubit_number):
        if i<j:
            obs.append(qml.PauliZ(i)@qml.PauliZ(j))

for i in range(qubit_number):
    for j in range(qubit_number):
        if i<j:
            obs.append(qml.PauliX(i)@qml.PauliX(j))

for i in range(qubit_number):
    for j in range(qubit_number):
        if i<j:
            obs.append(qml.PauliY(i)@qml.PauliY(j))
for v1 in range(len(var)):
    for i in range(run):
        H.append(qml.Hamiltonian(coeffs_arr[i+run*v1], obs))
print((H[59]))
print(len(H))


# In[30]:


G.edges()


# In[31]:


Trotter_steps=1 # Should remain 1


# In[32]:


Rx_Rz_layers=3 # Should remain 3


# In[33]:


params=np.random.rand(Rx_Rz_layers*qubit_number+qubit_number+1)*0.1


# In[34]:


#@qml.qnode(dev)
def full_HVA_ansatz(params, **kwargs):
    
    #for i in range(Trotter_steps):
    ansatz(params)
    
    #return qml.state()  


# In[35]:


start_time = time.time()    
cost_fn_arr=[]
for v in range(len(var)):
    for i in range(run):
        cost_fn=qml.ExpvalCost(full_HVA_ansatz,H[i+run*v],dev)
        cost_fn_arr.append(cost_fn)
    
len(cost_fn_arr)
end_time = time.time()
print('Time taken: ', (end_time-start_time)/60, ' mins.')


# In[94]:


#cost_fn=qml.ExpvalCost(full_HVA_2,H[0],dev)


# In[101]:


par_arr=[]
costarr=[]
opt_val_arr=[]


# In[102]:


start_time = time.time()    
for v1,v in enumerate(var):
    for i in range(run):
    #par_arr=[]
    #costarr=[]
        optimizer = qml.AdamOptimizer(stepsize=0.1)
        params=np.random.rand(Rx_Rz_layers*qubit_number+qubit_number+1)*0.1
        
        start_time2 = time.time()
        
        for j in range(0, 100+1):
            params, cost = optimizer.step_and_cost(cost_fn_arr[i+run*v1], params)

    # Prints the value of the cost function
            if j!=0 and j % 100 == 0:
                print(f"Cost at Step {j} of {i}th run for variance {v}: {cost}")
                
                end_time2 = time.time()
                print('Time taken at this step: ', (end_time2-start_time2)/60, ' mins.')
                
                costarr.append(cost)
                par_arr.append(params)
        opt_val_arr.append(cost)

end_time = time.time()
print('Total time taken: ', (end_time-start_time)/60, ' mins.')


# In[106]:


optimaleng=[]
for j in range(len(var)-3):
    optimaleng.append(np.sum(costarr[run*j:run*(j+1)])/len(costarr[run*j:run*(j+1)]))

(optimaleng)


# In[115]:


plt.style.use("seaborn")
#x=np.linspace(1,100,100)
plt.plot(var[:7],min_eng[:7],"turquoise",label="Ground state energies using exact diagonalization")
plt.plot(var[:7],optimaleng,"black",label="Ground state energies using QPU")
plt.ylabel("Ground state energies", fontsize=18)
plt.xlabel("Variance", fontsize=18)
plt.tick_params(axis="both", colors='black',which="major", labelsize=16)   # helps to increase the size of the values in X and Y axis
plt.tick_params(axis="both", colors='black', which="minor", labelsize=16)
plt.title("Spin glass model with mean=0.1", fontsize=17)
plt.ylim(-8, -5.5)
plt.legend()
plt.show()


# In[117]:


min_eng_arr=[]
for j in range(len(var)):
    min_eng_arr.append(min_energy[run*j:run*(j+1)])


variences1=[]
for i in range(len(var)):
    variences1.append(np.std(min_eng_arr[i]))
    

opt_eng_arr=[]
for j in range(len(var)-3):
    opt_eng_arr.append(costarr[run*j:run*(j+1)])
    
variences2=[]
for i in range(len(var)-3):
    variences2.append(np.std(opt_eng_arr[i]))
    
variences2==variences1


# In[121]:


fig, ax = plt.subplots()


ax.errorbar(var[:7], min_eng[:7],
            yerr=variences1[:7],
            fmt='-o',color="turquoise",label="Ground state energies using exact diagonalization")

ax.errorbar(var[:7], optimaleng,
            yerr=variences2,
            fmt='-o',color="black",label="Ground state energies using QPU")

ax.set_xlabel('Variance', fontsize=17)
ax.set_ylabel('Ground state energies', fontsize=17)
ax.set_title(f'Spin glass model with mean={mean} and no. of var points={len(var)-3}', fontsize=15)
plt.tick_params(axis="both", colors='black',which="major", labelsize=15)   # helps to increase the size of the values in X and Y axis
plt.tick_params(axis="both", colors='black', which="minor", labelsize=15)
#plt.ylim(-12,-4)
plt.legend()
plt.show()
# this is using hardware efficient ansatz not HVA


# In[124]:


np.linalg.norm(np.array(min_eng[:7])-np.array(optimaleng)) # gives the L_2 norm between true and observed ground state using QC


# In[99]:


@qml.qnode(dev)
def final_state_2(params,**kwargs):
    
    zz_params=params[:len(G.edges)]
    x_params=params[len(G.edges):]
    [qml.Hadamard(wires=w) for w in range(qubit_number)]
    
    for n in range(Trotter_steps):
        
        HVA_layer(zz_params,x_params)
    return qml.state()   


# In[100]:


cost_fn_test=qml.ExpvalCost(full_HVA_2,H[len(H)-1],dev)
cost_fn_test(params)
print(f"Ground state energy:{cost_fn_test(params)}")


# In[101]:


final_state_2(params)  # This is the last ket state which has ground state energy -12.757099592485492


# In[ ]:




