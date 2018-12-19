import pdb
import copy
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def Dsigmoid(z):
    return np.diag((np.exp(-z)/(1 + np.exp(-z))**2).flatten())

def forward(net, x0, Sig):
    x = {}
    z = {}
    x[0] = x0[:,np.newaxis]
    z[0] = x0[:,np.newaxis]
    for layer in range(1,len(net)+1):
        A = net[layer][0]
        b = net[layer][1]
        # pdb.set_trace()
        z[layer] = A.dot(x[layer-1]) + b
        x[layer] = Sig[layer](z[layer])

    return x, z
        
def cost(net, x0, y, Sig):
    c = 0
    for epoch in range(x0.shape[0]):
        x, z = forward(net, x0[epoch,:], Sig)
        # pdb.set_trace()
        c += (x[len(net)] - y[epoch])**2
    return c/2.0

def line_search(net0, e0, DeDA, DeDb, x0, y, rate, Sig):
    net = copy.deepcopy(net0)
    e1 = e0
    sup_rate = 0
    local_minimum = 0
    # pdb.set_trace()
    while True:
        # Update the network
        for layer in range(1,L):
            net[layer][0] = net[layer][0] - rate*np.transpose(DeDA[layer])
            net[layer][1] = net[layer][1] - rate*np.transpose(DeDb[layer])

        # Compute cost function of updated network
        e2 = cost(net, x0, y, Sig)
        if e2 < e1:
            # Cost has decreased; keep going, but save best result so far.
            local_minimum = 0
            sup_rate += rate
            e1 = e2
            net1 = copy.deepcopy(net)
        else:
            # Cost is no longer decreasing; either decrease rate or stop.
            if e1 < e0:
                # We had a better cost along the way. Settle for it and stop
                return net1, sup_rate-rate
            else:
                # We never found a better cost, therefore either the rate was
                # too large or we hit a local minimum.
                if local_minimum == 10:
                    # We are at a local minimum
                    return net0, rate
                else:
                    # Rate was too large. Backtrack and try again with a lower
                    # rate.
                    local_minimum += 1
                    net = copy.deepcopy(net0)
                    sup_rate = 0
                    rate *= 0.1


# Number of samples for training                    
T = 1000

# Number of layers
L = 3

# Dimension of input and output for this fake test
dim = 1

# Dimension of inner layers
D = 5

# Rate memory
M = 10

# Number of samples in training batch
B = 2*dim*D + (L-2)*D**2

try:
    debug_mode = debug_mode
except:
    debug_mode = False

#### Create data
np.random.seed(0)
x0 = np.random.uniform(low=-2,high=4,size=(T, dim))
f_x = lambda x: (x-1)**3/8.0 - (x-1.5)/2.0 + 1
#y = 2*np.sin(x0*np.pi)
#y = ((x0-1)**3/8 - (x0-1)/2) + 1#(np.sum(x0, axis=1))
scale = 0.4
y_pure = f_x(x0)
y = y_pure + scale*np.random.standard_normal(size=(T,dim))
#x0 = x0

#### Create network

top = {}
Sig = {}
DSig = {}
for layer in range(1,L-1):
    top[layer] = (D, D)
    Sig[layer] = sigmoid
    DSig[layer] = Dsigmoid
top[1] = (D, dim)
top[L-1] = (1,D)
Sig[L-1] = lambda z: z
DSig[L-1] = lambda z: np.ones_like(z)

net = {}
for layer in range(1,L):
    A = np.random.standard_normal(size=(top[layer][0], top[layer][1]))
    b = np.random.standard_normal(size=(top[layer][0], 1))
    net[layer] = [A,b]

net0 = copy.deepcopy(net)

#### Initialize backpropagation
rate = 1e-2
N_iter = 3000
error_history = np.zeros(shape=(N_iter,))
rate_history = np.ndarray(shape=(N_iter,))
# Initialize first gradient
DeDb_old = {}
DeDA_old = {}
for layer in range(L-1,0,-1):
    DeDb_old[layer] = 0.0
    DeDA_old[layer] = 0.0
# Initialize conjugate gradient    
DeDb = {}
DeDA = {}
for layer in range(L-1,0,-1):
    DeDb[layer] = 0.0
    DeDA[layer] = 0.0
    
for iteration in range(N_iter):
    rate_history[iteration] = rate
    increase_rate = False
    net_old = copy.deepcopy(net)        
    DeDb_new = {}
    DeDA_new = {}
    for layer in range(L-1,0,-1):
        DeDb_new[layer] = 0.0
        DeDA_new[layer] = 0.0

    batch = np.random.choice(range(0,T), size=B, replace=False)
    for epoch in batch:
        # Forward propagation
        x, z = forward(net, x0[epoch,:], Sig)

        # Derivative at last layer
        DeDx = {}
        DeDx[L-1] = x[L-1] - y[epoch]
        error_history[iteration] += DeDx[L-1][0,0]**2
        
        # Derivative at other layers
        for layer in range(L-1,0,-1):
            DSig_z = DSig[layer](z[layer])
            DeDb_epoch = DeDx[layer].dot(DSig_z)
            DeDb_new[layer] += DeDb_epoch
            DeDA_new[layer] += np.outer(x[layer-1], DeDb_epoch)
            DxDx = np.reshape(DSig_z.dot(net[layer][0]), top[layer])
            DeDx[layer-1] = DeDx[layer].dot(DxDx)

    # Conjugate gradient
    if iteration > 0:
        n_n = 0.0
        n_n_minus = 0.0
        n_minus_n_minus = 0.0
        for layer in range(L-1,0,-1):
            n_n += np.sum(DeDA_new[layer]*DeDA_new[layer])
            n_n += np.sum(DeDb_new[layer]*DeDb_new[layer])
            n_n_minus += np.sum(DeDA_new[layer]*DeDA_old[layer])
            n_n_minus += np.sum(DeDb_new[layer]*DeDb_old[layer])
            n_minus_n_minus += np.sum(DeDA_old[layer]*DeDA_old[layer])
            n_minus_n_minus += np.sum(DeDb_old[layer]*DeDb_old[layer])        

        beta = (n_n - n_n_minus)/n_minus_n_minus
    else:
        beta = 1.0
            
    for layer in range(L-1,0,-1):
        DeDA[layer] = DeDA_new[layer] + beta*DeDA[layer]
        DeDb[layer] = DeDb_new[layer] + beta*DeDb[layer]        

    # Update the gradient
    DeDA_old = copy.deepcopy(DeDA_new)
    DeDb_old = copy.deepcopy(DeDb_new)

    # Normalization
    error_history[iteration] /= 2.0
            
    # Update the parameters
    net, new_rate = line_search(net, error_history[iteration],
                                DeDA, DeDb, x0[batch,:], y[batch], rate, Sig)
    rate = ((M-1)*rate + new_rate)/M

    if debug_mode:
        pdb.set_trace()
        
    #for layer in range(1,L):
    #    net[layer][0] = net[layer][0] - rate*np.transpose(DeDA[layer])
    #    net[layer][1] = net[layer][1] - rate*np.transpose(DeDb[layer])

    if (iteration % 100 == 0):
        print("Error at iteration %d: %f" % (iteration, error_history[iteration]))
    if iteration > 0 and False:
        if error_history[iteration] > error_history[iteration-1]:
            # Error is increasing! Revert, and decrease learning rate
            net = copy.deepcopy(net_old)
            rate /= 1.1
            increase_rate = True
            continue
        elif (error_history[iteration-1] - error_history[iteration])/error_history[iteration] < 1e-9:
            # Error is decreasing too slowly. Stop
            break
        else:
            # Error is decreasing very fast. Increase learning rate
            if increase_rate:
                increase_rate = False
                rate *= 1.1
        
#### Plots
# Original predictions
y_0 = np.empty(shape=(T,))
y_ = np.empty(shape=(T,))
for epoch in range(0,T):
    x, z = forward(net0, x0[epoch,:], Sig)
    y_0[epoch] = x[L-1].flatten()[0]

    x, z = forward(net, x0[epoch,:], Sig)
    y_[epoch] = x[L-1].flatten()[0]

plt.close('all')
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
ax1.scatter(x0, y)
ax1.scatter(x0, y_pure, color='g')
ax1.scatter(x0,y_0,color='r')
ax2.scatter(x0, y)
ax2.scatter(x0, y_pure, color='g')
ax2.scatter(x0,y_,color='r')

plt.show()
