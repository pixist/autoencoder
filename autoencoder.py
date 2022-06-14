import h5py
import numpy as np
import matplotlib.pyplot as plt

class AutoEncoder:
    def __init__(self, Lin, Lhid, lmbda, beta, rho):
        self.Lin = Lin
        self.Lhid = Lhid
        self.lmbda = lmbda
        self.beta = beta
        self.rho = rho
        
    def initializeLayers(self, N):
        self.hidden = np.zeros((self.Lhid,N))
        
    def initializeWeights(self):
        w0  = np.sqrt(6/(self.Lin+self.Lhid))
        self.weights = np.random.uniform(-w0,w0,(self.Lhid,self.Lin))
        self.bIH = np.random.uniform(-w0,w0,self.Lhid)
        self.bHO = np.random.uniform(-w0,w0,self.Lin)
        self.JgradW_ho, self.JgradW_ih, self.Jgradb_ho, self.Jgradb_ih = 0, 0, 0, 0
    
    def forward(self, data):
        self.A1 = np.dot(self.weights,data.T)+self.bIH[:,None]
        self.hidden = sigmoid(self.A1)
        self.p_b = np.mean(self.hidden,axis=1)[:,None]
        self.A2 = np.dot(self.weights.T,self.hidden)+self.bHO[:,None]
        self.out = sigmoid(self.A2).T
        return self.out
    
    def calculateCost(self, data, batchCount):
        meanSquaredError = np.square(data-self.out).mean()/2
        tykhonov = (np.sum(np.square(self.weights))+np.sum(np.square(self.weights.T)))*self.lmbda/2
        KL = self.rho*np.log(self.rho/self.p_b) + (1-self.rho)*np.log((1-self.rho)/(1-self.p_b))
        self.J = meanSquaredError + tykhonov/batchCount + self.beta*np.sum(KL)
        return self.J, meanSquaredError, tykhonov/batchCount, self.beta*np.sum(KL)
        
    def calculateGrad(self, data, momentum, batchCount):
        # Sorry for sphagett
        N = data.shape[0]
        gradWT, gradW, gradb1, gradb2 = 0,0,0,0
        for i in range(N):
            outAvg = self.out[i][:,None]
            hidAvg = self.hidden[:,i][:,None]
            datAvg = data[i][:,None]
            dJ1_do__do_dA2 = (outAvg-datAvg)*(outAvg*(1-outAvg))/N
            T3 = self.beta*((1-self.rho)/(1-hidAvg)-self.rho/hidAvg)*(1/N)
            gradWT = gradWT + dJ1_do__do_dA2*hidAvg.T
            gradW = gradW + (np.sum(dJ1_do__do_dA2*(self.weights.T),axis=0).reshape((self.Lhid,1))+T3)*(hidAvg*(1-hidAvg))*datAvg.T
            gradb2 = gradb2 + dJ1_do__do_dA2
            gradb1 = gradb1 + (np.sum(dJ1_do__do_dA2*(self.weights.T),axis=0).reshape((self.Lhid,1))+T3)*(hidAvg*(1-hidAvg))
        self.JgradW_ho = gradWT + momentum*self.JgradW_ho + self.lmbda*self.weights.T/batchCount
        self.JgradW_ih = gradW + momentum*self.JgradW_ih + self.lmbda*self.weights/batchCount
        self.Jgradb_ho = gradb2 + momentum*self.Jgradb_ho
        self.Jgradb_ih = gradb1 + momentum*self.Jgradb_ih
    
        
        
    def updateWeights(self, LR):
        self.weights = self.weights - LR*self.JgradW_ih - LR*self.JgradW_ho.T
        self.bIH = self.bIH - LR*self.Jgradb_ih.reshape(self.Lhid,)
        self.bHO = self.bHO - LR*self.Jgradb_ho.reshape(self.Lin,)
        
def trainMiniQ1(encoder, data, epoch, learningRate, momentum, batchSize = 32):
    global lossList
    global samples
    lossList = []
    N = data.shape[0]
    batchCount = N//batchSize
    remainder = N % batchSize
    remLimit = N - remainder
    for e in range(epoch):
        permutation = np.random.permutation(N)
        shuffled_samples = data[permutation]
        samples = np.array_split(shuffled_samples[:remLimit], batchCount)
        if remainder != 0:
            samples.append(shuffled_samples[remLimit:])
            print("yes")
        loss = 0
        for j in range(len(samples)):
            bSize = samples[j].shape[0]
            encoder.forward(samples[j])
            loss += np.array(encoder.calculateCost(samples[j], len(samples)))
            encoder.calculateGrad(samples[j], momentum, len(samples))
            encoder.updateWeights(learningRate)
        lossList.append(np.trunc(loss*10**6)/(10**6))
        print(f"Loss [Total, MSE, Tykhonov, KL] in epoch {e+1}: {lossList[e]}")
    return lossList, samples

def train(encoder, data, epoch, learningRate, momentum):
    global lossList
    lossList = []
    for i in range(epoch):
        encoder.forward(data)
        loss = np.array(encoder.calculateCost(data))
        lossList.append(np.trunc(loss*10**6)/(10**6))
        print(f"Loss [Total, MSE, Tykhonov, KL] in epoch {i+1}: {lossList[i]}")
        encoder.calculateGrad(data, momentum)
        encoder.updateWeights(learningRate)
    return lossList

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def normalizeGray(arrayIn):
    gray_scale = 0.2126*arrayIn[:,0,:] + 0.7152*arrayIn[:,0,:] + 0.0722*arrayIn[:,0,:]
    mean = gray_scale.mean(axis=1)
    gray_scale = gray_scale - mean[:,None]
    std = gray_scale.std()
    clipped = np.clip(gray_scale, -3*std, 3*std)
    minG, maxG = np.min(clipped), np.max(clipped)
    normalizedOut = (clipped-minG)*4/(maxG-minG)/5 + 0.1
    return normalizedOut


def plotArrayGray(arrayIn, rows, columns, offset=0):
    fig = plt.figure(figsize=(2*rows,1.2*columns))
    minA, maxA = np.min(arrayIn), np.max(arrayIn)
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(arrayIn[i+offset-1].reshape((16,16)), cmap='gray', vmin=minA, vmax=maxA)
        plt.axis('off')
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()

def plotArrayRGB(arrayIn, rows, columns, offset=0):
    fig = plt.figure(figsize=(2*rows,1.2*columns))
    minA, maxA = np.min(arrayIn), np.max(arrayIn)
    arrayIn = arrayIn.reshape((columns*rows,3,16,16))
    arrayIn = [[[tuple(row) for row in xdim] for xdim in np.moveaxis(instance, 0, -1)] for instance in arrayIn]
    #arrayIn = [tuple(row) for row in np.moveaxis(arrayIn, 1, -1)]
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(arrayIn[i+offset-1], vmin=minA, vmax=maxA)
        plt.axis('off')
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

def plotParameter(metric, labels, metricName):
    plt.figure(figsize = (12,6))
    xlabel = [str(i) for i in range(len(metric[0]))]
    for i in range(len(labels)):
        plt.plot(xlabel, metric[i], marker='o', markersize=6, linewidth=2, label=labels[i])
    plt.ylabel(metricName[0])
    plt.title(f'{metricName[1]} with {metricName[2]} Hidden Neurons, Lambda: {metricName[3]}, Beta: {metricName[4]}, Rho: {metricName[5]} Learning Rate: {metricName[6]}, Momentum: {metricName[7]}')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
# In[] Read the data
filename = "data1.h5"

with h5py.File(filename, "r") as f:
    groupKeys = list(f.keys())
    sets = []
    for key in groupKeys:
        sets.append(list(f[key]))
# In[]
images_clip = np.array(sets[0][:])
images = images_clip.reshape((images_clip.shape[0],3,256))
normalized = normalizeGray(images)
# In[]
idx = np.random.choice(images.shape[0], 200, replace=False)
"""
# In[]
plotArrayGray(normalized[idx], 10, 20)
plotArrayRGB(images[idx], 10, 20)

# In[]
Lhid = 8**2 #4*4, 7*7, 9*9
lmbda = 0.0005*320
beta = 0.005
rho = 0.4
Encoder = AutoEncoder(256, Lhid, lmbda, beta, rho)
Encoder.initializeLayers(32)
Encoder.initializeWeights()
loss = []
# In[]
lr = 0.3
mm = 0.6
epoch = 30
print(f"Started Training with learning rate = {lr}, momentum = {mm}, beta = {beta}, rho ={rho}")
l, out_shuffled = trainMiniQ1(Encoder, normalized, epoch, lr, mm)
loss.extend(l)
# In[]
plotArrayGray(out_shuffled[-1], 4,8)
plotArrayGray(Encoder.out,4,8)
plotArrayGray(Encoder.weights,8,8)
# In[]
plotParameter(np.array(loss).T, ["Loss Function","MSE","Tykhonov","KL"], ["Loss","Auto-Encoder",Lhid,lmbda,beta,rho,lr,mm])
"""
# In[]
hids = [4,7,9]
lmbds = [0,0.0003, 0.001]
epochs = [70,70,70]
for i in range(3):
    for j in range(3):
# In[]
        Lhid = hids[i]**2 #4*4, 7*7, 9*9
        lmbda = lmbds[j]
        beta = 0.005
        rho = 0.4
        Encoder = AutoEncoder(256, Lhid, lmbda, beta, rho)
        Encoder.initializeLayers(10240)
        Encoder.initializeWeights()
        loss = []
        # In[]
        lr = 0.3
        mm = 0.6
        epoch = epochs[i]
        print(f"Started Training with learning rate = {lr}, momentum = {mm}, beta = {beta}, rho ={rho}")
        l, out_shuffled = trainMiniQ1(Encoder, normalized, epoch, lr, mm)
        loss.extend(l)
        # In[]
        #plotArrayGray(Encoder.out[idx],10,20)
        plotArrayGray(Encoder.weights,hids[i],hids[i])
        # In[]
        plotParameter(np.array(loss).T, ["Loss Function","MSE","Tykhonov","KL"], ["Loss","Auto-Encoder",Lhid,lmbda,beta,rho,lr,mm])
#%%
#%%"""
