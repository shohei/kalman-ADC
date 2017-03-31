import matplotlib.pyplot as plt
import math

sigmaxs_hat_bar = [] # a priori
sigmaxs_hat = [] # a posteriori
sigmax_0 = 37.0
sigmaz_list = [10.0,50.0,100.0,1000.0]
sigmaz = sigmaz_list[1] 
ks = []

# fin = open("master.csv").readlines()
# fin = open("5.csv").readlines()
fin = open("nucleo.csv").readlines()

zs = []
xs_hat_bar = [ ] # a priori estimated value
xs_hat = [] # a posteriori estimated value
counter = 0
counters = []

def estimate_step(counter,x_hat,sigmax_hat):
    xs_hat_bar.append(x_hat)
    sigmaxs_hat_bar.append(math.sqrt((sigmax_hat*sigmax_hat) + (sigmax_0*sigmax_0) ))

def filter_step():
    k = sigmaxs_hat_bar[-1]**2 / (sigmaxs_hat_bar[-1]**2 + sigmaz**2)
    ks.append(k)
    x_hat = xs_hat_bar[-1] + k * (zs[-1] - xs_hat_bar[-1])
    xs_hat.append(x_hat)
    sigmax_hat = (1-k)*sigmaxs_hat_bar[-1]
    sigmaxs_hat.append(sigmax_hat)

for line in fin:
    z = float(line.rstrip().split(',')[1])
    zs.append(z)
    if(counter==0):
        xs_hat.append(z)
        sigmaxs_hat.append(sigmax_0)
    estimate_step(counter,xs_hat[-1],sigmaxs_hat[-1])
    filter_step()
    counters.append(counter)
    counter+=1

plt.plot(counters,xs_hat[1:],'r',linewidth=4.0,label="filtered state")
plt.plot(counters,zs,'b',label="observation")
plt.xlim([200,300])
plt.legend()
plt.show()
