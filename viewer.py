import matplotlib.pyplot as plt

ctrs = []
zs = []
xs = []

fin = open("filtered.csv").readlines()
for line in fin:
    ctr = int(line.rstrip().split(',')[0])
    val = int(line.rstrip().split(',')[1])
    filt = float(line.rstrip().split(',')[2])

    ctrs.append(ctr)
    zs.append(val)
    xs.append(filt)


plt.plot(ctrs,zs,'b',label="observation")
plt.plot(ctrs,xs,'r',label="state")
plt.legend()
plt.show()
