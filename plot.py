import matplotlib.pyplot as plt
import re
import serial

# pattern = re.compile(r'\d+\.\d')

fout = open("out.csv","w")

ser = serial.Serial("/dev/tty.usbmodem1411",9600)
i = 0
ii = []
res = []
while (i<1001):
    snw = ser.readline().rstrip()
    # snw = pattern.findall(line)
    if snw:
        # snw = float(snw)
        # snw = float(snw)
        # plt.plot(ii,res)
        # plt.show()
        print("{},{}\n".format(i,snw))
        fout.write("{},{}\n".format(i,snw))
        i = i+1 

ser.close()

fout.close()

