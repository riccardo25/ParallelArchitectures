import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.DataFrame({"resolution": [], "iterations": [], "extime":[], "savetime":[]}, dtype='int32')

for file in os.listdir("../log/"):
    
    if file.endswith(".log"):
        #print(os.path.join("../log/", file))
        f = open(os.path.join("../log/", file), mode="r")
        list = f.readlines()
        f.close()
        row = []
        for s in list:
            num =s.split(" ")
            row.append(int(num[-1]))
        if(row):
            data = data.append({"resolution": row[0], "iterations": row[1], "extime":row[2], "savetime":row[3]}, ignore_index = True)
        else:
            print(file)

print(data)
data.to_csv("results.csv")

if(False):

    d = data[ data.resolution == 800]["extime"]
    d_x = data[ data.resolution == 800]["iterations"]
    e = data[ data.resolution == 840]["extime"]
    e_x = data[ data.resolution == 840]["iterations"]
    f = data[ data.resolution == 880]["extime"]
    f_x= data[ data.resolution == 880]["iterations"]
    g = data[ data.resolution == 900]["extime"]
    g_x = data[ data.resolution == 900]["iterations"]
    h = data[ data.resolution == 940]["extime"]
    h_x = data[ data.resolution == 940]["iterations"]
    i = data[ data.resolution == 980]["extime"]
    i_x = data[ data.resolution == 980]["iterations"]

    x = np.arange(100, 1100, 100)

    plt.plot(d_x, d)
    plt.ylabel("Execution time")
    plt.plot(e_x, e)
    plt.ylabel("Execution time")
    plt.plot(f_x, f)
    plt.ylabel("Execution time")
    plt.plot(g_x, g)
    plt.ylabel("Execution time")
    plt.plot(h_x, h)
    plt.ylabel("Execution time")
    plt.plot(i_x, i)
    plt.ylabel("Execution time")
elif(False):
    d = data[ data.iterations == 100]["extime"]
    d_x = data[ data.iterations == 100]["resolution"]
    e = data[ data.iterations == 300]["extime"]
    e_x = data[ data.iterations == 300]["resolution"]
    f = data[ data.iterations == 600]["extime"]
    f_x= data[ data.iterations == 600]["resolution"]
    g = data[ data.iterations == 1000]["extime"]
    g_x = data[ data.iterations == 1000]["resolution"]
    plt.plot(d_x, d)
    plt.ylabel("Execution time")
    plt.plot(e_x, e)
    plt.ylabel("Execution time")
    plt.plot(f_x, f)
    plt.ylabel("Execution time")
    plt.plot(g_x, g)
elif(True):
    da2 = pd.read_csv("rsingle.csv")
    data = data.sort_values(["resolution"])
    da2 = da2.sort_values(["resolution"])
    d = data[ data.iterations == 800]["extime"]
    d_x = data[ data.iterations == 800]["resolution"]
    e = da2[ da2.iterations == 800]["extime"]
    e_x = da2[ da2.iterations == 800]["resolution"]
    plt.plot(d_x, d)
    plt.ylabel("Execution time")
    plt.plot(e_x, e)
    plt.ylabel("Execution time")

plt.show()




