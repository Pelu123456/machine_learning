import matplotlib.pyplot as plt

x1 = -2
x2 = 1
y1 = 2.5
y2 = -1

plt.style.use('fivethirtyeight')
plt.style.use('ggplot') 

plt.xlim(-1.5, 2.5)
plt.ylim(-1.5, 2.5) 

plt.scatter([1,0,1], [1,1,0])
plt.scatter([0],[0])

plt.plot([x1, y1], [x2, y2], label="0.4 * x1 + 0.9 * x2 - 0.1 = 0", linewidth=0.5) 

plt.legend(loc="upper left")
plt.show()


