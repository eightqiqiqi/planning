import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1, 2, 3, 4], [1, 4, 9, 16])
ax.set_title("Test Plot")
plt.show()
