import matplotlib.pyplot as plt
arr = [9.9703443e-01, 3.8586048e-05, 1.7995077e-01, 1.2243818e-04, 2.3720576e-03, 1.1884812e-05, 9.5516414e-05, 3.0711814e-04]
labels = ['classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae']

plt.bar(labels, list(map(lambda x:x*100,arr)))
plt.show()