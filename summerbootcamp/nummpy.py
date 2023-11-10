import numpy as np

a = np.array([1,2,3,4])
b = np.array([2,3,4,5])


random = np.random.randint(0, 10, size=10)

normal = np.random.normal(10, 4, (3, 4))


a.ndim	
a.shape	
a.size
a.dtype

random.reshape(3, 3)	#error 10 cannot div to (3, 3)

m = np.random.randint(10, size=(3, 5))

m[2, 3] = 2.9 # m[2, 3] will be 2
m[:, 0]	#select all rows

v = np.arange(0, 30, 3)

catch = [1,2,3]
v[catch]	#catch[1], catch[2] and catch[3] *FANCY INDEX*

v[v < 3] #just smaller than 3

v / 5	#all elements divided to 5

v = np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.min(v)
np.max(v)
np.sum(v)
np.var(v)


n = np.array([[5,1], [1,3]])
k = np.array([12, 10])

						# x0 + 3*x1 = 10 
np.linalg.solve(n, k)	# 5*x0 + x1 =12

