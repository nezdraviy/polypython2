import numpy as np
import sys
import array


# 1
# Ответ: f -float,c-complex,s-str,b-bool,U-Unicode,o - object,V-void,u-unit

# 2
a1 =array.array('f',[1.1,2.2,3.3]) 
print (sys.getsizeof(a1))
print (type(a1))

# 3 
a2 = np.array([i/4 for i in range(5)])
print(a2)
a2 = np.linspace(0, 1, 5 )
print(a2)

# 4 
a3=np.random.uniform(0, 1, 5)
print(a3)
a3=np.random.rand(5)
print(a3)

# 5
a4 = np.random.normal(0, 1, 5)
print(a4)
print(a4.mean())
print(np.median(a4))
print(np.var(a4)) 

# 6
a5 = np.random.randint(10, size = 5)
print(a5)

# 7
a6 = np.random.randint(10,size=(3,4))
print(a6)
print(a6[:2,:3])
print(a6[:3,1])
print(a6[::-1,::-1])
print(a6[:,1])
print(a6[2])

# 8
a7 = a6[:,:]
print(a7)
print(a6 is a7)

# 9
a8= np.arange(1,13)
print(a8)
print(a8[np.newaxis,:])
print(a8[:,np.newaxis])

# 10
x= np.array([1,2,3])
y= np.array([4,5,6])
r1 = np.dstack([x,y])
print(r1)
print(np.dstack([r1,r1]))

# 11
a10= np.array([1, 2, 3, 4, 5, 6])
print(np.split(a10,2))
a10 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np.vsplit(a10,3))
a10  = array_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(np.hsplit(a10,2))
a10 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
print(np.dsplit(a10,2))

# 12
x = np.arange(10)

result_add = np.add(x, 5)  
print("Сложение (x + 5):", result_add)

result_subtract = np.subtract(x, 3)
print("Вычитание (x - 3):", result_subtract)

result_multiply = np.multiply(x, 2)  
print("Умножение (x * 2):", result_multiply)

result_divide = np.divide(x, 2) 
print("Деление (x / 2):", result_divide)

result_integer_divide = np.floor_divide(x, 2)
print("Целочисленное деление (x // 2):", result_integer_divide)

result_power = np.power(x, 2)  
print("Возведение в степень (x ** 2):", result_power)

result_modulus = np.mod(x, 3) 
print("Остаток от деления (x % 3):", result_modulus)