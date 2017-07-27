import numpy as np

a = np.array([ [[1, 2, 3],
                [2, 1, 4],
                [5, 2, 1],
                [6, 3, 2]],
               [[5, 1, 3],
                [1, 3, 4],
                [4, 2, 6],
                [3, 9, 3]],
               [[4, 5, 6],
                [7, 4, 3],
                [2, 1, 5],
                [4, 3, 1]] ])

# print(np.argmax(a, axis=2))  # [1, 2, 3] -> 2, [2, 1, 4] -> 2, [5, 2, 1] -> 0, [6, 3, 2] -> 0 ::=> [[2 2 0 0], [0 2 2 1], [2 0 2 0]]
# print(np.argmax(a, axis=1))  # [1, 2, 5, 6] -> 3, [2, 1, 2, 3] -> 3, [3, 4, 1, 2] -> 1 ::=> [[3 3 1], [0 3 2], [1 0 0]]
# print(np.argmax(a, axis=0))  # [1, 5, 4] -> 1, [2, 1, 5] -> 2, [3, 3, 6] -> 2 ::=> [[1 2 2], [2 2 0], [0 0 1], [0 1 1]]

b = np.array([1, 3, 4, 2])
c = np.array([5, 3, 2, 1])
print(np.mean(b == c))

v = [1,2,3,4,5]
print(np.mean(v))