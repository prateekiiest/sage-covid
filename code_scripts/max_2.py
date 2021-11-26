def maximum(A, n):
	
	if (n == 0):
		return A[n]

	max_value = maximum(A, n-1)

	if (max_value > A[n]):
		return max_value
	else: 
		return A[n]

A = [8,2,4,7,1]

print(maximum(A, 4))
