
def maximum(A, i, n, max_value):

        if (A[i] >= max_value):
                max_value = A[i]

	if (i == n-1):
		return max_value

	return maximum(A, i+1, n, max_value)
	
A = [3,5,2,1,7]

y = maximum(A, 0, 5, A[0])
print(y)	

