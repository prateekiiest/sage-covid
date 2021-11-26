
def subsets(S, i, n):

        S2 = S + [i]
        print(S2)

	if (i == 1):
		print('[]')

	if (i == n):
		return
	
	subsets(S2, i+1, n)
	subsets(S, i+1, n)

S=[]
subsets(S, 1, 4)

