import numpy as np
import random

tau = 1000 #exploration steps
T = 4000
mu = 0.9 #confidence parameter
n = 10 #no. of clusters
c = np.array([random.uniform(0.5,0.8) for i in range(T)]) #c is the bid money for cluster
# def customer_simulate(i, ):
a = np.zeros((n, tau)) #a is 1 if offer given to i on t
p = np.zeros((n, tau)) #payment received by i at t
lambda_plus = np.zeros(n)
lambda_minus = np.zeros(n)
lambda_real = np.array([random.uniform(0.1,0.9) for i in range(n)]) #simulating actual lambda

lambda_est = np.zeros(n) #estimation of lambda
count = np.zeros(n) #number of times i has been requested
success = np.zeros(n) #number of times i has accepted DR request

e = np.array([random.uniform(0.1,0.2) for i in range(T)])

for t in range(tau):
	i = (t-1)%n
	a[i][t] = 1
	chance = random.uniform(0,1)
	count[i]+=1
	if (chance < lambda_real[i]): #offer accepted
		p[i][t] = max(c)
		success[i]+=1

for i in range(n):
	lambda_est[i] = success[i]/count[i]
	term = n*np.log(2/mu)/(2*tau)
	lambda_plus[i] = min(1, lambda_est[i]+term)
	lambda_minus[i] = max(0, lambda_est[i]-term)

lambda_real = np.round(lambda_real, 3)
lambda_est = np.round(lambda_est, 3)
lambda_minus = np.round(lambda_minus, 3)
lambda_plus = np.round(lambda_plus, 3)
print("Real       ", lambda_real)
print("Estimated  ", lambda_est)
print("Lower bound", lambda_minus)
print("Upper bound", lambda_plus)

#Exploitation begins
cumulative_regret = 0

for t in range(tau+1,T):
	argmin_arr = [lambda_plus[j]*(lambda_minus[j]+c[j]-2*e[t]) for j in range(n)]
	i = argmin_arr.index(min(argmin_arr))
	#offer will be made to consumer i
	argmin_arr_real = [lambda_real[j]*(lambda_real[j]+c[j]-2*e[t]) for j in range(n)]
	i_star = argmin_arr_real.index(min(argmin_arr_real))

	cumulative_regret += lambda_est[i]*(lambda_est[i]+c[i]-2*e[t]) - lambda_real[i_star]*(lambda_real[i_star]+c[i_star]-2*e[t])

	if lambda_minus[i]+c[i]-2*e[t] < 0:
		a[i][t] = max(argmin_arr);
		j = argmin_arr.index(min(argmin_arr))
		if (lambda_minus[j]+c[j]-2*e[t] < 0):
			term1 = lambda_plus[j]*(lambda_minus[j]+c[j])-2*e[j]
			term2 = lambda_plus[i]*(lambda_minus[i]+c[i])-2*e[t]

			p[i][t] = min((term1-term2)/lambda_plus[i], max(c))
			#making a payment of p[i] to consumer i if accepted
		else:
			p[i][t] = min(max(c), -(lambda_minus[i]-2*e[t]))