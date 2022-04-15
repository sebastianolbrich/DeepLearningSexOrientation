import math
import matplotlib.pyplot as plt

x = []

for j in range(500): #Half the range of n
    x.append(2*j-1)

probabilities = [0.51, 0.52, 0.53, 0.55, 0.57]
ProbSum = 0
ResultProbs59 = []
probability = 0.59

for n in range(1000):
    if n%2 == 1: #Prevents fluctuation from even n giving lower results, as the even distribution does not count.
        for i in range(n + 1): 
            if i > n/2:
                Prob = math.factorial(n)/(math.factorial(i)*math.factorial(n-i)) * probability ** i * (1-probability) ** (n - i)
                ProbSum = ProbSum + Prob

        ResultProbs59.append(ProbSum - (ProbSum%0.0001))
        ProbSum = 0

#print(ResultProbs)

plt.plot(x,ResultProbs51, color = "red", label = "51%")
plt.plot(x,ResultProbs52, color = "blue", label = "52%")
plt.plot(x,ResultProbs53, color = "grey", label = "53%")
plt.plot(x,ResultProbs54, color = "yellow", label = "54%")
plt.plot(x,ResultProbs55, color = "brown", label = "55%")
plt.plot(x,ResultProbs56, color = "darkblue", label = "56%")
plt.plot(x,ResultProbs57, color = "darkgrey", label = "57%")
plt.plot(x,ResultProbs58, color = "darkgreen", label = "58%")
plt.plot(x,ResultProbs59, color = "darkred", label = "59%")
plt.plot(x,ResultProbs60, color = "green", label = "60%")
plt.ylabel('Probability of correct classification')
plt.xlabel('Segments for classification per subject')
plt.title('Increases probability for correct classification ')
plt.legend()
plt.show()