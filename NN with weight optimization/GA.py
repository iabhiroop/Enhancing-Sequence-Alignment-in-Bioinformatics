import pandas as pd

# load dataset
df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")

# drop unnecessary columns
df = df.drop(["ID", "ZIP Code"], axis=1)

# convert categorical variables to one-hot encoding
df = pd.get_dummies(df, columns=["Family", "Education", "Securities Account", "CD Account", "Online", "CreditCard"])

# split data into input and output variables
X = df.drop(["Personal Loan"], axis=1).values
Y = df["Personal Loan"].values.reshape(-1, 1)

# normalize input variables
X = (X - X.mean()) / X.std()

#%%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NeuralNetwork:

    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01, num_iterations=10000):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.params = {}
        self.grads = {}
        self.losses = []
        
    def initialize_parameters(self):
        np.random.seed(0)
        self.params['W1'] = np.random.randn(self.input_dim, self.hidden_dim) / np.sqrt(self.input_dim)
        self.params['b1'] = np.zeros((1, self.hidden_dim))
        self.params['W2'] = np.random.randn(self.hidden_dim, self.output_dim) / np.sqrt(self.hidden_dim)
        self.params['b2'] = np.zeros((1, self.output_dim))
        
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        Z1 = np.dot(X, self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.params['W2']) + self.params['b2']
        A2 = self.sigmoid(Z2)
        cache = (Z1, A1, Z2, A2)
        return A2, cache

    def compute_loss(self, Y, Y_pred):
        m = Y.shape[0]
        loss = (-1/m) * np.sum(Y * np.log(Y_pred) + (1-Y) * np.log(1-Y_pred))
        return loss

    def backward_propagation(self, X, Y, cache):
        m = X.shape[0]
        Z1, A1, Z2, A2 = cache
        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.params['W2'].T)
        dZ1 = dA1 * (Z1 > 0)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        self.grads['dW1'] = dW1
        self.grads['db1'] = db1
        self.grads['dW2'] = dW2
        self.grads['db2'] = db2
        
    def update_parameters(self):
        self.params['W1'] -= self.learning_rate * self.grads['dW1']
        self.params['b1'] -= self.learning_rate * self.grads['db1']
        self.params['W2'] -= self.learning_rate * self.grads['dW2']
        self.params['b2'] -= self.learning_rate * self.grads['db2']
        
    def fit(self, X, Y):
        self.initialize_parameters()
        for i in range(self.num_iterations):
            Y_pred, cache = self.forward_propagation(X)
            loss = self.compute_loss(Y, Y_pred)
            self.losses.append(loss)
            self.backward_propagation(X, Y, cache)
            self.update_parameters()
            if i % 1000 == 0:
                print(f"Iteration {i}: Loss={loss}")
                
    def predict(self, X):
        Y_pred, _ = self.forward_propagation(X)
        return np.round(Y_pred)

    def evaluate(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        print(f"Accuracy: {acc}")


#%%

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# nn = NeuralNetwork(input_dim=X.shape[1], hidden_dim=10, output_dim=1, learning_rate=0.01, num_iterations=100000)
# nn.fit(X_train, Y_train)

# nn.evaluate(X_test, Y_test)

#%%


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, elitism_rate, num_generations):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.num_generations = num_generations
        self.population = []
        self.fitness_scores = []
        self.best_individual = None
        self.best_fitness_score = None
        self.generation_losses = []
        self.initialize_population()
    
    def initialize_population(self):
        for i in range(self.population_size):
            nn = NeuralNetwork(input_dim=X.shape[1], hidden_dim=10, output_dim=1, learning_rate=0.01, num_iterations=10000)
            nn.initialize_parameters()
            self.population.append(nn.params)
    
    def compute_fitness_scores(self):
        self.fitness_scores = []
        for i in range(self.population_size):
            nn = NeuralNetwork(input_dim=X.shape[1], hidden_dim=10, output_dim=1, learning_rate=0.01, num_iterations=10000)
            nn.params = self.population[i]
            Y_pred, _ = nn.forward_propagation(X_train)
            fitness_score = accuracy_score(Y_train, np.round(Y_pred))
            self.fitness_scores.append(fitness_score)
    
    def select_parents(self):
        parents = []
        num_elites = int(self.elitism_rate * self.population_size)
        elite_indices = np.argsort(self.fitness_scores)[-num_elites:]
        elite_parents = [self.population[i] for i in elite_indices]
        parents += elite_parents
        non_elite_indices = np.argsort(self.fitness_scores)[:-num_elites]
        non_elite_fitness_scores = [self.fitness_scores[i] for i in non_elite_indices]
        total_fitness = sum(non_elite_fitness_scores)
        probabilities = [score/total_fitness for score in non_elite_fitness_scores]
        while len(parents) < self.population_size:
            parent1_index = np.random.choice(range(len(non_elite_indices)), p=probabilities)
            parent2_index = np.random.choice(range(len(non_elite_indices)), p=probabilities)
            parent1 = self.population[non_elite_indices[parent1_index]]
            parent2 = self.population[non_elite_indices[parent2_index]]
            parents.append(parent1)
            parents.append(parent2)
        return parents
    
    def crossover(self, parent1, parent2):
        child1 = {}
        child2 = {}
        for key in parent1.keys():
            if np.random.rand() < self.crossover_rate:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
            else:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
        return child1, child2
    
    def mutate(self, child):
        for key in child.keys():
            if np.random.rand() < self.mutation_rate:
                mutation = np.random.randn(*child[key].shape) * 0.01
                child[key] += mutation
        return child
    
    def evolve(self):
        for i in range(self.num_generations):
            self.compute_fitness_scores()
            self.generation_losses.append(1 - max(self.fitness_scores))
            if i % 10 == 0:
                print(f"Generation {i}: Loss={1 - max(self.fitness_scores)}")
            parents = self.select_parents()
            children = []
            for j in range(0, len(parents), 2):
                parent1 = parents[j]
                parent2 = parents[j+1]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                children.append(child1)
                children.append(child2)
            self.population = children
        self.compute_fitness_scores()
        best_index = np.argmax(self.fitness_scores)
        self.best_individual = self.population[best_index]
        self.best_fitness_score = self.fitness_scores[best_index]

#%%

# Define the genetic algorithm
ga = GeneticAlgorithm(population_size=1000, mutation_rate=0.15, crossover_rate=0.8, elitism_rate=0.3, num_generations=100)

# Run the genetic algorithm
ga.evolve()

# Print the best fitness score and accuracy achieved
print(f"Best Fitness Score: {ga.best_fitness_score}")

# Use the best individual to make predictions on test data
nn_test = NeuralNetwork(input_dim=X.shape[1], hidden_dim=10, output_dim=1, learning_rate=0.01, num_iterations=10000)
nn_test.params = ga.best_individual
Y_pred_test, _ = nn_test.forward_propagation(X_test)

# Round the predictions to get binary outputs
Y_pred_test = np.round(Y_pred_test)

# Calculate the accuracy score
accuracy = accuracy_score(Y_test, Y_pred_test)
print(f"Accuracy on Test Data: {accuracy}")


#%%

# nn = NeuralNetwork(input_dim=X.shape[1], hidden_dim=10, output_dim=3, learning_rate=0.01, num_iterations=10000)
# nn.params = ga.best_individual
# Y_pred, _ = nn.forward_propagation(X_test)
# Y_pred = np.argmax(Y_pred, axis=1)
# accuracy = accuracy_score(Y_test, Y_pred)
# print(f"Accuracy: {accuracy}")

#%%

