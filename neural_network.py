import numpy as np
import pygame
import random
from math import *


POSITIVE_COLOR = (36,232,70)
NEGATIVE_COLOR = (230,0,230)
NEUTRAL_COLOR = (0,0,0)
class NeuralNetwork:
    def __init__(self, input_node_number, output_node_number, hidden_layer_node_number, hidden_layer_number, activation_function):

        self.input_node_number = input_node_number
        self.output_node_number = output_node_number
        self.hidden_layer_node_number = hidden_layer_node_number
        self.hidden_layer_number = hidden_layer_number

        self.biases = np.array([np.zeros(self.hidden_layer_node_number) for _ in range(self.hidden_layer_number)]) # initialisation de la matrice à deux dimensions qui contient les biais de chaque nodes
        self.weights = np.array([[np.zeros(self.hidden_layer_node_number) for j in range(self.hidden_layer_node_number)] for i in range(self.hidden_layer_number-1)]) #initialisation de la matrice à trois dimensions qui contient les poids de chaque liaison


        self.input_weights = np.array([np.zeros(self.input_node_number) for _ in range(self.hidden_layer_node_number)])
        self.output_weights = np.array([np.zeros(self.hidden_layer_node_number) for _ in range(self.output_node_number)])

        self.inputs = np.zeros(self.input_node_number)
        self.outputs = np.zeros(self.output_node_number)

        self.hidden_layers = np.array([np.zeros(self.hidden_layer_node_number) for _ in range(self.hidden_layer_number)])

        self.activation_function = activation_function

        


        # data to draw the visualization
        self.x, self.y, self.width, self.height  = 0,0,0,0
        self.set_visualization_window()

        
    def generate_random_configuration(self, reference, mutation_probability, evolution_probability, unchanged_probability): #proba_mutation + proba_evolution + proba_continuite = 1
        for i in range(self.hidden_layer_node_number):
            for j in range(self.input_node_number):
                random_number = random.uniform(0,1)
                if random_number < mutation_probability : #fully random
                    self.input_weights[i][j] = random.uniform(-2,2)
                elif random_number < mutation_probability + evolution_probability : #evolution depending on the reference
                    self.input_weights[i][j] = min(max(reference.input_weights[i][j] + random.uniform(-0.5, 0.5), -2), 2)
                else : #no change at all
                    self.input_weights[i][j] = reference.input_weights[i][j]

        for i in range(self.hidden_layer_number-1):
            for j in range(self.hidden_layer_node_number):
                for k in range(self.hidden_layer_node_number):
                    random_number = random.uniform(0,1)
                    if random_number < mutation_probability : #fully random
                        self.weights[i][j][k] = random.uniform(-2,2)
                    elif random_number < mutation_probability + evolution_probability : #evolution depending on the reference
                        self.weights[i][j][k] = min(max(reference.weights[i][j][k] + random.uniform(-0.5, 0.5), -2), 2)
                    else : #no change at all
                        self.weights[i][j][k] = reference.weights[i][j][k]                  
            
        for i in range(self.output_node_number):
            for j in range(self.hidden_layer_node_number):
                random_number = random.uniform(0,1)
                if random_number < mutation_probability : #fully random
                    self.output_weights[i][j] = random.uniform(-2,2)
                elif random_number < mutation_probability + evolution_probability : #evolution depending on the reference
                    self.output_weights[i][j] = min(max(reference.output_weights[i][j] + random.uniform(-0.5,0.5), -2), 2)
                else : #no change at all
                    self.output_weights[i][j] = reference.output_weights[i][j]

        ##############################################################

        for i in range(self.hidden_layer_number):
            for j in range(self.hidden_layer_node_number):
                random_number = random.uniform(0,1)
                if random_number < mutation_probability : #fully random
                    self.biases[i][j] = random.uniform(-1,1)
                elif random_number < mutation_probability + evolution_probability : #evolution depending on the reference
                    self.biases[i][j] = min(max(reference.biases[i][j] + random.uniform(-0.05, 0.05), -1), 1)
                else : #no change at all
                    self.biases[i][j] = reference.biases[i][j]

        # print("new model : ")
        # self.display()


    def predict(self):
        
        self.normalize_input()
        self.hidden_layers[0] = np.dot(self.input_weights, self.inputs)
        self.hidden_layers[0] = np.add(self.hidden_layers[0], self.biases[0])


        
        
        
        for i in range(1,self.hidden_layer_number):
            
            self.hidden_layers[i] = np.dot(self.weights[i-1],self.hidden_layers[i-1])
            self.hidden_layers[i] = np.add(self.hidden_layers[i-1], self.biases[i-1])
            self.hidden_layers[i] = self.activation_function(self.hidden_layers[i-1])
            
            # if (np.all(self.hidden_layers[i+1]==self.hidden_layers[i])):
            #     print("hidden layers are equal")

        # print("\n ================================================\n")
        # print(self.hidden_layers[1])
        # print(self.weights[0])

        # print(self.biases[0])
        # print(self.hidden_layers[2])

        
        # print(self.hidden_layers)
        # print("\n================================================\n")

        # print(self.inputs)
        # print(self.hidden_layers)
        # print(self.outputs)
                
        self.outputs = np.dot(self.output_weights, self.hidden_layers[-1])
        self.outputs = self.activation_function(self.outputs)

        
    def save(self, filename):
        with open(filename, 'w') as file:
            for i in range(self.hidden_layer_node_number):
                for j in range(self.input_node_number):
                    file.write(str(self.input_weights[i][j]) + ";")
                file.write("\n")

            file.write("\n\n")
            for k in range (self.hidden_layer_number-1):
                for i in range(self.hidden_layer_node_number):
                    for j in range(self.hidden_layer_node_number):
                        file.write(str(self.weights[k][i][j])+";")
                    file.write("\n")
                file.write("\n")


            file.write("\n")
            for i in range(self.output_node_number):
                for j in range(self.hidden_layer_node_number):
                    file.write(str(self.output_weights[i][j]) + ";")
                file.write("\n")
            file.write("\n\n")
            for i in range(self.hidden_layer_number):
                for j in range(self.hidden_layer_node_number):
                    file.write(str(self.biases[i][j]) + ";")
                file.write("\n")
    
            

    def load(self,filename):
        with open(filename, 'r') as file:

            document = []
            for line in file:
                document.append(line)

            k = 0
            i_relative = 0
            # print(self.hidden_layer_node_number + 1)
            # print(self.hidden_layer_node_number+(self.hidden_layer_node_number+1)*(self.hidden_layer_number-1) + 2)
            # print(self.hidden_layer_node_number+(self.hidden_layer_node_number+1)*(self.hidden_layer_number-1) +self.output_node_number + 4)
            for i in range(0,len(document)):
                if i <= self.hidden_layer_node_number + 1:
                    if document[i]!= "\n":
                        values = document[i].split(';')[:-1]
                        for j in range(len(values)):
                            self.input_weights[i][j] = values[j]
                elif i <= self.hidden_layer_node_number+(self.hidden_layer_node_number+1)*(self.hidden_layer_number-1) + 2:
                    
                    
                    if document[i]== "\n":
                        k+=1
                        i_relative = 0
                        
                    else:
                        values = document[i].split(';')[:-1]
                        for j in range(len(values)):
            
                            self.weights[k][i_relative][j] = values[j]
                        i_relative+=1
                elif i <= self.hidden_layer_node_number+(self.hidden_layer_node_number+1)*(self.hidden_layer_number-1) +self.output_node_number + 4:
                    if document[i]== "\n":
                        i_relative = 0
                    else:
                        values = document[i].split(';')[:-1]
                        
                        for j in range(len(values)):

                            self.output_weights[i_relative][j] = values[j]
                        i_relative += 1
                else:
                    if document[i]== "\n":
                        i_relative = 0
                    else:
                        values = document[i].split(';')[:-1]
                        for j in range(len(values)):
                            self.biases[i_relative][j] = values[j]
                        i_relative += 1

    def copy(self):
        copy_neural_network = NeuralNetwork(self.input_node_number, self.output_node_number, self.hidden_layer_node_number, self.hidden_layer_number, self.activation_function)
        copy_neural_network.input_weights = self.input_weights.copy()
        copy_neural_network.weights = self.weights.copy()
        copy_neural_network.output_weights = self.output_weights.copy()
        copy_neural_network.biases = self.biases.copy()
        copy_neural_network.activation_function = self.activation_function
        copy_neural_network.x = self.x
        copy_neural_network.y = self.y
        copy_neural_network.width = self.width
        copy_neural_network.height = self.height
        copy_neural_network.node_radius = self.node_radius
        copy_neural_network.layers_spacing = self.layers_spacing
        copy_neural_network.nodes_spacings = self.nodes_spacings.copy()

        return copy_neural_network
    
    def display(self):
        print(self.input_weights)
        print(self.weights)
        print(self.output_weights)
        print(self.biases)   
    
    def set_visualization_window(self, x = 0, y = 0, width = 600, height = 300):
        self.x, self.y, self.width, self.height = x, y, width, height

        self.node_radius = 20

        self.layers_spacing = self.width / (self.hidden_layer_number+3)

        self.nodes_spacings = [self.height/(self.input_node_number + 1)]

        for i in range(self.hidden_layer_number):
            self.nodes_spacings.append(self.height/(self.hidden_layer_node_number+1))
        self.nodes_spacings.append(self.height/(self.output_node_number + 1))


    def draw_visualization(self, screen):
        pygame.draw.rect(screen, (255,255,255), ((self.x, self.y), (self.width, self.height)),3)
        for i in range(self.input_node_number):
            node_color = self.compute_color(self.inputs[i])
            pygame.draw.circle(screen, node_color, (self.x + self.layers_spacing, self.y + self.nodes_spacings[0]*(i+1)), self.node_radius)
            for j in range(self.hidden_layer_node_number):
                pygame.draw.line(screen, node_color, (self.x + self.layers_spacing, self.y + self.nodes_spacings[0]*(i+1)), (self.x + self.layers_spacing*2, self.y + self.nodes_spacings[1]*(j+1)), max(1,int(abs(self.input_weights[j][i])*10)))
        for j in range(self.hidden_layer_number):
            for i in range(self.hidden_layer_node_number):
                node_color = self.compute_color(self.hidden_layers[j][i])
                pygame.draw.circle(screen, node_color, (self.x + self.layers_spacing*(j+2), self.y + self.nodes_spacings[j+1]*(i+1)), self.node_radius)
                if j == self.hidden_layer_number - 1:
                    for k in range(self.output_node_number):
                        pygame.draw.line(screen, node_color, (self.x + self.layers_spacing*(j+2), self.y + self.nodes_spacings[j+1]*(i+1)), (self.x + self.layers_spacing*(j+3), self.y + self.nodes_spacings[j+2]*(k+1)), max(1,int(abs(self.output_weights[k][i])*10)))
                else :
                    for k in range(self.hidden_layer_node_number):
                        pygame.draw.line(screen, node_color, (self.x + self.layers_spacing*(j+2), self.y + self.nodes_spacings[j+1]*(i+1)), (self.x + self.layers_spacing*(j+3), self.y + self.nodes_spacings[j+2]*(k+1)),max(1,int(abs(self.weights[j][k][i])*10)))
        for i in range(self.output_node_number):
            node_color = self.compute_color(self.outputs[i])
            
            pygame.draw.circle(screen, node_color, (self.x + self.layers_spacing*(self.hidden_layer_number+2), self.y + self.nodes_spacings[-1]*(i+1)), self.node_radius)

        
    def compute_color(self,value):

        
        normalized_value = atan(value)*2/pi
        
        res = [0,0,0]
        
        if normalized_value >= 0:
            res[0] = (POSITIVE_COLOR[0] - NEUTRAL_COLOR[0]) * normalized_value + NEUTRAL_COLOR[0]
            res[1] = (POSITIVE_COLOR[1] - NEUTRAL_COLOR[1]) * normalized_value + NEUTRAL_COLOR[1]
            res[2] = (POSITIVE_COLOR[2] - NEUTRAL_COLOR[2]) * normalized_value + NEUTRAL_COLOR[2]
        else:
            res[0] = (NEUTRAL_COLOR[0] - NEGATIVE_COLOR[0]) * normalized_value + NEUTRAL_COLOR[0]
            res[1] = (NEUTRAL_COLOR[1] - NEGATIVE_COLOR[1]) * normalized_value + NEUTRAL_COLOR[1]
            res[2] = (NEUTRAL_COLOR[2] - NEGATIVE_COLOR[2]) * normalized_value + NEUTRAL_COLOR[2]


        #print((int(res[0]), int(res[1]), int(res[2])))

        return (int(res[0]), int(res[1]), int(res[2]))

    def normalize_input(self):
        self.inputs = np.arctan(self.inputs)*2/pi
        #print(self.inputs)
def ReLU(x):
    if (x<=0):
        return 0
    else:
        return x

def step(x):
    if (x<=0):
        return 0
    else:
        return 1

def arctan(x):
    return np.arctan(x)*2/pi

def Id(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
