import pygame
from math import *
from neural_network import *
import numpy as np
import time


BG_COLOR = (45,45,45)

class Cart:
    def __init__(self):
        self.f = 1
    
        self.m = 1

        self.length = 5
        self.angle = 0

        self.rail_color = (110,110,110)
        self.rail_width = 0.02

        self.cart_color = (255,0,0)
        self.cart_width = 0.1
        self.cart_height = 0.05


        self.Fmax = 15
        self.u = 0
        self.F = self.Fmax*self.u
        
        self.x = self.length/2
        self.v = 0
        self.a = 0

        self.pendulum = Pendulum()
        self.pendulum.x = self.x
        self.pendulum.y = 0


        
    def reset(self):
        self.F = self.Fmax*self.u
        self.v = 0
        self.a = 0
        self.x = self.length/2
        self.pendulum.reset()


    def update(self, framerate):
        self.F = self.Fmax*self.u
        
        self.a = (self.F - self.f*self.v)/self.m
        self.v += self.a/framerate
        self.x += self.v/framerate

        if (self.x < 0):
            self.a = 0
            self.v = 0
            self.x = 0

        elif (self.x > self.length):
            self.a = 0
            self.v = 0
            self.x = self.length

        self.pendulum.update(framerate, self)

    def get_position_onscreen(self):
        return (cameraX + int(convert_pixels_to_meters(None,self.x-self.cart_width/2)), cameraY - int(convert_pixels_to_meters(None,self.cart_height/2)))


    def display(self, screen):


        pygame.draw.line(screen, self.rail_color, (cameraX, cameraY), (cameraX + int(convert_pixels_to_meters(None,self.length)), cameraY), int(convert_pixels_to_meters(None,self.rail_width)))
        position_onscreen = self.get_position_onscreen()
        pygame.draw.rect(screen, self.cart_color, (position_onscreen[0], position_onscreen[1], int(convert_pixels_to_meters(None,self.cart_width)), int(convert_pixels_to_meters(None,self.cart_height))))
        self.pendulum.display(screen)



class Pendulum:
    def __init__(self):

        self.x = 0
        self.y = 0

        self.l1 = 1
        
        
        self.theta1 = 0
        self.omega1 = 0
        self.alpha1 = 0
        
        self.m1 = 5

        self.f = 0.1

        self.x1 = self.x + self.l1*sin(self.theta1)
        self.y1 = self.y + self.l1*cos(self.theta1)

    def update(self, framerate, cart):
        self.x = cart.x
        self.alpha1 = (-9.81/self.l1*sin(self.theta1) - cart.a/self.l1*cos(self.theta1)) - self.f*self.omega1
        self.omega1 += self.alpha1/framerate
        self.theta1 += self.omega1/framerate

        self.x1 = self.x + self.l1*sin(self.theta1)
        self.y1 = self.y + self.l1*cos(self.theta1)

    def reset(self):
        self.theta1 = 0
        self.omega1 = 0
        self.alpha1 = 0

    def display(self, screen):
        x_disp = int(cameraX + convert_pixels_to_meters(None, self.x))
        y_disp = int(cameraY + convert_pixels_to_meters(None, self.y))
        x1_disp = int(cameraX + convert_pixels_to_meters(None,self.x1))
        y1_disp = int(cameraY + convert_pixels_to_meters(None,self.y1))
        
        pygame.draw.line(screen, (0, 0, 0),(x_disp, y_disp), (x1_disp, y1_disp) , int(2*scale))
        
        pygame.draw.circle(screen, (0, 0, 0),(x1_disp, y1_disp), int(3*scale))
        pygame.draw.circle(screen, (0, 0, 0),(x_disp, y_disp), int(3*scale))




class Trainer:
    def __init__(self, screen, save_folder_path, batch_size):

        self.save_folder_path = save_folder_path
        self.screen = screen

        self.screenWidth = 1080
        self.screenHeight = 720

        
        self.font = pygame.font.Font(None, 36)

        self.framerate = 60
        
        self.clock = pygame.time.Clock()


        self.display_active = True

        self.cart = Cart()
        
        self.batch_size = batch_size

        self.iteration = 1
        self.generation = 1

        self.random_coefficients =[1,0,0]

        self.generation_best = {"score": 0, "generation": 0, "neural_network": create_adapted_neural_network()}
        self.original_neural_network = create_adapted_neural_network()
        self.neural_network = create_adapted_neural_network()
        #self.original_neural_network.display()
        self.neural_network.generate_random_configuration(self.original_neural_network, self.random_coefficients[0], self.random_coefficients[1], self.random_coefficients[2])
        self.precision = pi/6
        self.score = 0
        self.time = 0
        self.testing_time = 20

        self.last_display_trigger_time = 0
        self.last_epsilon_change_time = 0


    def handling_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                

        

        keys = pygame.key.get_pressed()

        if keys[pygame.K_d] and ((time.time()-self.last_display_trigger_time) > 0.5):
            self.last_display_trigger_time = time.time()
            self.display_active = not self.display_active
            time.sleep(0.5)
        if keys[pygame.K_s]:
            self.running = False
        if keys[pygame.K_UP] and ((time.time()-self.last_epsilon_change_time) > 0.4):
            self.random_coefficients[0] = min(1, self.random_coefficients[0]+0.1)
            self.random_coefficients[1] = (1-self.random_coefficients[0])/2
            self.random_coefficients[2] = (1-self.random_coefficients[0])/2

            self.last_epsilon_change_time = time.time()

            print(f"Random Coefficients: {self.random_coefficients}")

        if keys[pygame.K_DOWN] and ((time.time()-self.last_epsilon_change_time) > 0.4):
            self.random_coefficients[0] = max(0.05, self.random_coefficients[0]-0.1)
            self.random_coefficients[1] = (1-self.random_coefficients[0])/2
            self.random_coefficients[2] = (1-self.random_coefficients[0])/2

            self.last_epsilon_change_time = time.time()

            print(f"Random Coefficients: {self.random_coefficients}")

        

    def update(self):
        # Neural Network choice

        self.neural_network.inputs = np.array([pi -self.cart.pendulum.theta1, self.cart.pendulum.omega1, self.cart.x-self.cart.length/2, self.cart.v])
        self.neural_network.predict()
        self.cart.u = self.neural_network.outputs[0]

        self.cart.update(self.framerate)
        

        if abs(abs(self.cart.pendulum.theta1) - pi) <= self.precision:
            self.score += 10

        if abs(abs(self.cart.pendulum.theta1) - pi) > pi/2 and self.time>2:
            self.reset()

        if self.cart.x == 0 or self.cart.x == self.cart.length:
            self.score = self.score /2
            self.reset()


        if (self.time > self.testing_time):
            self.generation_best = {"score": self.score, "generation": self.generation, "neural_network": self.neural_network.copy()}
            self.neural_network.save(self.save_folder_path +f"/Generation_{self.generation}_best_model.txt")
            print("Training complete")
            self.running = False
        

    def display(self):
        self.screen.fill(BG_COLOR)

        
        
        self.cart.display(self.screen)
        #self.pendulum.display(self.screen, self.cart)

        

        text_surface = self.font.render(f"Generation : {self.generation}", True, (255,255,255))
        self.screen.blit(text_surface, (10,0))

        text_surface = self.font.render(f"Iteration : {self.iteration}", True, (255,255,255))
        self.screen.blit(text_surface, (10,40))

        text_surface = self.font.render(f"Elapsed Time : {self.time:.2f}", True, (255,255,255))
        self.screen.blit(text_surface, (10,80))

        text_surface = self.font.render(f"Current score : {self.score:.2f}", True, (255,255,255))
        self.screen.blit(text_surface, (10,120))

        text_surface = self.font.render(f"Best score : {self.generation_best["score"]:.2f}", True, (255,255,255))
        self.screen.blit(text_surface, (10,160))


        self.neural_network.draw_visualization(self.screen)

        pygame.display.flip()
    def reset(self):
        
        if self.iteration ==1 :
            self.display_active = False
        

        if self.score > self.generation_best["score"]:
            self.generation_best = {"score": self.score, "generation": self.generation, "neural_network": self.neural_network.copy()}
            self.neural_network.save(self.save_folder_path +f"/Generation_{self.generation}_best_model.txt")
            print(f"Generation {self.generation} Best Score: {self.score}")
        
        self.cart.reset()
        self.time = 0
        self.score = 0
        if self.iteration == self.batch_size:
            self.iteration = 1
            self.generation += 1
            self.display_active = True
            self.random_coefficients[0] = max(0, self.random_coefficients[0]-0.05)

            self.random_coefficients[1] = (1-self.random_coefficients[0])*2/3
            self.random_coefficients[2] = (1-self.random_coefficients[0])/3

            self.original_neural_network = self.generation_best["neural_network"].copy()
            self.neural_network = self.generation_best["neural_network"].copy()

            self.generation_best = {"score":0, "neural_network": 0}
    
            print(f"\nGeneration {self.generation}\n")

            print(f"Random Coefficients: {self.random_coefficients}")
        else:
            
            self.iteration += 1
            self.neural_network.generate_random_configuration(self.original_neural_network, self.random_coefficients[0], self.random_coefficients[1], self.random_coefficients[2])

        
        
    def run(self):
        self.running = True
        while self.running:
            self.handling_event()
            self.update()
            if self.display_active:
                self.display()
                self.clock.tick(self.framerate)
            self.time += 1/self.framerate

    


class Tester:
    def __init__(self, screen, base_model):

        self.screen = screen

        self.screenWidth = 1080
        self.screenHeight = 720

        
        self.font = pygame.font.Font(None, 36)

        self.framerate = 60
        
        self.clock = pygame.time.Clock()


        self.display_active = True

        self.cart = Cart()
        self.neural_network = create_adapted_neural_network()
        self.neural_network.load(base_model)

        self.score = 0
        self.time = 0
        self.testing_time = 30

        self.precision = pi/2

        self.last_display_trigger_time = 0
        self.last_epsilon_change_time = 0


    def handling_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
        keys = pygame.key.get_pressed()

        if keys[pygame.K_s]:
            self.running = False
        
    def update(self):
        # Neural Network choice

        self.neural_network.inputs = np.array([self.cart.pendulum.theta1, self.cart.pendulum.omega1, self.cart.x-self.cart.length/2, self.cart.v])
        self.neural_network.predict()
        self.cart.u = self.neural_network.outputs[0]

        self.cart.update(self.framerate)
        

        if abs(abs(self.cart.pendulum.theta1) - pi) <= self.precision:
            self.score += 10

        if abs(abs(self.cart.pendulum.theta1) - pi) > pi/2 and self.time>2:
            self.reset()

        if self.cart.x == 0 or self.cart.x == self.cart.length:
            self.score = self.score /2
            self.reset()

        if (self.time>= self.testing_time):
            self.reset()
        

    def display(self):
        self.screen.fill(BG_COLOR)
        self.cart.display(self.screen)

        text_surface = self.font.render(f"Elapsed Time : {self.time:.2f}", True, (255,255,255))
        self.screen.blit(text_surface, (10,0))

        text_surface = self.font.render(f"Current score : {self.score:.2f}", True, (255,255,255))
        self.screen.blit(text_surface, (10,40))


        self.neural_network.draw_visualization(self.screen)

        pygame.display.flip()
    def reset(self):
        
        

                
        self.cart.reset()
        self.time = 0
        self.score = 0
        
    def run(self):
        self.running = True
        while self.running:
            self.handling_event()
            self.update()
            if self.display_active:
                self.display()
                self.clock.tick(self.framerate)
            self.time += 1/self.framerate

    


# class Agent:
#     def __init__(self):
        

#         self.screenWidth = 1080
#         self.screenHeight = 720

#         self.best_neural_network = create_adapted_neural_network()
#         self.best_score = -1

        
#         self.testing_time = 30 #seconds
#         self.testing_precision = pi/2
#         self.number_of_tests_by_generation = 5000
#         self.generation = 0
#         self.random_coefficients =[1, 0,0]

#         pygame.init()
#         self.screen = pygame.display.set_mode((self.screenWidth, self.screenHeight))
#         self.clock = pygame.time.Clock()

#         self.best_number = 1

#         self.display_active = True


#     def train(self, filepath, number, framerate):

#         if (filepath != None):
#             self.current_neural_network.load("saves/best_"+str(number)+".csv")
#             self.best_neural_network = self.current_neural_network
#             simulation = Simulation(self.testing_precision, self.testing_time, self.screen, self.clock, self.display_active, framerate)

#             simulation.neural_network = self.current_neural_network
#             self.current_score, self.display_active, self.training_stopped = simulation.run()
#             self.best_score = self.current_score
#             self.best_number = number + 1

#         while not self.training_stopped:
#             print("Generation :" + str(self.generation)+"\n")
#             print("Random coefficients : " + str(self.random_coefficients))

#             generation_best_neural_network = create_adapted_neural_network()
#             generation_best_score = -1
#             temp_score = 0
#             temp_neural_network = create_adapted_neural_network()
            
#             for i in range(self.number_of_tests_by_generation):
#                 if (not self.training_stopped):
#                     if (i==0):
#                         temp_neural_network = self.best_neural_network.copy()
#                         self.best_neural_network.display()
#                         temp_neural_network.display()
                
#                     else:
#                         temp_neural_network.generate_random_configuration(self.best_neural_network, self.random_coefficients[0], self.random_coefficients[1], self.random_coefficients[2])

#                     simulation = Simulation(self.testing_precision, self.testing_time, self.screen, self.clock, self.display_active, framerate)
#                     # self.current_neural_network = create_adapted_neural_network()
#                     # self.current_neural_network.generate_random_configuration(self.best_neural_network, self.random_coefficients[0], self.random_coefficients[1], self.random_coefficients[2])
                
#                     simulation.neural_network = temp_neural_network
#                     temp_score, self.display_active, self.training_stopped = simulation.run()

#                     if temp_score > generation_best_score:
#                         temp_neural_network.save("saves/generation_"+str(self.generation)+"_best.csv")
#                         generation_best_score = temp_score
#                         generation_best_neural_network = temp_neural_network
#                         print("New best score: " + str(generation_best_score))

#             if (generation_best_score > self.best_score):
#                 self.best_neural_network=generation_best_neural_network
#                 self.best_score = generation_best_score

#             self.generation += 1

#             self.random_coefficients = [0,0,1]
#             # self.random_coefficients[0] = max(0, self.random_coefficients[0]-0.1)
#             # self.random_coefficients[1] = min(0.05, self.random_coefficients[1]+0.005)
#             # self.random_coefficients[2] = 1 - self.random_coefficients[0] - self.random_coefficients[1]

#         pygame.quit()


#     def test_best(self, best_number):
        
        
#         simulation = Simulation(self.testing_precision, self.testing_time, self.screen, self.clock, self.display_active, 60)
#         self.current_neural_network.load("saves/best_"+str(best_number)+".csv")
#         simulation.neural_network = self.current_neural_network
#         self.score = simulation.run()
#         print(self.score)
        


    
    


def convert_pixels_to_meters(pixels, meters):
    # 200 pixels equals 1 meter

    if (meters == None):
        return pixels / (100*scale)

    else:
        return meters * (100*scale)


def create_adapted_neural_network():
    res = NeuralNetwork(4, 1, 3 ,1, arctan)
    res.set_visualization_window(280,0,800, 300)
    return res



scale = 2
cameraX = 40
cameraY = 600


#agent.test_best(17)


# test = create_adapted_neural_network()
# test.load("saves/best_1.csv")
# print(test.biases)
