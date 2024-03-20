#train a large s-x randomly generated space.
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.mplot3d import Axes3D
#import pandas as pd
#import h5py

#functions 


def test_function(x1, x2, x3, mH2, s12, s14):
    mt2 = 1
    F1 = mt2 + 2*mt2*(x1 + x2 + x3) + mt2*(x1**2 + x2**2 + x3**2) + 2*mt2*(x2*x3 + x1*x3 + x1*x2) - mH2*(x1*x2 + x2*x3) - x2*s14 - x1*x3*s12
    # return x1**2 + x2*x1
    return 1/F1**2

def new_test_function(t1, t2, t3, mH2, s12, s14):
    mt2 = 1
    x1 = (3-2*t1)*t1**2
    x2 = (3-2*t2)*t2**2
    x3 = (3-2*t3)*t3**2

    F1 = mt2 + 2*mt2*(x1 + x2 + x3) + mt2*(x1**2 + x2**2 + x3**2) + 2*mt2*(x2*x3 + x1*x3 + x1*x2) - mH2*(x1*x2 + x2*x3) - x2*s14 - x1*x3*s12
    # return x1**2 + x2*x1

    w1 = 6*t1*(1-t1)
    w2 = 6*t2*(1-t2)
    w3 = 6*t3*(1-t3)

    return w1*w2*w3/F1**2

def normalised_test_function(x1, x2, x3, mH2, s12, s14):
    normalised_value = new_test_function(x1, x2, x3, mH2, s12, s14)/test_function(0.5, 0.5, 0.5, mH2, s12, s14)
    return normalised_value

def dummy_function2(t1, t2, t3, mH2, s12, s14):
    x1 = (3-2*t1)*t1**2
    x2 = (3-2*t2)*t2**2
    x3 = (3-2*t3)*t3**2

    
    w1 = 6*t1*(1-t1)
    w2 = 6*t2*(1-t2)
    w3 = 6*t3*(1-t3)

    numerator = s12*s14*x1*x2 + mH2*x3 
    denominator = 0.25*s12*s14 + 0.5*mH2

    dummy_function_tilda = numerator/denominator


    return w1*w2*w3*(dummy_function_tilda)#normalised_value

def dummy_function(x1,x2,x3,mH2,s12,s14):
    return x1*x2*x3*mH2*s12*s14


#2.5
def xavier_initialisation(size_going_in, size_going_out, xavier_gain2):
            return xavier_gain2*np.sqrt(6.0/(size_going_in + size_going_out)) #maybe add a gain

class NeuralNetwork(nn.Module):
    def __init__(self, number_of_hidden_layers, number_of_auxillary_variable, number_of_phase_space_parameters, hidden_layer_size,
                  output_layer_size, activation_function, batch_size, normalisation_coefficient, xavier_gain):
        super(NeuralNetwork, self).__init__()
        
        self.activation_function = activation_function
        self.batch_size = batch_size
        self.normalisation_coefficient = normalisation_coefficient

        self.number_of_hidden_layers = number_of_hidden_layers   
        self.number_of_auxillary_variable = number_of_auxillary_variable  
        self.number_of_phase_space_parameters = number_of_phase_space_parameters
        self.input_layer_size = self.number_of_auxillary_variable + self.number_of_phase_space_parameters
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.xavier_gain = xavier_gain
        
        
        

        #torch.device('cpu')#'cuda')
        
        

        #xavier init
        xavier_init_in_limit = xavier_initialisation(self.input_layer_size, self.hidden_layer_size, xavier_gain)
        xavier_init_hidden_limit = xavier_initialisation(self.hidden_layer_size, self.hidden_layer_size, xavier_gain)
        xavier_init_out_limit = xavier_initialisation(self.hidden_layer_size, self.output_layer_size, xavier_gain)

       # calcultae the std of for each network param
        with torch.no_grad():
           
            print(xavier_init_in_limit)
            print(xavier_init_hidden_limit)
            print(xavier_init_out_limit)

       
        #going to define a rank 3 weights tensor, and will make bias a rank 2 tensor 
        #first need the input to hidden layer, hidden layer-hidden layer and then hidden layer to output weights and il stack them
        self.weights_input_hidden = nn.Parameter(torch.from_numpy(np.random.uniform(low = -xavier_init_in_limit, high = xavier_init_in_limit, size = (self.input_layer_size, self.hidden_layer_size))).to(torch.float64))
        self.weights_hidden_hidden = nn.Parameter(torch.from_numpy(np.random.uniform(low = -xavier_init_hidden_limit, high = xavier_init_hidden_limit, size = (self.number_of_hidden_layers-1, self.hidden_layer_size, self.hidden_layer_size))).to(torch.float64))
        self.weights_hidden_output = nn.Parameter(torch.from_numpy(np.random.uniform(low = -xavier_init_out_limit, high = xavier_init_out_limit, size = (self.hidden_layer_size, self.output_layer_size))).to(torch.float64))
        self.bias_hidden = nn.Parameter(torch.zeros(self.number_of_hidden_layers+1, self.hidden_layer_size).to(torch.float64))
        #self.bias_hidden = nn.Parameter(torch.zeros(self.number_of_hidden_layers+1, self.hidden_layer_size).to(torch.float64).to(device))
        self.bias_output = nn.Parameter(torch.zeros(1, 1).to(torch.float64))
        
        #
        
        #bias from hidden to output layer doesn't get trained
        #print(self.weights_hidden_hidden[0].shape) is hidden1 to hidden2
        #self.bias_output = nn.Parameter(torch.zeros(self.number_of_hidden_layers+1, self.hidden_layer_size).to(torch.float64).to(device))

        

    

     #   with torch.no_grad():
      #  #turn matrix into a vector and get a flat distitbution 1D
       #     plt.hist(self.weights_input_hidden.view(-1).cpu(), bins = 5)
        #    plt.show()
         #   for i in range(0, self.number_of_hidden_layers-1):
          #      plt.hist(self.weights_hidden_hidden[i].cpu(), bins = 10)
           #     plt.show()
       #     plt.hist(self.weights_hidden_output.view(-1).cpu(), bins = 5)
        #    plt.show() 
       
          
        #save model once training
        #hdf5 files save my model, 

    def activation_function_normal(self, x):
        if self.activation_function == "tanh":
            activation_function_value = torch.tanh(x)

        if self.activation_function == "sigmoid":
            activation_function_value = torch.sigmoid(x)

        if self.activation_function == "GELU":
            a_value = 0.044715
            b_value = np.sqrt(2/np.pi)
            activation_function_value = 0.5*x*(1 + torch.tanh(b_value*(x + a_value*x**3)))

        return activation_function_value
    
    def activation_function_first_derivative(self, x):
        if self.activation_function == "tanh":
            tanh = torch.tanh(x)
            y_first_derivative = (1-tanh)*(1+tanh)

        if self.activation_function == "sigmoid":
            sigmoid = torch.sigmoid(x)
            y_first_derivative = sigmoid*(1-sigmoid)

        if self.activation_function == "GELU":
            a_value = 0.044715
            b_value = np.sqrt(2/np.pi)
            argument = b_value* (x + a_value*x**3)
            sech = 1 / torch.cosh(argument)
            derivative_tanh = b_value*(1+3*a_value*x**2)*sech**2
            y_first_derivative = 0.5 + 0.5*torch.tanh(argument) + 0.5*x*derivative_tanh
    

        return y_first_derivative
    
    def activation_function_second_derivative(self, x):
        if self.activation_function == "tanh":
            t = torch.tanh(x)
            y_second_derivative = -2*(1-t)*(1+t)*t

        if self.activation_function == "sigmoid":
            sigmoid_x = torch.sigmoid(x)
            y_second_derivative = sigmoid_x * (1 - sigmoid_x)*(1-2*sigmoid_x)
        
        if self.activation_function == "GELU":
            a_value = 0.044715
            b_value = np.sqrt(2/np.pi)
            argument = b_value* (x + a_value*x**3)
            sech = 1 / torch.cosh(argument)
            derivative_tanh = b_value*(1+3*a_value*x**2)*sech**2
            derivative_sech_squared = -2*b_value*(1+3*a_value*x**2)*torch.tanh(argument)*sech**2
            second_derivative_tanh = b_value*(6*a_value*x)*sech**2 + b_value*(1+3*a_value*x**2)*derivative_sech_squared
            y_second_derivative = derivative_tanh + 0.5*x*second_derivative_tanh


        return y_second_derivative
    
    def activation_function_third_derivative(self, x):
        if self.activation_function == "tanh":
            t = torch.tanh(x)
            y_third_derivative = -2*(1-t)*(1+t)*(1-3*t**2)

        if self.activation_function == "sigmoid":
            sigmoid_x = torch.sigmoid(x)
            y_third_derivative = sigmoid_x * (1 - sigmoid_x)*(1-6*sigmoid_x + 6*sigmoid_x**2)

        if self.activation_function == "GELU":
            a_value = 0.044715
            b_value = np.sqrt(2/np.pi)
            argument = b_value* (x + a_value*x**3)
            sech = 1 / torch.cosh(argument)
            derivative_tanh = b_value*(1+3*a_value*x**2)*sech**2
            derivative_sech_squared = -2*b_value*(1+3*a_value*x**2)*torch.tanh(argument)*sech**2
            second_derivative_tanh = b_value*(6*a_value*x)*sech**2 + b_value*(1+3*a_value*x**2)*derivative_sech_squared
            second_derivative_sech_squared = -2*b_value*torch.tanh(argument)*(1+3*a_value*x**2)*derivative_sech_squared -2*b_value*(1+3*a_value**2)*derivative_tanh*sech**2 -2*b_value*torch.tanh(argument)*(6*a_value*x)*sech**2
            third_derivative_tanh = b_value*6*a_value*sech**2 + 2*b_value*6*a_value*x*derivative_sech_squared + b_value*(1+3*a_value*x**2)*second_derivative_sech_squared

            y_third_derivative = 1.5*second_derivative_tanh + 0.5*x*third_derivative_tanh


        return y_third_derivative
    

    #def derivative(self, x, s):#
    def derivative(self, s_x_params):
        

          
      

        ai_0 = s_x_params
        #print(f'ai_0: {ai_0[1:3]}') 
        ai_0.requires_grad_()

      #  ai_0_normal = self.batch_normalisation(ai_0)
       # ai_0_normal.requires_grad_()
        #print(f'ai_0 after batch: {ai_0[1:3]}') 
       
        
       # print(f'ai_0: {ai_0.shape}') 
#      
        
    
        wij_1 = self.weights_input_hidden
        
        wij_1.requires_grad_()
        #print(f'wij_1: {wij_1.shape}') 
        bi_1 = self.bias_hidden[0].unsqueeze(1)
       # print(f'bi_1: {bi_1.shape}') 
    
        zi_1= torch.matmul(wij_1.T, ai_0.T) + bi_1
        #print(f'torch.matmul(wij_1.T, ai_0.T): {torch.matmul(wij_1.T, ai_0.T).shape}')
        #print(f'zi_1: {zi_1.shape}')#, {zi_1[0,1:6]}')
     
       
        #print(f'self.activation_function_first_derivative(zi_1): {self.activation_function_first_derivative(zi_1).shape}')
        #print(f'wij_1[0].unsqueeze(1): {wij_1[0].unsqueeze(1).shape}')
        dai_1_dx1 = torch.mul(self.activation_function_first_derivative(zi_1),wij_1[0].unsqueeze(1)) #torch.Size([10000, 50])
        dai_1_dx2 = torch.mul(self.activation_function_first_derivative(zi_1),wij_1[1].unsqueeze(1))
        dai_1_dx3 = torch.mul(self.activation_function_first_derivative(zi_1),wij_1[2].unsqueeze(1))
        #print(f'dai_1_dx1: {dai_1_dx1.shape}') #torch.Size([27, 17])
        #print(f'dai_1_dx2: {dai_1_dx2.shape}')
        #print(f'dai_1_dx3: {dai_1_dx3.shape}')
        d2a_1_dx1dx2 = self.activation_function_second_derivative(zi_1)*wij_1[0].unsqueeze(1)*wij_1[1].unsqueeze(1)
        #print(f'd2a_1_dx1dx2: {d2a_1_dx1dx2.shape}')
        d2a_1_dx1dx3 = self.activation_function_second_derivative(zi_1)*wij_1[0].unsqueeze(1)*wij_1[2].unsqueeze(1)
        d2a_1_dx2dx3 = self.activation_function_second_derivative(zi_1)*wij_1[1].unsqueeze(1)*wij_1[2].unsqueeze(1)
        d3a_1_dx1dx2dx3 = self.activation_function_third_derivative(zi_1)*wij_1[0].unsqueeze(1)*wij_1[1].unsqueeze(1)*wij_1[2].unsqueeze(1)
        #print(f'd3a_1_dx1dx2dx3: {d3a_1_dx1dx2dx3.shape}')
       

        ai_1 = self.activation_function_normal(zi_1)#.view(self.hidden_layer_size, 1)
        #print(f'ai_1 : {ai_1.shape}') #torch.Size([27, 17])
        #print(ai_0)
        #print(self.activation_function_normal(zi_1))
        
     
        
        ai_m_minus_1 = ai_1 

        zi_m = self.bias_hidden[1]  #check this
        #print(f'zi_m bias part: {zi_m.shape}')
        #print(f'zi_m.unsqueeze(1): {zi_m.unsqueeze(1).shape}')



        dai_m_minus_1_dx1 = dai_1_dx1
        dai_m_minus_1_dx2 = dai_1_dx2
        dai_m_minus_1_dx3 = dai_1_dx3
        d2ai_m_minus_1_dx1dx2 = d2a_1_dx1dx2
        d2ai_m_minus_1_dx2dx3 = d2a_1_dx2dx3
        d2ai_m_minus_1_dx1dx3 = d2a_1_dx1dx3
        d3ai_m_minus_1_dx1dx2dx3 = d3a_1_dx1dx2dx3
        #print(f'dai_m_minus_1_dx1: {dai_m_minus_1_dx1.shape}')


      
        for m in range(2, self.number_of_hidden_layers+1):
            weights_between_hidden_layers = self.weights_hidden_hidden[m-2] #checkl this it is a square matrix
      #      print(f'weights: {weights_between_hidden_layers.shape}')
            #is it weights_between_hidden_layers or weights_between_hidden_layers.T
       #     print(f' torch.matmul(weights_between_hidden_layers , ai_m_minus_1): { torch.matmul(weights_between_hidden_layers , ai_m_minus_1).shape }')
            zi_m = zi_m.unsqueeze(1) + torch.matmul(weights_between_hidden_layers.T , ai_m_minus_1) #torch.Size([27, 17])
         #   print(f'zi_m in training : {zi_m.shape}') #torch.Size([27, 17])
           # zi_m = zi_m + (ai_m_minus_1 @ weights_between_hidden_layers )

            dzi_m_dx1 = torch.matmul(weights_between_hidden_layers.T , dai_m_minus_1_dx1)
            dzi_m_dx2 = torch.matmul(weights_between_hidden_layers.T , dai_m_minus_1_dx2)
            dzi_m_dx3 = torch.matmul(weights_between_hidden_layers.T , dai_m_minus_1_dx3)
        
           # print(f'dzi_m_dx1: {dzi_m_dx1.shape}')
        
            d2zi_m_dx1dx2 = torch.matmul(weights_between_hidden_layers.T , d2ai_m_minus_1_dx1dx2)
            d2zi_m_dx1dx3 = torch.matmul(weights_between_hidden_layers.T, d2ai_m_minus_1_dx1dx3)
            d2zi_m_dx2dx3 = torch.matmul(weights_between_hidden_layers.T , d2ai_m_minus_1_dx2dx3)

            d3zi_m_dx1dx2dx3 = torch.matmul(weights_between_hidden_layers.T , d3ai_m_minus_1_dx1dx2dx3)

      #      print(f'd2zi_m_dx1dx2: {d2zi_m_dx1dx2.shape}')
       #     print(f'd3zi_m_dx1dx2dx3: {d3zi_m_dx1dx2dx3.shape}')
            ai_m = self.activation_function_normal(zi_m) #use a different activation fucntion like tanh symmetric
          #  print(f'ai_m iter{m}: {ai_m.shape}')#torch.Size([27, 17])
            
            dai_m_dx1 = self.activation_function_first_derivative(zi_m)*dzi_m_dx1#dzi_m_dx1[m-2] 
           # print(f'self.activation_function_first_derivative(zi_m): {self.activation_function_first_derivative(zi_m).shape}')
            #print(f'dzi_m_dx2: {dzi_m_dx2.shape}')
            dai_m_dx2 = self.activation_function_first_derivative(zi_m)*dzi_m_dx2#*dzi_m_dx2[m-2]        
            dai_m_dx3 = self.activation_function_first_derivative(zi_m)*dzi_m_dx3#*dzi_m_dx3[m-2]
           # print(f'dai_m_dx1: {dai_m_dx1.shape}')#torch.Size([27, 17])
            d2ai_m_dx1dx2 = self.activation_function_second_derivative(zi_m)*dzi_m_dx1*dzi_m_dx2 + self.activation_function_first_derivative(zi_m)*d2zi_m_dx1dx2
            d2ai_m_dx1dx3 = self.activation_function_second_derivative(zi_m)*dzi_m_dx1*dzi_m_dx3 + self.activation_function_first_derivative(zi_m)*d2zi_m_dx1dx3          
            d2ai_m_dx2dx3 = self.activation_function_second_derivative(zi_m)*dzi_m_dx2*dzi_m_dx3 + self.activation_function_first_derivative(zi_m)*d2zi_m_dx2dx3     
          
            d3ai_m_dx1dx2dx3 = self.activation_function_third_derivative(zi_m)*dzi_m_dx1*dzi_m_dx2*dzi_m_dx3 + self.activation_function_second_derivative(zi_m)*(dzi_m_dx1*d2zi_m_dx2dx3 +  dzi_m_dx2*d2zi_m_dx1dx3 +  dzi_m_dx3*d2zi_m_dx1dx2) + self.activation_function_first_derivative(zi_m)*d3zi_m_dx1dx2dx3
         #   print(f'd3ai_m_dx1dx2dx3:  {d3ai_m_dx1dx2dx3.shape}')#torch.Size([27, 17])


            ai_m_minus_1 = ai_m   
            dai_m_minus_1_dx1 = dai_m_dx1
            dai_m_minus_1_dx2 = dai_m_dx2
            dai_m_minus_1_dx3 = dai_m_dx3
        
            d2ai_m_minus_1_dx1dx2 = d2ai_m_dx1dx2
            d2ai_m_minus_1_dx1dx3 = d2ai_m_dx1dx3
            d2ai_m_minus_1_dx2dx3 = d2ai_m_dx2dx3
            d3ai_m_minus_1_dx1dx2dx3 = d3ai_m_dx1dx2dx3
           # print(f'ai_m_minus_1: {ai_m_minus_1.shape}')
            
            zi_m = self.bias_hidden[m]
          #  print(f'zi_m bias part: {zi_m.shape}')
           # print(f' zi_m at end of training: {zi_m.shape}')
          #  print(torch.mean(d3ai_m_minus_1_dx1dx2dx3))
                #is this a m--1]
            #print(d3ai_m_minus_1_dx1dx2dx3)
      
        #print(d3ai_m_minus_1_dx1dx2dx3.shape)
        #print(self.weights_hidden_output.shape)
       # print(f'weights hidden to output: {self.weights_hidden_output.shape}')
        #print(f'd3ai_m_minus_1_dx1dx2dx3 at training end:  {d3ai_m_minus_1_dx1dx2dx3.shape}')
        
        d3y_dx1dx2dx3 = torch.matmul(self.weights_hidden_output.T , d3ai_m_minus_1_dx1dx2dx3)
        #print(f'd3y_dx1dx2dx3: {d3y_dx1dx2dx3.shape}')
        
        return d3y_dx1dx2dx3
        #return self.batch_unnormalisation(d3y_dx1dx2dx3)
    


    
    def activation_value_collector(self, s_x_params):#

      

        ai_0 = s_x_params
        ai_0.requires_grad_()


        
        #print(f'ai_0: {ai_0.shape}') 
      #  ai_0_normal = self.batch_normalisation(ai_0)
#      
       # ai_0_normal.requires_grad_()

        with torch.no_grad():
            ai_j = []
            d3ai_j_dx3 = []
            ai_j_std = []
            d3ai_j_dx3_std = []

          
    
        wij_1 = self.weights_input_hidden
        
        wij_1.requires_grad_()
        #print(f'wij_1: {wij_1.shape}') 
        bi_1 = self.bias_hidden[0].unsqueeze(1)
        #(f'bi_1: {bi_1.shape}') 
    
        zi_1= torch.matmul(wij_1.T, ai_0.T) + bi_1
        #print(f'torch.matmul(wij_1.T, ai_0.T): {torch.matmul(wij_1.T, ai_0.T)[:,0].shape}')
        #print(f'zi_1: {zi_1.shape}')#, {zi_1[0,1:6]}')
     
       
    #    print(f'self.activation_function_first_derivative(zi_1): {self.activation_function_first_derivative(zi_1).shape}')
    #    print(f'wij_1[0].unsqueeze(1): {wij_1[0].unsqueeze(1).shape}')
        dai_1_dx1 = torch.mul(self.activation_function_first_derivative(zi_1),wij_1[0].unsqueeze(1)) #torch.Size([10000, 50])
        dai_1_dx2 = torch.mul(self.activation_function_first_derivative(zi_1),wij_1[1].unsqueeze(1))
        dai_1_dx3 = torch.mul(self.activation_function_first_derivative(zi_1),wij_1[2].unsqueeze(1))
  #      print(f'dai_1_dx1: {dai_1_dx1.shape}') #torch.Size([27, 17])
   #     print(f'dai_1_dx2: {dai_1_dx2.shape}')
    #    print(f'dai_1_dx3: {dai_1_dx3.shape}')
        d2a_1_dx1dx2 = self.activation_function_second_derivative(zi_1)*wij_1[0].unsqueeze(1)*wij_1[1].unsqueeze(1)
     #   print(f'd2a_1_dx1dx2: {d2a_1_dx1dx2.shape}')
        d2a_1_dx1dx3 = self.activation_function_second_derivative(zi_1)*wij_1[0].unsqueeze(1)*wij_1[2].unsqueeze(1)
        d2a_1_dx2dx3 = self.activation_function_second_derivative(zi_1)*wij_1[1].unsqueeze(1)*wij_1[2].unsqueeze(1)
        d3a_1_dx1dx2dx3 = self.activation_function_third_derivative(zi_1)*wij_1[0].unsqueeze(1)*wij_1[1].unsqueeze(1)*wij_1[2].unsqueeze(1)
      #  print(f'd3a_1_dx1dx2dx3: {d3a_1_dx1dx2dx3.shape}')
       

        ai_1 = self.activation_function_normal(zi_1)#.view(self.hidden_layer_size, 1)
       # print(f'ai_1 : {ai_1.shape}') #torch.Size([27, 17])
        #print(ai_0)
       # print(self.activation_function_normal(zi_1))
        
        with torch.no_grad():
            ai_j.append(torch.mean(ai_1))#/len(ai_1)
            d3ai_j_dx3.append(torch.mean(d3a_1_dx1dx2dx3))
            ai_j_std.append(torch.std(ai_1))#/len(ai_1)
            d3ai_j_dx3_std.append(torch.std(d3a_1_dx1dx2dx3))
       

            
        
        ai_m_minus_1 = ai_1 

        zi_m = self.bias_hidden[1]  #check this
   #     print(f'zi_m bias part: {zi_m.shape}')
    #    print(f'zi_m.unsqueeze(1): {zi_m.unsqueeze(1).shape}')
        


        dai_m_minus_1_dx1 = dai_1_dx1
        dai_m_minus_1_dx2 = dai_1_dx2
        dai_m_minus_1_dx3 = dai_1_dx3
        d2ai_m_minus_1_dx1dx2 = d2a_1_dx1dx2
        d2ai_m_minus_1_dx2dx3 = d2a_1_dx2dx3
        d2ai_m_minus_1_dx1dx3 = d2a_1_dx1dx3
        d3ai_m_minus_1_dx1dx2dx3 = d3a_1_dx1dx2dx3
   #     print(f'dai_m_minus_1_dx1: {dai_m_minus_1_dx1.shape}')


      
        for m in range(2, self.number_of_hidden_layers+1):
            weights_between_hidden_layers = self.weights_hidden_hidden[m-2] #checkl this it is a square matrix
      #      print(f'weights: {weights_between_hidden_layers.shape}')
            #is it weights_between_hidden_layers or weights_between_hidden_layers.T
       #     print(f' torch.matmul(weights_between_hidden_layers , ai_m_minus_1): { torch.matmul(weights_between_hidden_layers , ai_m_minus_1).shape }')
            zi_m = zi_m.unsqueeze(1) + torch.matmul(weights_between_hidden_layers.T , ai_m_minus_1) #torch.Size([27, 17])
        #    print(f'zi_m in training : {zi_m.shape}') #torch.Size([27, 17])
           # zi_m = zi_m + (ai_m_minus_1 @ weights_between_hidden_layers )

            dzi_m_dx1 = torch.matmul(weights_between_hidden_layers.T , dai_m_minus_1_dx1)
            dzi_m_dx2 = torch.matmul(weights_between_hidden_layers.T , dai_m_minus_1_dx2)
            dzi_m_dx3 = torch.matmul(weights_between_hidden_layers.T , dai_m_minus_1_dx3)
        
           # print(f'dzi_m_dx1: {dzi_m_dx1.shape}')
        
            d2zi_m_dx1dx2 = torch.matmul(weights_between_hidden_layers.T , d2ai_m_minus_1_dx1dx2)
            d2zi_m_dx1dx3 = torch.matmul(weights_between_hidden_layers.T, d2ai_m_minus_1_dx1dx3)
            d2zi_m_dx2dx3 = torch.matmul(weights_between_hidden_layers.T , d2ai_m_minus_1_dx2dx3)

            d3zi_m_dx1dx2dx3 = torch.matmul(weights_between_hidden_layers.T , d3ai_m_minus_1_dx1dx2dx3)

      #      print(f'd2zi_m_dx1dx2: {d2zi_m_dx1dx2.shape}')
       #     print(f'd3zi_m_dx1dx2dx3: {d3zi_m_dx1dx2dx3.shape}')
            ai_m = self.activation_function_normal(zi_m) #use a different activation fucntion like tanh symmetric
        #    print(f'ai_m iter{m}: {ai_m.shape}')#torch.Size([27, 17])
            
            dai_m_dx1 = self.activation_function_first_derivative(zi_m)*dzi_m_dx1#dzi_m_dx1[m-2] 
           # print(f'self.activation_function_first_derivative(zi_m): {self.activation_function_first_derivative(zi_m).shape}')
            #print(f'dzi_m_dx2: {dzi_m_dx2.shape}')
            dai_m_dx2 = self.activation_function_first_derivative(zi_m)*dzi_m_dx2#*dzi_m_dx2[m-2]        
            dai_m_dx3 = self.activation_function_first_derivative(zi_m)*dzi_m_dx3#*dzi_m_dx3[m-2]
           # print(f'dai_m_dx1: {dai_m_dx1.shape}')#torch.Size([27, 17])
            d2ai_m_dx1dx2 = self.activation_function_second_derivative(zi_m)*dzi_m_dx1*dzi_m_dx2 + self.activation_function_first_derivative(zi_m)*d2zi_m_dx1dx2
            d2ai_m_dx1dx3 = self.activation_function_second_derivative(zi_m)*dzi_m_dx1*dzi_m_dx3 + self.activation_function_first_derivative(zi_m)*d2zi_m_dx1dx3          
            d2ai_m_dx2dx3 = self.activation_function_second_derivative(zi_m)*dzi_m_dx2*dzi_m_dx3 + self.activation_function_first_derivative(zi_m)*d2zi_m_dx2dx3     
          
            d3ai_m_dx1dx2dx3 = self.activation_function_third_derivative(zi_m)*dzi_m_dx1*dzi_m_dx2*dzi_m_dx3 + self.activation_function_second_derivative(zi_m)*(dzi_m_dx1*d2zi_m_dx2dx3 +  dzi_m_dx2*d2zi_m_dx1dx3 +  dzi_m_dx3*d2zi_m_dx1dx2) + self.activation_function_first_derivative(zi_m)*d3zi_m_dx1dx2dx3
         #   print(f'd3ai_m_dx1dx2dx3:  {d3ai_m_dx1dx2dx3.shape}')#torch.Size([27, 17])


            ai_m_minus_1 = ai_m   
            dai_m_minus_1_dx1 = dai_m_dx1
            dai_m_minus_1_dx2 = dai_m_dx2
            dai_m_minus_1_dx3 = dai_m_dx3
        
            d2ai_m_minus_1_dx1dx2 = d2ai_m_dx1dx2
            d2ai_m_minus_1_dx1dx3 = d2ai_m_dx1dx3
            d2ai_m_minus_1_dx2dx3 = d2ai_m_dx2dx3
            d3ai_m_minus_1_dx1dx2dx3 = d3ai_m_dx1dx2dx3
            
            zi_m = self.bias_hidden[m]
          #  print(f'zi_m bias part: {zi_m.shape}')
           # print(f' zi_m at end of training: {zi_m.shape}')
          #  print(torch.mean(d3ai_m_minus_1_dx1dx2dx3))
                #is this a m--1]
            #print(d3ai_m_minus_1_dx1dx2dx3)
            with torch.no_grad():
                ai_j.append(torch.mean(ai_m))
                d3ai_j_dx3.append(torch.mean(d3ai_m_minus_1_dx1dx2dx3))
                ai_j_std.append(torch.std(ai_m))
                d3ai_j_dx3_std.append(torch.std(d3ai_m_minus_1_dx1dx2dx3))
            


        
            
        

        return [ai_j, d3ai_j_dx3, ai_j_std, d3ai_j_dx3_std]#, weight_array, bias_array]
        
    
    
   
    def forward(self, x_s): #plotting
       

        ai_0 = x_s
#      
        ai_0.requires_grad_()
        #print(f'ai_0 : {ai_0}') #torch.Size([27, 17])

        #maybe remove this
        #ai_0 = self.batch_normalisation(ai_0)
        #print(f'ai_0_normal: {ai_0_normal}')
        #ai_0.requires_grad_()
        wij_1 = self.weights_input_hidden
        
        wij_1.requires_grad_()
        #print(f'wij_1: {wij_1.shape}') 
        bi_1 = self.bias_hidden[0].unsqueeze(1)
        #print(f'bi_1: {bi_1.shape}') 
    
        zi_1= torch.matmul(wij_1.T, ai_0.T) + bi_1
      
        ai_1 = self.activation_function_normal(zi_1)#.view(self.hidden_layer_size, 1)
       # print(f'ai_1 : {ai_1.shape}') #torch.Size([27, 17])
        #print(ai_0)
       # print(self.activation_function_normal(zi_1))
        
     
        
        ai_m_minus_1 = ai_1 

        zi_m = self.bias_hidden[1]  #check this
  


      
        for m in range(2, self.number_of_hidden_layers+1):
            weights_between_hidden_layers = self.weights_hidden_hidden[m-2] #checkl this it is a square matrix
      #      print(f'weights: {weights_between_hidden_layers.shape}')
            #is it weights_between_hidden_layers or weights_between_hidden_layers.T
       #     print(f' torch.matmul(weights_between_hidden_layers , ai_m_minus_1): { torch.matmul(weights_between_hidden_layers , ai_m_minus_1).shape }')
            zi_m = zi_m.unsqueeze(1) + torch.matmul(weights_between_hidden_layers.T , ai_m_minus_1) #torch.Size([27, 17])
        
            ai_m = self.activation_function_normal(zi_m) #use a different activation fucntion like tanh symmetric
        #    print(f'ai_m iter{m}: {ai_m.shape}')#torch.Size([27, 17])
            
        

            ai_m_minus_1 = ai_m   
            
            zi_m = self.bias_hidden[m]
          #  print(f'zi_m bias part: {zi_m.shape}')
        Y_output = torch.matmul(self.weights_hidden_output.T , ai_m_minus_1) + self.bias_output
        
    #    print(f'self.weights_hidden_output.T: {self.weights_hidden_output.T.shape}') 
     #   print(f'self.bias_output: {self.bias_output.shape}')
      
        
        #print(f'torch.matmul(self.weights_hidden_output.T , ai_m_minus_1).T: {torch.matmul(self.weights_hidden_output.T , ai_m_minus_1).T[10:13]}')
      #  print(f'ai_m_minus_1:  {ai_m_minus_1.shape}')
       # print(f'Y_output.shape: {Y_output.shape}')
        #print(f'Y_output.T: {Y_output.T.shape}')
        
        #return self.batch_unnormalisation(Y_output.T)
        return Y_output.T
        






       
       
     
  
   