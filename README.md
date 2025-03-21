# Higgs_boson_scattering_amplitudes_DNN

[![Project Website](https://img.shields.io/badge/website-Project%20Site-blue)](https://www.aretetheory.co.uk/) 

## Performing Loop Integration Using Neural Networks (Dated: April 17, 2024) 
- **Abstract**: Neural network (NN) technology was used to integrate the Feynman parameterised integral for the 1 loop process of Higgs boson pair production, from a top loop, over a phase space region. Randomly sampled phase-space and Feynman parameters were used to obtain exact integrand values, that were then fitted to the derivative of the neural network. The neural network then evaluated these integrals over the trained region ≃ 10 times faster than the Monte Carlo integrator pySecDec, which integrates over specific phase-space configurations. 
Different activation functions were applied to the neural network to further the theoretical understanding and accuracy shown in the current literature. The performance of the architectures differed, because the shapes of their activation function’s derivatives affected the behaviour of their backpropagated gradients during training. 
The GELU based architecture was the most accurate, with a mean of 3.9 ± 0.2 correct digits over the trained phase-space region, beating the tanh-based network 3.4 ± 0.2 digits (the most accurate in the original literature). Larger batch sizes improved the accuracy of architectures, as the GELUbased network obtained an accuracy of 3.4±0.3 digits, when trained on a batch size
25 times smaller. Deep GELU networks were slightly less accurate 3.8 ± 0.2 correct digits. GELU based networks had better generalisation for the boundaries of the sample space, than the softsign, sigmoid, and tanh. The shape of the GELU’s first derivative made it less susceptible to dead node formation, than the other activation functions tested.
- **Keywords:** Neural Network, Activation Function, GELU, Parametric integration, Backprop

## Files
**dNN/dx code**
- I1_cpu_script_16_04_24.py
- I1_gpu_script_20_04_24.py

**experimentation notebook**
- reloaderI2_19_03_24.ipynb



