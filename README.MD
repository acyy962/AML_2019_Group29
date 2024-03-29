# A breif comparision of gradient descent methods
As an optimization algorithm, gradient descent is widely used to minimize loss functions by iterative moves. In machine learning, it is used in many areas. e.g. find the best parameters of linear regression model or the optimal weights in neural network. In this project, we are going to perform three popular types of gradient descent that are important to machine learning algorithms.
 
## Loss function and gradients
The loss function implies how good the model performs in making predictions for a given set of parameters. It has its own gradients, which are just the derivatives of the function with respect to its variables in mathematics. In this report, we choose the three-dimensional Three-Hump Camel Function as our loss function. The 3D plot of this function is indicates as below.

![alt text](https://user-images.githubusercontent.com/52373417/61050393-d1d56080-a3de-11e9-81eb-3361a0792f7f.png)
 
## Plain vanilla gradient descent
Vanilla (or Batch) gradient descent takes small steps in the downhill direction. Starting from some random initial points of parameters and then find the gradient at the starting points. If the gradient at the start is not close to zero, then iterate the values of parameters until it is close to zero. Let’s run some experiments to see how the algorithm performs.
 
* Initialize X1 and X2 at 5
* Learning rate: 0.001
* Number of iterations: 10000
 
The process converged in 8220 steps with X* = (0, 0) and loss function equals zero. Although vanilla gradient descent is well-known for its computational effectiveness, the convergence speed is fairly slow for vanilla gradient descent. As is shown in Figure 1 (b) that the algorithm is quite effective in finding the global minima without too much jumps around the minimum point. 
 
![alt text](https://user-images.githubusercontent.com/52373417/60932957-377afd00-a2b8-11e9-94e4-140318469e1d.png)
 
The plot illustrates that the path of the loss function does follow a gradient descent. Moreover, the smaller the stepsize is, the more the steps it needs to converge. However, the algorithm cannot stand too high a stepsize due to extremely high value of loss function becomes. 

![alt text](https://user-images.githubusercontent.com/52373417/60933215-3b5b4f00-a2b9-11e9-98d1-cc84f68d8977.png)
 
## Momentum gradient descent
According to Andrew Ng, a momentum is a moving average of gradients. It finds global minima faster than vanilla descent. Because momentum calculates an exponentially weighted average of the gradients, and then use this gradient to update the parameters. This means that momentum can accumulate velocity in the direction where the gradient points in the same direction through iterations. Let’s run the same experiment as in vanilla descent to see if momentum is more effective.
 
![alt text](https://user-images.githubusercontent.com/52373417/60927968-91bd9300-a2a3-11e9-9913-38cbb64addc5.png)

Momentum is obviously more effective than vanilla descent with the loss of 3.097e-11 been achieved within only 688 steps. Moreover, under the same number of iterations, momentum can reach global minima with larger learning rate and lower loss function than vanilla gradient descent.

![alt text](https://user-images.githubusercontent.com/52373417/60933051-a2c4cf00-a2b8-11e9-95f0-eb183dead555.png)
 
## Nesterov accelerated gradient descent
Unlike momentum, Nesterov gradient descent first looking at a point where current momentum is facing and then calculating gradients from there. It is smarter than momentum in finding the descent direction.
 
![alt text](https://user-images.githubusercontent.com/52373417/60927999-aac64400-a2a3-11e9-9207-4ebc82d850e0.png)
 
From the plot, there are less oscillations than momentum. It reaches global minima within only 679 steps. The effectiveness of gradient descent has generally been improved. Its effectiveness also reflected by converging with smaller stepsize than momentum. Although, the smaller the stepsize is, the more steps it needs to reach convergence. 

![alt text](https://user-images.githubusercontent.com/52373417/60933029-86289700-a2b8-11e9-8027-beda6845082d.png)

## Reference:
* Muhammad Rizwan (2019), Gradient Descent with Momentum (https://engmrk.com/gradient-descent-with-momentum/)
* Deep Learning Specialization by Andrew Ng
* Sebastian Ruder (2016), An overview of gradient descent optimization algorithms (http://ruder.io/optimizing-gradient-descent/)

