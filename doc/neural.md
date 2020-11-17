# `NeuralNetwork`: `neural` library documentation (v0.0.1).
## Contents:
0. [Introduction & general usage](#0-introduction-&-general-usage).
1. [Notation, user-defined types and operators](#1-notation-user-defined-types-and-operators).
2. [Exhaustive review of classes](#2-exhaustive-review-of-all-declared-classes).
3. [Utility functions](#3-review-of-utility-functions-used-in-the-code).
4. [Who am I + contact info](#4-who-am-i).
5. [To-Do list](#5-to-do-list).

----------

## 0. Introduction & general usage.
`NeuralNetwork` is a one-author project ([me!](#4-who-am-i)) consisting of creating an Artificial Neural Network library for the **C++** language. Its first version (0.0.0) was completed for release on my summer 2020 holidays. As of this first version, I have implemented regression MLPs, using the squared error loss function and the backpropagation algorithm.

### 0.0. How to use `neural`.
To use the `neural` header/library you just have to use the following line in your **C++** main file (or in the file where you want to use the library):
```{c++}
#include "<path-to-header-file>/neural"
```
In this way, you can just compile the program as you were doing before, for example by using
```{sh}
g++ -Wall -std=c++17 -O2 -o main main.cc
```
So, you can go ahead and use the functions and classes from the library without much difficulty.

## 1. Notation, user-defined types and operators.
Throughout the documentation (and the code itself) `vd` is used as a `vector<double>` **C++** identifier. Likewise, `matd` refers to a *matrix double*, or `vector<vd>`, and `tend` refers to a *3-tensor double* (using tensor as the multidimensional array concept rather than the multilinear map), or a `vector<matd>`. Both `matd` and `tend` have little regard towards dimension consistency, as in the code itself it is often considered that both 2- and 3-dimensional arrays can have slices of different dimensions. To refer to an activation function, the defined type `activation` is used as a `double [](double)`. Also (but less frequently) used is the alias `ui` for `unsigned int`. There is intent on dropping this last one.

### 1.0. User-defined operators on the user-defined types.

#### 1.0.0. `vd`.
`vd` has three different user-defined operators: the first is `double operator*(const vd&,const vd&)`, which implements the scalar product between two same-dimensional vectors. The second one is the vector difference `vd operator-(const vd&,const vd&)`, which also does only work on a pair of same-dimensional vectors. The third one is the vector sum, `vd operator+(const vd&,const vd&)`.

#### 1.0.1. `matd`.
`matd` has three different user-defined operators: the first one is `matd operator*(const matd&,const matd&)`, which implements the well-known matrix multiplication. The second operator is the matrix difference `matd operator-(const matd&,const matd&)`, and the third one is the matrix sum `matd operator+(const matd&,const matd&)`. There has been no need to implement the determinant or the inverse of a matrix (yet). There is an additional user-defined operator that involves the `matd` class, and that is the matrix-vector product `vd operator*(const matd&,const vd&)`. As this involves both `vd` and `matd`, I do not count it as part of either.

#### 1.0.2. `tend`.
`tend` has only two user-defined operators, the 3-dimensional array difference and sum, `tend operator-(const tend&,const tend&)` and `tend operator+(const tend&,const tend&)`. There has been no need of implementing tensor contraction or the like.

## 2. Exhaustive review of all declared classes.

### 2.0. `Neuron`.
The `Neuron` class is the base of all the other classes in this library. Its private attributes are a weight vector, `vd weights`, a local additive parameter called `double intercept`, an input dimension `ui inputDim` and a pointer to an activation function, `activation *act`. It also has many member functions, namely:

#### 2.0.0. Constructors.
I implemented constructors as I saw the need for them. In total, there are 8 `Neuron` constructors:

- `Neuron(ui inDim, vd weights, double incpt, activation g) : weights(weights), intercept(incpt), inputDim(inDim), act(g) {}`.
- `Neuron(ui inDim, activation g) : weights(vd(inDim, 0)), intercept(0.0), inputDim(inDim), act(g) { randomInitIncpt(); randomInitWeights(); }`.
- `Neuron(ui inDim, vd weights, double incpt) : weights(weights), intercept(incpt), inputDim(inDim), act(tanh) {}`.
- `Neuron(vd weights, activation g) : weights(weights), intercept(0.0), inputDim(weights.size()), act(g) { randomInitIncpt(); }`.
- `Neuron(vd weights, double incpt) : weights(weights), intercept(incpt), inputDim(weights.size()), act(tanh) {}`.
- `Neuron(ui inDim, vd weights) : weights(weights), intercept(0.0), inputDim(inDim), act(tanh) { randomInitIncpt(); }`.
- `Neuron(ui inDim) : weights(vd(inDim, 0.0)), intercept(0.0), inputDim(inDim), act(tanh) { randomInitWeights(); randomInitIncpt(); }`
- `Neuron() : weights(vd(0)), intercept(0.0), inputDim(0), act(tanh) {}`.

Note that the attributes of a particular `Neuron` object can be changed through the setter functions in the following section.

#### 2.0.1. Setters.
The following member functions can be used to set a `Neuron` object attributes to a particular given value.
- `void setInputDim(int)`: sets the neuron input dimension to the `int` parameter.
- `void setActivation(activation)`: sets the neuron activation function to the `activation` parameter.
- `void setWeights(vd)`: sets the neuron weights vector to the `vd` parameter.
- `void setIncpt(double)`: sets the neuron intercept to the `double` parameter.

#### 2.0.2. Weights and intercept random initializers.
The `Neuron` class has two different functions to initialize the weights and the intercept randomly. These are `void randomInitWeights()` and `void randomInitIncpt()`. They use a global normal distribution with zero mean and unit variance (`normal01`, together with a random engine `gen`) to set the values.

#### 2.0.3. Getters.
The following functions are used to get the value of the neuron attributes. They do not stay up to date when the attributes are changed.
- `int getInputDim() const`.
- `vd getWeights() const`.
- `activation *getAct() const`.
- `double getIntercept() const`.

#### 2.0.4. `double execute(const vd &) const`.
This function executes a neuron's "operation", that is, calculates its output based on an input `vd`.

#### 2.0.5. `double getNeuronRaw(const vd&) const`.
This function returns the "raw" neuron output, this is, the scalar product of weights with the input `vd` parameter plus the neuron intercept, no activation function involved. This is primarily used as a utility functin to make the `NNet::train()` function more readable and understandable.
<!---
#### 2.0.6. `void train(const matd &)`.
--->

#### 2.0.6. Further plans.
There is a plan in place to implement the `Neuron::train()` function using maximum likelihood arguments. This is not an easy thing in C++, so it is left for a future version.

### 2.1. `Layer`.
The `Layer` class is a utility class. It mainly represents a neuron array `vector<Neuron>`, which has an alias, `neuronArray`. Its private attributes are a name (`string`) and the layer's units (`neuronArray`). It was basically made into a class in order to be able to add member functions to a `neuronArray`, which is not possible when just using a `vector<Neuron>`. It has a name field as I consider this to be a sufficiently structured object for the programmer to be able to distinguish easily between different `Layer` objects. The class has a number of member functions:

#### 2.1.0. Constructors.
The constructors implemented for the `Layer` class are the following ones:

- `Layer(string id, ui size, ui inDim, activation h)`.
- `Layer(string id, ui size, ui inDim)`.
- `Layer()`.

#### 2.1.1. Setters.
The setters for this class are:

- `void setInputDim(int)`.
- `void setLayerActivation(activation)`.

There is also a special function to set at the same time the size, the input dimension and the activation function of a `Layer` object. The function is:

- `void setupNeurons(int howMany, int inputDim, acivation g = tanh)`.

It is mainly intended to be used after the `NNet::addLayer()` empty overload. It is also used in its non-empty overload `NNet::addLayer(int,activation)` to set up the neurons in the newly added layer according to the parameters.

#### 2.1.2. Getters.
The getter functions for this class are:

- `int getSize() const`.
- `string getName() const`.
- `neuronArray getNeurons() const`.
- `matd getWeights() const`.
- `vector<activation*> getLayerActivations() const`.
- `vd getLayerIntercepts() const`.

#### 2.1.3. `Neuron *operator[](int i)`.
This function is a workaround to be able to edit directly the parameters of a `Neuron` by reference. Note this cannot be done through the `Layer::getNeurons()` method.

#### 2.1.3. `vd execute(const vd &input) const`.
This function returns a vector with the outputs of all the neurons in the layer, using the `double Neuron::execute(const vd&)`.

#### 2.1.4. `vd getNeuronRaws(const vd &input) const`.
This function returns a vector with the *raw* outputs of the layers' neuron, this is, without the activation function.
<!---
#### 2.1.5. `void train(const vd &) const`.
--->

### 2.2. `NNet`.
The `NNet` class is the most important class we use in this package. An object of this class consists of a name (`string`), an array of `Layer` objects and a pair of `int` attributes, the input and the output dimension of the network. This is useful within the network's inner workings, and is useful primarily when one creates an empty `NNet` object and wants to add a layer to it.

#### 2.1.0. Constructors.
As of now, there are four different constructors to construct a `NNet` object:

- `NNet(string name, int inDim, int outDim, vector<Layer> layers)`. This first constructor uses all of the attributes of the class at play to create a network from some pre-defined layers.
  /* This constructor builds numLayers empty layers, which then have to be
  better specified with: size of the layers, activation function(s), etc
- `NNet(string name, int inDim, int outDim, int numLayers)`. This constructor creates `numLayers` empty layers. Then, these have to be better specified using a number of member functions of the `Layer` or the `NNet` classes.
- `NNet(string name, int inDim, int outDim)`. This constructor takes in the input and output dimensions and creates an almost-empty `NNet` object. The layers must be added after creation using the appropiate `addLayer()` overload.
- `NNet()`. This creates an empty `NNet` object, which has to be better specified using the setter member functions.

#### 2.1.1. Setters.
The setters for this class are pretty much self-explanatory.

- `void setDim(ui in, ui out)`.
- `void setWeights(const tend &newWeights)`.
- `void setName(string newName)`.

#### 2.1.2. `addLayer()` overloads.
The member function `addLayer()` appends a `Layer` object at the end of the `layers` vector in the referenced `NNet` object. There are two ways to do this. The first overload is `void addLayer(int newSize, activation g)`, which adds a new layer that is of size `newSize` and has a layer-wide activation function, `g`. The second overload is `void addLayer()`, which adds a layer of the same size as the (previously) last one and with a the default layer-wide activation function `tanh`. Both overloads make sure that the input dimension of the newly added layer is the same as the size of the (previously) last layer.

#### 2.1.3. `insertLayer(int where, Layer what)`.
This function inserts a `Layer` object in the position specified. It is still in development so I do not recommend to use it, unless you take a look at the source code and know what you are doing and what you need to set after the insertion.

#### 2.1.2. Getters.
The getters are pretty self-explanatory too. They are the following ones.

- `string getName() const`.
- `vector<Layer> getLayers() const`.
- `int getDepth() const`: returns the network's number of layers.
- `tend getWeights() const`: returns a `tend` object with the weights indexed by layer-neuron-weight.
- `vector<int> getDim() const`.

#### 2.1.3. `Layer *operator[](int i)`
As previously done within the `Layer` class, I needed an operator to be able to modify the layers of the network directly, without the use of special setter functions that used many parameters. Again, note this cannot be done through the `NNet::getLayers()` method.

#### 2.1.4. `matd propagateForward(const vd &input)`.
This function is a similar drill to the `execute` methods in the `Neuron` and `Layer` classes. It generates the outputs of all the neurons in the network and returns them in a two-dimensional array, indexed by layer-neuron.

#### 2.1.5. `void train(const matd &data, int targetColumn, int maxEpochs = 100, double alphaStep = 0.05)`.
This function implements the Backpropagation Algorithm for regression MLP training. It has a target variable, a maximum number of iterations and a learning rate, known as the alpha parameter, which has a default value of 0.05. It uses a slightly tweaked version of the one present in my course notes of Machine Learning 1 (*Aprenentatge Automàtic 1*), which can be found [here](https://www.github.com/Atellas23/apunts/blob/master/Q4/AA1/ML1.pdf).

## 3. Review of utility functions used in the code.

### 3.0. `void print()` overloads.
There are three `void print()` overloads in the code. They are:

- `void print(const vd&)`.
- `void print(const matd&)`.
- `void print(const tend&)`.

The `vd` overload prints a vector in a line. The `matd` overload prints the matrix by lines, regarding the lines as vectors. Finally, the `tend` overload prints the 3-tensor by slices, printing a matrix each time.

### 3.1. `matd transpose(const matd&)`.
As the name suggests, this just returns the transpose of the given `matd`.

### 3.2. `double logistic(double)`.
This function implements the logistic map,
<center>
<figure>
<img src = "https://wikimedia.org/api/rest_v1/media/math/render/svg/faaa0c014ae28ac67db5c49b3f3e8b08415a3f2b">
<figcaption> Logistic function (from Wikipedia's article on sigmoidal functions).
</figure>
</center>
It is a widely used activation function, and it can be directly invoked if you are using the library.

### 3.3. `double machineEpsilon()`.
This function calculates an approximation to the quantity known as the "machine epsilon", the smallest possible double precision floating point number eps such that 1+eps>1.

### 3.4. `double derivativeApproximation(activation,double)`.
This function calculates a derivative approximation of the `activation` parameter on the point represented by the `double` parameter. It calculates it using the second order Taylor derivative approximation.

### 3.5. `double tensorL2Norm(const tend&)`.
This function calculates the L2 norm of a tensor, adding up all of its components squared and returning the square root of this quantity.

### 3.6. `vd removeColumn(matd&,int)`.
This function takes a matrix reference `matd&` and an integer `int` as parameters, and returns a copy of the column pointed at by the `int` parameter. Also, this column is erased from the matrix reference.

### 3.7. `schur()` overloads.
The Schur product of two matrices (or two-dimensional arrays, respectively two vectors) is the matrix (vector) resulting of multiplying each number in the first matrix (vector) by the number in the same position in the second one. It is an associative and commutative operation, and it is distributive with respect to the sum. The two overloads of this function, hence, are the vector Schur product `vd schur(const vd&,const vd&)` and the matrix Schur product `matd schur(const matd&,const matd&)`.

## 4. Who am I.
My name's Àlex. I'm a student at **Universitat Politècnica de Catalunya (UPC-BarcelonaTech)**, currently studying the Bachelors of [Mathematics](https://www.fme.upc.edu/en/studies/degrees/bachelors-degree-in-mathematics-1) and [Data Science and Engineering](https://www.dse.upc.edu) through the double-degree plan at [CFIS](https://www.cfis.upc.edu). I'm from a town called Arenys de Munt, a 40-minute car-drive from Barcelona. If you want to contact me, feel free to do so emailing me at [alex.batlle01@gmail.com](mailto:alex.batlle01@gmail.com?subject=GitHub%20NeuralNetwork%20contact).

## 5. To-Do list.
To-do list in order of priority:

- Write more documentation.
- Control the network parameter growth, and if needed, regularize the parameter learning pro
- Add **Radial Basis Function** models.
- Add genetic training methods for models. In this regard,
  - Particle Swarm Optimization
  - Ant Colony Optimization
  - Usual Genetic Algorithm
- Add meta-heuristic training methods to train models in a more understandable and adaptative way.
- Add **Extreme Learning Machines** as a usable model class.
- Finish the implementation of the `NNet::insertLayer()` method. Namely, make sure that the layer after the inserted one has the correct input dimension, and that the layer behind the inserted one has the correct output dimension. Notice the cascade effect that this may have on the following layers in the network, and adapt it to the changes.
- Implement the delta rule in `void Layer::train()`.
- Implement the `void Neuron::train()` method.
- Study where can parallelism be exploited and add it efficiently.
- Implement connection dropout methods.
- Implement classification MLPs.
- Implement the `void NNet::train()` method for multi-dimensional targets.
