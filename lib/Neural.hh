#ifndef NEURAL
#define NEURAL
#ifndef VEC
#define VEC
#include <vector>
#endif
#ifndef IO
#define IO
#include <iostream>
#endif
#include <random>
#include <math.h>
using namespace std;

typedef vector<double> vd;
typedef vector<vd> matd;
typedef vector<matd> tend;
typedef double activation(double);
typedef unsigned int ui;

double operator*(const vd &a, const vd &b);
vd operator-(const vd &a, const vd &b);
vd operator+(const vd &a, const vd &b);
vd operator*(const matd& A, const vd& x);
vd schur(const vd &a, const vd &b);
matd operator*(const matd &a, const matd &b);
matd operator-(const matd &a, const matd &b);
matd operator+(const matd &a, const matd &b);
matd schur(const matd &a, const matd &b);
matd transpose(const matd &m);
tend operator-(const tend &a, const tend &b);
tend operator+(const tend &a, const tend &b);
void print(const vd &v);
void print(const matd &v);
void print(const tend &v);
double logistic(double x);
double machineEpsilon();
double derivativeApproximation(activation g, double x);
vd derivativeApproximation(const vector<activation*> &g, const vd &x);
double tensorL2Norm(const tend &t);
// double relativeTensorDifferenceNorm(const tend &reference, const tend &minuend);
vd removeColumn(matd &m, int c);

// The NEURON class, fundamental to a neural network
class Neuron
{
public:
  // Constructors
  Neuron(ui inDim, vd weights, double incpt, activation g);
  Neuron(ui inDim, activation g);
  Neuron(vd weights, activation g);
  Neuron(ui inDim, vd weights, double incpt);
  Neuron(vd weights, double incpt);
  Neuron(ui inDim, vd weights);
  Neuron(ui inDim);
  Neuron();
  // Setter for the input dimension, if we previously used the empty constructor
  void setInputDim(ui inDim);
  // Setter for the activation function
  void setActivation(activation h);
  // Setters for the neuron weights and intercept
  void setWeights(vd tempWeights);
  void setIncpt(double tempIntercept);
  // Random initializers for the neuron weigths and the intercept
  void randomInitWeights();
  void randomInitIncpt();
  // Getter for the input dimension
  int getInputDim() const;
  // Getter for the neuron weights
  vd getWeights() const;
  // Getter for the activation function
  activation *getAct() const;
  // Getter for the intercept
  double getIntercept() const;

  // Calculates the output of the neuron
  double execute(const vd &input) const;

  // Utility function to calculate the neuron raw output (raw = without activation)
  double getNeuronRaw(const vd &input) const;

  /* Network.train(const vector<vd> &data);
  Trains the neuron according to the given (numeric) data. Note that this is
  done through maximum likelihood arguments, which is different from the method
  used in the layer and network trainings. The former ones use (respectively)
  the delta rule and the backpropagation algorithm.  
  */
  void train(const matd &data);

private:
  // Weights of the neuron
  vd weights;
  // Intercept of the neuron
  double intercept;
  // Input and output dimensions
  ui inputDim;
  // Activation function (as a type defined above)
  activation *act;
};

typedef vector<Neuron> neuronArray;

// The LAYER class; we will use it later in the construction of full networks
class Layer
{
public:
  // Constructor
  // We MUST specify the input dimension of the newly created neurons
  Layer(string id, ui size, ui inDim, activation h);
  Layer(string id, ui size, ui inDim);
  Layer();
  // Sets the layer's size, input dimension and activation function
  void setupNeurons(int howMany, int inputDim, activation g = tanh);
  // Setter for the units input dimension
  void setInputDim(int newDim);
  // Setter for the layer units activation function
  void setLayerActivation(activation g);
  // Getter for the layer size
  int getSize() const;
  // Getter for the layer name
  string getName() const;
  // Getter for a layer neurons' array copy
  neuronArray getNeurons() const;
  // Getter for the layer's weights
  vector<vd> getWeights() const;
  // Getter for the layer activation functions
  vector<activation*> getLayerActivations() const;
  // Getter for the layer neurons' intercepts
  vd getLayerIntercepts() const;
  // Operator to access the i-th neuron of the layer
  Neuron *operator[](int i);
  // Output calculations
  /* 
      As this is an MLP, the previous layer is fully connected to
      this one, and hence all the neurons have the same input vector
  */
  vd execute(const vd &input) const;
  // Utility function to calculate neuron raw outputs (raw = without activation)
  vd getNeuronRaws(const vd &input) const;

  /* Layer.train(const vector<vd> &data);
  Trains the layer according to the given (numeric) data. Note that this is
  done through the delta rule, which is different from the method used in the
  neuron and full network trainings. The former ones use (respectively) maximum
  likelihood and the backpropagation algorithm.
  */
  void train(const matd &data, double alphaStep = 0.05);

private:
  string name;
  neuronArray units;
};

// Between-layer fully-connected MLP
class NNet
{
public:
  // Constructor using pre-defined layers
  NNet(string name, int inDim, int outDim, vector<Layer> layers);
  // Constructor using a number of layers
  /* This constructor builds numLayers empty layers, which then have to be
  better specified with: size of the layers, activation function(s), etc
  */
  NNet(string name, int inDim, int outDim, int numLayers);
  // Constructor using the input and output dimensions ONLY
  /* This constructor just uses the input and output dimensions, creating 0 empty layers.
  These have to be then manually added, using the desired addLayer() method overload.
  */
  NNet(string name, int inDim, int outDim);
  // Empty constructor
  NNet();
  // Setter for the network's weights
  void setWeights(const tend &newWeights);
  // Setter for the network's input and output dimensions
  void setDim(ui in, ui out);
  // Setter for the network's name
  void setName(string newName);
  // Method to add a layer to the back of the neural network
  void addLayer(int newSize, activation g);
  // Overload to add an EMPTY layer to the back of the NN
  void addLayer();
  // Method to insert a layer in the position specified
  void insertLayer(int where, Layer what);
  // Getter for the network identifier
  string getName() const;
  // Getter for the network's layers
  vector<Layer> getLayers() const;
  // Getter for the network's depth
  int getDepth() const;
  // Getter for the network's weights
  tend getWeights() const;
  // Operator to access the i-th layer of the network
  Layer *operator[](int i);
  // Getter for the input and output dimensions of the neural network
  vector<int> getDim() const;

  // Function to calculate all of the neurons outputs
  matd propagateFwd(const vd &input) const;

  /* NNet.train(const vector<vd> &data, double alphaStep = 0.05);
  Trains the network according to the given (numeric) data. Note that this is
  done through the backpropagation algorithm, which is different from the
  method used in the neuron and layer trainings. The former ones use
  (respectively) maximum likelihood and the delta rule.
  */
  void train(const matd &data, int targetColumn, int maxEpochs = 100, double alphaStep = 0.05);

private:
  string name;
  vector<Layer> layers;
  int inputDim, outputDim;
};

#endif
