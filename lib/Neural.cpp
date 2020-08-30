#include "Neural.hh"
using namespace std;

int NeuronPool = 0;
int LayerPool = 0;
int NetPool = 0;

normal_distribution<double> normal01(0, 1);
default_random_engine gen;

const double eps = machineEpsilon();

enum errorType
{
  unconsistentDimensions,
  unknownIdentifier,
  outOfRange
};

void error(errorType err)
{
  cout << "Error: ";
  if (err == unconsistentDimensions)
    cout << "The input and the expected dimensions are different. Aborting." << endl;
  else if (err == unknownIdentifier)
    cout << "The identifier is not known. Aborting operation." << endl;
  else if (err == outOfRange)
    cout << "The position specified is out of the vector's range. Aborting operation." << endl;
}

double operator*(const vd &a, const vd &b)
{
  double s = 0;
  int n = a.size(), m = b.size();
  if (n != m)
  {
    error(unconsistentDimensions);
    return NAN;
  }
  for (int i = 0; i < n; ++i)
    s += a[i] * b[i];
  return s;
}

vd operator-(const vd &a, const vd &b)
{
  int n = a.size(), m = b.size();
  if (n != m)
  {
    error(unconsistentDimensions);
    return vd(0);
  }
  vd res(n);
  for (int i = 0; i < n; ++i)
    res[i] = a[i] - b[i];

  return res;
}

vd operator+(const vd &a, const vd &b) {
  vd res = a;
  if (a.size() != b.size()) {
    error(unconsistentDimensions);
    return vd(0);
  }
  for (int i = 0; i < (int)a.size(); ++i) res[i] += b[i];
  return res; 
}

vd operator*(const matd& A, const vd& x) {
  if (A[0].size() != x.size()) {
    error(unconsistentDimensions);
    return vd(0);
  }
  vd res(A.size());
  for (int i = 0; i < (int)A.size(); ++i) res[i] = A[i]*x;
  return res;
}

vd schur(const vd &a, const vd &b) {
  vd res = a;
  if (a.size() != b.size()) {
    error(unconsistentDimensions);
    return vd(0);
  }
  for (int i = 0; i < (int)a.size(); ++i) res[i] *= b[i];
  return res;
}

matd operator*(const matd &a, const matd &b)
{
  int n1 = a.size(), m1 = a[0].size();
  int n2 = b.size(), m2 = b[0].size();
  if (m1 != n2)
  {
    error(unconsistentDimensions);
    return matd(0);
  }
  matd res(n1, vd(m2, 0));
  for (int i = 0; i < n1; ++i)
  {
    for (int j = 0; j < m2; ++j)
      res[i][j] = a[i] * transpose(b)[j];
  }
  return res;
}

matd operator-(const matd &a, const matd &b)
{
  int n1 = a.size(), m1 = a[0].size();
  int n2 = b.size(), m2 = b[0].size();
  if (not(n1 == n2 and m1 == m2))
  {
    error(unconsistentDimensions);
    return matd(0);
  }
  matd res(n1, vd(m1, 0));
  for (int i = 0; i < n1; ++i)
    res[i] = a[i] - b[i];
  return res;
}

matd operator+(const matd &a, const matd &b)
{
  if (a.size() != b.size())
  {
    error(unconsistentDimensions);
    return matd(0);
  }
  matd res(a.size());
  for (int i = 0; i < (int)a.size(); ++i)
    res[i] = a[i] + b[i];
  return res;
}

matd schur(const matd &a, const matd &b) {
  matd res = a;
  if (a.size() != b.size() or a[0].size() != b[0].size()) {
    error(unconsistentDimensions);
    return matd(0);
  }
  for (int i = 0; i < (int)a.size(); ++i) {
    for (int j = 0; j < (int)a[0].size(); ++j) res[i][j] *= b[i][j];
  }
  return res;
}

matd transpose(const matd &m)
{
  int n1 = m.size(), n2 = m[0].size();
  matd res(n2, vd(n1, 0.0));
  for (int i = 0; i < n1; ++i)
  {
    for (int j = 0; j < n2; ++j)
      res[j][i] = m[i][j];
  }
  return res;
}

tend operator-(const tend &a, const tend &b)
{
  int n1 = a.size(), m1 = a[0].size(), l1 = a[0][0].size();
  int n2 = b.size(), m2 = b[0].size(), l2 = b[0][0].size();
  if (not(n1 == n2 and m1 == m2 and l1 == l2))
  {
    error(unconsistentDimensions);
    return tend(0);
  }
  tend res(n1, matd(m1, vd(l1, 0.0)));
  for (int i = 0; i < n1; ++i)
    res[i] = a[i] - b[i];
  return res;
}

tend operator+(const tend &a, const tend &b)
{
  if (a.size() != b.size())
  {
    error(unconsistentDimensions);
    return tend(0);
  }
  tend res(a.size());
  for (int i = 0; i < (int)a.size(); ++i)
    res[i] = a[i] + b[i];
  return res;
}

void print(const vd &v)
{
  int n = v.size();
  for (int i = 0; i < n; ++i)
    cout << (i ? " " : "") << v[i];
  cout << endl;
}

void print(const matd &m)
{
  for (vd row : m)
    print(row);
  cout << endl;
}

void print(const tend &t)
{
  int i = 0;
  for (matd m : t)
  {
    cout << "Matrix at depth " << ++i << ":\n";
    print(m);
  }
  cout << endl;
}

double logistic(double x) { return 1.0 / (1.0 + exp(-x)); }

double machineEpsilon()
{
  double candidate = 1;
  while (1 + candidate > 1)
    candidate /= 2;
  return 2 * candidate;
}

double derivativeApproximation(activation g, double x)
{
  double h = sqrt(eps);
  return (g(x + h) - g(x - h)) / (2 * h);
}

vd derivativeApproximation(const vector<activation*> &g, const vd &x) {
  if (g.size() != x.size()) {
    error(unconsistentDimensions);
    return vd(0);
  }
  vd res(x.size(), 0.0);
  for (int i = 0; i < (int)x.size(); ++i) {
    res[i] = derivativeApproximation(g[i], x[i]);
  }
  return res;
}

double tensorL2Norm(const tend &t)
{
  double norm = 0.0;
  for (matd m : t)
  {
    for (vd &v : m)
    {
      for (double &a : v)
        norm += a * a;
    }
  }
  return sqrt(norm);
}

// double relativeTensorDifferenceNorm(const tend &reference, const tend &minuend) { return tensorL2Norm(reference - minuend) / tensorL2Norm(reference); }

vd removeColumn(matd &m, int c)
{
  vd res(m.size(), 0.0);
  for (vd &row : m)
  {
    res.push_back(row[c]);
    row.erase(row.begin() + c);
  }
  return res;
}

/****** NEURON CLASS ******/

Neuron::Neuron(ui inDim, vd weights, double incpt, activation g) : weights(weights), intercept(incpt), inputDim(inDim), act(g) {}

Neuron::Neuron(ui inDim, activation g) : weights(vd(inDim, 0)), intercept(0.0), inputDim(inDim), act(g)
{
  randomInitIncpt();
  randomInitWeights();
}

Neuron::Neuron(ui inDim, vd weights, double incpt) : weights(weights), intercept(incpt), inputDim(inDim), act(tanh) {}

Neuron::Neuron(vd weights, activation g) : weights(weights), intercept(0.0), inputDim(weights.size()), act(g) { randomInitIncpt(); }

Neuron::Neuron(vd weights, double incpt) : weights(weights), intercept(incpt), inputDim(weights.size()), act(tanh) {}

Neuron::Neuron(ui inDim, vd weights) : weights(weights), intercept(0.0), inputDim(inDim), act(tanh) { randomInitIncpt(); }

Neuron::Neuron(ui inDim) : weights(vd(inDim, 0.0)), intercept(0.0), inputDim(inDim), act(tanh)
{
  randomInitWeights();
  randomInitIncpt();
}

Neuron::Neuron() : weights(vd(0)), intercept(0.0), inputDim(0), act(tanh) {}

void Neuron::setInputDim(ui inDim)
{
  inputDim = inDim;
  // WARNING: this deletes the previously existant weights
  // weights = vd(inDim, 0.0);
}

void Neuron::setWeights(vd tempWeights) { weights = tempWeights; }

void Neuron::setIncpt(double tempIntercept) { intercept = tempIntercept; }

void Neuron::randomInitIncpt() { intercept = normal01(gen); }

void Neuron::randomInitWeights()
{
  weights = vd(inputDim, 0);
  for (auto &w : weights)
    w = normal01(gen);
}

void Neuron::setActivation(activation h) { act = h; }

int Neuron::getInputDim() const { return inputDim; }

vd Neuron::getWeights() const { return weights; }

activation *Neuron::getAct() const { return act; }

double Neuron::getIntercept() const { return intercept; }

double Neuron::execute(const vd &input) const
{
  ui n = input.size();
  if (n != inputDim)
  {
    error(unconsistentDimensions);
    return NAN;
  }
  return act(input * weights + intercept);
}

double Neuron::getNeuronRaw(const vd &input) const {
  return (weights * input) + intercept;
}

/****** LAYER CLASS ******/

Layer::Layer(string id, ui size, ui inputDim, activation h) : name(id), units(neuronArray(size))
{
  for (auto &neuron : units)
  {
    neuron.setActivation(h);
    neuron.setInputDim(inputDim);
    neuron.randomInitWeights();
    neuron.randomInitIncpt();
  }
}

Layer::Layer(string id, ui size, ui inputDim) : name(id), units(neuronArray(size))
{
  for (Neuron neuron : units)
  {
    neuron.setInputDim(inputDim);
    neuron.randomInitWeights();
    neuron.randomInitIncpt();
  }
}

Layer::Layer() : name("Layer" + to_string(LayerPool++)), units(neuronArray(0)) {}

void Layer::setupNeurons(int howMany, int inputDim, activation g)
{
  if (getSize() > 0)
  {
    cout << "Warning! This deletes the present neurons in the layer. Continue? (y/n)" << endl;
    char c;
    while (cin >> c and c != 'y' and c != 'n')
      cout << "Please use 'y' for yes and 'n' for no." << endl;
    if (c == 'n')
      return;
    else if (c == 'y')
    {
      cout << "Deleting the current existing neurons in the layer..." << endl;
      units = neuronArray(howMany); // called empty constructor howMany times
      for (Neuron neuron : units)
      {
        neuron.setInputDim(inputDim);
        neuron.randomInitWeights();
        neuron.randomInitIncpt();
        neuron.setActivation(g);
      }
      cout << "Setting up " << howMany << " neurons in this layer, using the first neuron's activation as the new layer activation function." << endl
           << "The new neurons' input dimension is " << inputDim << ". If you want to change this, please refer to the documentation." << endl;
    }
  }
  else
  {
    units = neuronArray(howMany); // called empty constructor howMany times
    for (Neuron neuron : units)
    {
      neuron.setInputDim(inputDim);
      neuron.randomInitWeights();
      neuron.randomInitIncpt();
      neuron.setActivation(g);
    }
    cout << "Setting up " << howMany << " neurons in this layer, using the first neuron's activation as the new layer activation function." << endl
         << "The new neurons' input dimension is " << inputDim << ". If you want to change this, please refer to the documentation." << endl;
  }
}

void Layer::setInputDim(int newDim)
{
  for (Neuron neuron : units)
    neuron.setInputDim(newDim);
}

void Layer::setLayerActivation(activation g)
{
  for (Neuron neuron : units)
    neuron.setActivation(g);
}

int Layer::getSize() const { return units.size(); }

string Layer::getName() const { return name; }

neuronArray Layer::getNeurons() const { return units; }

matd Layer::getWeights() const
{
  matd temp;
  for (Neuron neuron : units)
    temp.push_back(neuron.getWeights());
  return temp;
}

vector<activation*> Layer::getLayerActivations() const {
  vector<activation*> res;
  for (Neuron neuron : units) res.push_back(*neuron.getAct());
  return res;
}

vd Layer::getLayerIntercepts() const {
  vd res;
  for (Neuron neuron : units) res.push_back(neuron.getIntercept());
  return res;
}

vd Layer::execute(const vd &input) const
{
  vd outputs;
  for (Neuron unit : units)
    outputs.push_back(unit.execute(input));
  return outputs;
}

vd Layer::getNeuronRaws(const vd &input) const {
  vd res;
  for (Neuron unit : units)
    res.push_back(unit.getNeuronRaw(input));
  return res;
}

Neuron *Layer::operator[](int i) { return &units[i]; }

/****** NNET CLASS ******/

NNet::NNet(string name, int inDim, int outDim, vector<Layer> layers) : name(name), layers(layers), inputDim(inDim), outputDim(outDim) {}
// Constructor using a number of layers
/* This constructor builds numLayers "empty" layers, which then have to be
better specified with: size of the layers, activation function(s), etc
*/
NNet::NNet(string name, int inDim, int outDim, int numLayers) : name(name), layers(vector<Layer>(numLayers)), inputDim(inDim), outputDim(outDim) {}

NNet::NNet(string name, int inDim, int outDim) : name(name), layers(vector<Layer>(0)), inputDim(inDim), outputDim(outDim) {}
// Empty constructor
NNet::NNet() : name("Net" + to_string(NetPool++)), layers(vector<Layer>(0)), inputDim(0), outputDim(0) {}

void NNet::addLayer(int newSize, activation g) { layers.push_back(Layer("", newSize, (layers.size() ? layers.back().getSize() : inputDim), g)); }

void NNet::addLayer() { layers.push_back(Layer("", (layers.size() ? layers.back().getSize() : inputDim),
                                               (layers.size() ? layers.back().getSize() : inputDim), tanh)); }

void NNet::insertLayer(int where, Layer what)
{
  if (where >= layers.size() or where < 0)
  {
    error(outOfRange);
    return;
  }
  layers.insert(layers.begin()+where, what);
  /* if (where >= layers.size() or where < 0)
  {
    error(outOfRange);
    return;
  }

  for (int i = (int)layers.size() - 1; i >= where; --i)
  {
    if (i == (int)layers.size() - 1)
      addLayer();
    else
      layers[i + 1] = layers[i];
  }
  layers[where] = what; */
}

string NNet::getName() const { return name; }

vector<Layer> NNet::getLayers() const { return layers; }

int NNet::getDepth() const { return layers.size(); }

tend NNet::getWeights() const
{
  tend temp;
  for (Layer layer : layers)
    temp.push_back(layer.getWeights());
  return temp;
}

void NNet::setWeights(const tend &newWeights) {
  for (int l = 0; l < (int)newWeights.size(); ++l) {
    for (int n = 0; n < (int)newWeights[0].size(); ++n) 
      layers[l][n]->setWeights(newWeights[l][n]);
  }
}

void NNet::setDim(ui in, ui out) {
  inputDim = in;
  outputDim = out;
}

void NNet::setName(string newName) { name = newName; }

Layer *NNet::operator[](int i) { return &layers[i]; }

vector<int> NNet::getDim() const { return {inputDim, outputDim}; }

matd NNet::propagateFwd(const vd &input) const
{
  matd res;
  for (int i = 0; i < getDepth(); ++i)
    res.push_back(layers[i].execute(i ? res[i - 1] : input));
  return res;
}

void NNet::train(const matd &data, int targetColumn, int maxEpochs, double alphaStep)
{
  tend entryWeights;
  tend weightUpdates;
  for (int l = 0; l < getDepth(); ++l)
    weightUpdates[l] = matd(layers[l].getSize(), vd(layers[l][0]->getInputDim(), 0.0));
  matd newData = data;
  vd target = removeColumn(newData, targetColumn); // copies and erases the specified column
  int epoch = 0, nextCheckpoint = maxEpochs/10;
  do
  {
    entryWeights = getWeights();
    for (vd row : data)
    {
      // forward step
      matd neuronOutputs = propagateFwd(row);
      // backward step
      int h = getDepth() - 1; // this is the number of hidden layers
      matd deltas(h + 1);     // zero-indexed matrix of deltas

      // We initialize the different delta collections for each layer according to the size of the corresponding layer
      for (int i = 0; i < h + 1; ++i)
        deltas[i] = vd(layers[i].getSize(), 0.0);

      deltas[h] = schur(derivativeApproximation(layers[h].getLayerActivations(),
                                                entryWeights[h] * neuronOutputs[h - 1]
                                                + layers[h].getLayerIntercepts()),
                        neuronOutputs[h] - target);

      /*deltas[h] = derivativeApproximation(layers[h].getLayerActivations(),
                                          schur(entryWeights[h] * neuronOutputs[h - 1]
                                          + layers[h].getLayerIntercepts(),
                                          neuronOutputs[h] - target));*/

      for (int i = h - 1; i >= 0; --i) {
        deltas[i] = schur(transpose(layers[i+1].getWeights())*deltas[i+1],
                          derivativeApproximation(layers[i].getLayerActivations(),
                          layers[i].getNeuronRaws(neuronOutputs[i-1])));
      }

      tend weightUpdates = entryWeights;
      for (int l = 0; l < (int)weightUpdates.size(); ++l) {
        for (int j = 0; j < (int)weightUpdates[l].size(); ++j) {
          for (int i = 0; i < (int)weightUpdates[l][j].size(); ++i)
            weightUpdates[l][j][i] += alphaStep * deltas[l][j] * neuronOutputs[l-1][i];
        }
      }
    }
    setWeights(entryWeights + weightUpdates);
    if (epoch == nextCheckpoint) {
      cout << (epoch/maxEpochs) * 100 << "% of the training complete." << endl;
      nextCheckpoint += maxEpochs/10;
    }
  } while (epoch < maxEpochs and tensorL2Norm(getWeights() - entryWeights) > sqrt(eps));
  cout << "Training completed!" << endl << string(20,'*') << endl;
  if (epoch < maxEpochs) cout << "Training was complete without getting to the maximum epochs." << endl;
  // } while (epoch < maxEpochs and relativeTensorDifferenceNorm(getWeights(), entryWeights) > 0.001);
  return;
}

/* string defaultNetName()
{
  if (netMemPool > 0)
    return "network" + to_string(netMemPool--);
  else
  {
    char opt;
  fullNetMemMsg:
    cout << "NETWORK MEMORY FULL!\nDo you want to erase a network? (y/n) ";
    cin >> opt;
    if (opt == "y")
    {
      cout << "Please select a network to erase:\n";
      networkList();
    }
    goto fullNetMemMsg;
  }
} */