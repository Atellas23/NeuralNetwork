#ifndef IO
#define IO
#include <iostream>
#endif
#include <fstream>
#include <string>
#ifndef VEC
#define VEC
#include <vector>
#endif
#include <map>
#include "lib/neural"
#include "lib/dataHandler"

using namespace std;

#define inProgress cout << "I do not work correctly yet" << endl
#define unrComm cout << "error: unrecognized command" << endl
#define xivato cout << "SÓC UN XIVATO!" << endl
#define MAX_NETS 10
vs consoleLog;
vs recordLog;
bool recordTrigger = false;

map<string, NNet *> modelList;
map<string, dataFrame *> datasets;
map<string, activation *> activations;

string redify(string s)
{
  return "\x1B[31m" + s + "\033[0m";
}
void main_stream(const vs &c);

void save_log(string filename, const vs &what = consoleLog)
{
  ofstream logfile(filename);
  for (auto s : what)
    logfile << s << endl;
  logfile.close();
}

void log(string &s, vs &where = consoleLog) { where.push_back(s); }

void loadModel(const string &filename)
{
  ifstream file(filename);
  if (not file.good())
  {
    cout << "error: please use an existing file name" << endl;
    return;
  }
  string temp;
  getline(file, temp); // reads "modelname:" and then the model name
  string modelname = tokenize(temp)[1];
  modelList[modelname] = new NNet(modelname);
  getline(file, temp); // reads "layers:" and then the number of layers
  int depth = stoi(tokenize(temp)[1]);
  int m;
  vector<activation *> a;
  tend modelWeights;
  string act;
  matd layerWeights, modelIntercepts;
  for (int i = 0; i < depth; ++i)
  {
    getline(file, temp); // reads "layer" and then the layer's number
    // cout << temp << endl;
    getline(file, temp); // reads "size" and then the layer's size
    m = stoi(tokenize(temp)[1]);
    // cout << temp << endl;
    getline(file, temp); // reads the name of the activation function
    act = temp;
    // cout << act << endl;
    getline(file, temp); // reads "parameters"
    // cout << temp << endl;
    vd neuronWeights(0);
    for (int j = 0; j < m; ++j)
    { // read parameter matrix
      getline(file, temp);
      for (auto a : tokenize(temp))
        neuronWeights.push_back(stod(a));
      layerWeights.push_back(neuronWeights);
    }
    modelWeights.push_back(layerWeights);
    if (i == 0)
      modelList[modelname]->setDim(tokenize(temp).size(), 1); // for the moment, just 1-dimensional regression is supported
    modelList[modelname]->addLayer(m, activations[act]);
    getline(file, temp); // reads "intercepts"
    vd layerIntercepts;
    getline(file, temp);
    for (auto a : tokenize(temp))
      layerIntercepts.push_back(stod(a));
    modelIntercepts.push_back(layerIntercepts);
  }
  modelList[modelname]->setWeights(modelWeights);
  modelList[modelname]->setIntercepts(modelIntercepts);
}

void saveModel(const string &modelname, const string &filename)
{
  ofstream file(filename + ".NNet");
  NNet *mod = modelList[modelname];
  file << "modelname: " << modelname << endl;
  int m = mod->getDepth();
  file << "layers: " << m << endl;
  file << "spec" << endl;
  for (int i = 0; i < m; ++i)
  {
    Layer *l = mod->operator[](i);
    file << "layer " << i << endl;
    file << "size " << l->getSize() << endl;
    // file << "input " << l->getNeurons().front().getInputDim();
    activation *t = l->getLayerActivations().front();
    if (t == activations["logistic"])
      file << "logistic" << endl;
    else if (t == activations["relu"])
      file << "relu" << endl;
    else if (t == activations["tanh"])
      file << "tanh" << endl;
    else
      file << "unknown" << endl;
    file << "parameters" << endl;
    matd temp = l->getWeights(); // the input dimension can be deduced from here when loading the object
    for (auto vec : temp)
    {
      for (int j = 0; j < (int)vec.size(); ++j)
        file << (j ? " " : "") << vec[j];
      file << endl;
    }
    file << "intercepts" << endl;
    vd incpts = l->getLayerIntercepts();
    for (int j = 0; j < (int)incpts.size(); ++j)
      file << (j ? " " : "") << incpts[j];
    file << endl;
  }
}

void eraseModel(const string &modelname)
{
  delete modelList[modelname];
  modelList.erase(modelname);
}

void eraseDataset(const string &datasetName)
{
  delete datasets[datasetName];
  datasets.erase(datasetName);
}

void loadInstructions(const string &filename)
{
  ifstream file(filename);
  if (not file.good())
  {
    cerr << "error: the file \"" << filename << "\" could not be loaded." << endl;
    return;
  }
  string temp;
  while (getline(file, temp) and temp != "end")
  {
    vs comm = tokenize(temp);
    main_stream(comm);
  }
}

void loadData(const string &name, const string &filename)
{
  datasets[name] = new dataFrame();
  datasets[name]->read(filename);
  datasets[name]->setName(name);
}

void createModel(const string &modelname, int inDim, int outDim, int numLayers)
{
  if (modelList.find(modelname) != modelList.end())
  {
    cout << "Please use a name that is not already a model." << endl;
    return;
  }
  cout << "Creating model " + redify(modelname) + "." << endl;
  modelList[modelname] = new NNet(modelname, inDim, outDim, numLayers);
}

void showModelDetail(const string &modelname)
{
  if (modelList.find(modelname) == modelList.end())
  {
    cout << "error: please request details from an existing model" << endl;
    return;
  }
  NNet *model = modelList[modelname];
  cout << endl
       << "MODEL NAME: " + redify(model->getName()) << endl
       << "  Input dimension: " << model->getDim()[0] << endl
       << "  Output dimension: " << model->getDim()[1] << endl
       << "  Model depth: " << model->getDepth() << endl
       << "  Model parameters:" << endl;
  print(model->getWeights());
}

int find(const vs &v, const string &s)
{
  for (int i = 0; i < (int)v.size(); ++i)
  {
    if (v[i] == s)
      return i;
  }
  return -1;
}

void trainModel(const string &modelname, const string &datasetname, const string &targetcolumn, int numepochs, double learningrate)
{
  if (modelList.find(modelname) == modelList.end())
  {
    cout << "error: please use an existing model to train" << endl;
    return;
  }
  NNet *mod = modelList[modelname];
  if (datasets.find(datasetname) == datasets.end())
  {
    cout << "error: please use an existing dataset to train the model" << endl;
    return;
  }
  dataFrame *data = datasets[datasetname];
  auto names = data->colnames();
  int idx = find(data->colnames(), targetcolumn);
  if (idx == -1)
  {
    cout << "error: please use an existing column name in the dataset" << endl;
    return;
  }
  if (learningrate > 1 or learningrate < 0)
  {
    cout << "error: please use a positive learning rate strictly between 0 and 1" << endl;
    return;
  }
  mod->train(data->getData(), idx, numepochs, learningrate);
}

/*
void geneticTrain(const string &modelname, const string &datasetname, const string &targetcolumn)
{
  if (modelList.find(modelname) == modelList.end())
  {
    cout << "error: please use an existing model to train" << endl;
    return;
  }
  NNet *mod = modelList[modelname];
  if (datasets.find(datasetname) == datasets.end())
  {
    cout << "error: please use an existing dataset to train the model" << endl;
    return;
  }
  dataFrame *data = datasets[datasetname];
  int idx = find(data->colnames(), targetcolumn);
  if (idx == -1)
  {
    cout << "error: please use an existing column name in the dataset" << endl;
    return;
  }
  geneticTraining(*mod, data->getData(), idx);
}
*/

void printModelNames()
{
  int modelNum = 0;
  for (auto p : modelList)
    cout << ++modelNum << ": " + redify(p.first) << endl;
}

void printDatasetNames()
{
  int datasetNum = 0;
  for (auto p : datasets)
    cout << ++datasetNum << ": " + p.first << endl;
}

void modelEditSubconsole(const string &modelname)
{
  if (modelList.find(modelname) == modelList.end())
  {
    cout << "error: please use an existing model" << endl;
    return;
  }
  cout << "Editing model " + redify(modelname) << endl;
  cout << "[ME]> ";
  string temp;
  while (getline(cin, temp) and temp != "end")
  {
    vs comm = tokenize(temp);
    if (comm.size() < 1)
      ;
    else if (comm[0] == "addlayer")
    {
      if (comm.size() == 3)
      {
        if (not isNumber(comm[1]))
          cout << "error: correct use is \"addlayer {num layers} {activation function}\"" << endl;
        else if (activations.find(comm[2]) == activations.end())
          cout << "error: you must use an existing activation function (see documentation or type \"help activations\")" << endl;
        else
          modelList[modelname]->addLayer(stoi(comm[1]), activations[comm[2]]);
      }
      else if (comm.size() == 1)
        modelList[modelname]->addLayer();
      else
        spitLine(comm);
    }
    else if (comm[0] == "editlayer")
    {
      if (comm.size() < 2)
        cout << "error: you must specify which layer to update" << endl;
      else if (stoi(comm[1]) >= modelList[modelname]->getDepth() or stoi(comm[1]) < 0)
        cout << "error: please refer to an existing layer in this model" << endl;
      else
      {
        cout << "Editing layer " + comm[1] << endl;
        cout << "[WARNING] The layer editing functionality has still to be reviewed as it has some delicate operations. Proceed with caution." << endl;
        Layer *l = modelList[modelname]->operator[](stoi(comm[1]));
        cout << "[L" + comm[1] + "E]> ";
        while (getline(cin, temp) and temp != "end")
        {
          vs subcomm = tokenize(temp);
          if (subcomm[0] == "set")
          {
            if (subcomm.size() < 2)
              cout << "error: invalid setting operation" << endl;
            else if (subcomm[1] == "name")
            {
              if (subcomm.size() < 3)
                cout << "error: please enter a valid name for this layer" << endl;
              else
                l->setName(subcomm[2]);
            }
            else if (subcomm[1] == "activation")
            {
              if (subcomm.size() < 3)
                cout << "error: please enter a valid activation function for this layer" << endl;
              else
                l->setLayerActivation(activations[subcomm[2]]);
            }
            else if (subcomm[1] == "neurons")
            {
              if (subcomm.size() < 5)
                cout << "error: you need to specify the number of neurons, the input dimension of the layer, and the layer-wide activation function" << endl;
              else
                l->setupNeurons(stoi(subcomm[2]), stoi(subcomm[3]), activations[subcomm[4]]);
            }
            else
              unrComm;
          }
          cout << "[L" + comm[1] + "E]> ";
        }
      }
    }
    else
      unrComm;
    cout << "[ME]> ";
  }
}

void main_stream(const vs &c)
{
  if (c.size() < 1)
    return;
  if (c[0] == "load")
  {
    if (c.size() < 2)
    {
      cout << "error: invalid load operation" << endl;
      return;
    }
    else if (c[1] == "model")
    {
      if (c.size() < 3)
      {
        cout << "error: load model needs a file name from which to read the model" << endl;
        return;
      }
      loadModel(c[2]);
    }
    else if (c[1] == "instructions")
      loadInstructions(c[2]);
    else if (c[1] == "data")
    {
      if (c.size() < 4)
      {
        cout << "error: load data needs a name for the dataset and a filepath" << endl;
        return;
      }
      loadData(c[2], c[3]);
    }
    else
      cout << "error: invalid load operation" << endl;
  }
  else if (c[0] == "record")
  {
    if (recordTrigger)
      cout << "already recording!" << endl;
    else
    {
      recordTrigger = true;
      string d;
      cout << "[R]> ";
      while (getline(cin, d) and d != "end")
      {
        log(d, recordLog);
        vector<string> comm = tokenize(d);
        main_stream(comm);
        cout << "[R]> ";
      }
      recordTrigger = false;
      save_log((c.size() == 2 ? c[1] : "recording"), recordLog);
    }
  }
  else if (c[0] == "create")
  {
    if (c.size() < 2)
    {
      cout << "error: incomplete command. Please specify what you want to create" << endl;
      return;
    }
    else if (c[1] == "model")
    {
      if (c.size() < 6)
      {
        cout << "error: must give the new model a name, input and output dimensions, and number of layers" << endl;
        return;
      }
      if (modelList.size() >= MAX_NETS)
      {
        cout << "The model memory is full! Do you want to evict a model (y/n)? ";
        char c;
        while (cin >> c and c != 'y' and c != 'n')
          cout << "Please use 'y' for YES and 'n' for NO" << endl;
        if (c == 'n')
        {
          cout << "create operation canceled" << endl;
          return;
        }
        cout << "Choose a model from below:" << endl;
        printModelNames();
        cout << "Model to be erased (name): ";
        string name;
        cin >> name;
        cout << "Erasing model " + redify(name) + "." << endl;
        eraseModel(name);
      }
      createModel(c[2], stoi(c[3]), stoi(c[4]), stoi(c[5]));
      // return;
    }
    else
    {
      cout << "error: invalid object creation" << endl;
      return;
    }
  }
  else if (c[0] == "show")
  {
    if (c.size() < 2)
    {
      cout << "error: invalid show operation" << endl;
      return;
    }
    if (c[1] == "models")
      printModelNames();
    else if (c[1] == "datasets")
      printDatasetNames();
    else if (c[1] == "data")
    {
      if (c.size() < 3)
      {
        cout << "error: incomplete call to show data" << endl;
        return;
      }
      if (datasets.find(c[2]) == datasets.end())
      {
        cout << "error: please use an existing dataset name" << endl;
        return;
      }
      datasets[c[2]]->print();
    }
    else if (c[1] == "model")
      showModelDetail(c[2]);
    else
      cout << "error: invalid object to show" << endl;
  }
  else if (c[0] == "edit")
  {
    if (c.size() < 2)
    {
      cout << "error: no model object to edit" << endl;
      return;
    }
    modelEditSubconsole(c[1]);
  }
  else if (c[0] == "train")
  {
    if (c.size() < 6)
    {
      cout << "error: you must specify all of the following:" << endl
           << "- model to train" << endl
           << "- dataset to train from" << endl
           << "- target column of the dataset (name)" << endl
           << "- max number of epochs in training (int)" << endl
           << "- learning rate (double)" << endl;
      return;
    }
    trainModel(c[1], c[2], c[3], stoi(c[4]), stod(c[5]));
  }
  else if (c[0] == "save")
  {
    if (c.size() < 3)
    {
      cout << "error: current functionality of the \"save\" command is just to save a model to a file" << endl;
      return;
    }
    saveModel(c[1], c[2]);
  }
  else if (c[0] == "erase")
  {
    if (c.size() < 3)
    {
      cout << "error: invalid erase operation" << endl;
      return;
    }
    if (c[1] == "model")
    {
      if (modelList.find(c[2]) == modelList.end())
      {
        cout << "error: please use a valid model name" << endl;
        return;
      }
      eraseModel(c[2]);
    }
    else if (c[1] == "dataset")
    {
      if (datasets.find(c[2]) == datasets.end())
      {
        cout << "error: please use a valid dataset name" << endl;
        return;
      }
      eraseDataset(c[2]);
    }
    else
    {
      cout << "error: invalid erase operation" << endl;
      return;
    }
  }
  /*
  else if (c[0] == "genetictrain")
  {
    if (c.size() < 4)
    {
      cout << "error: you must specify all of the following:" << endl
           << "- model to train" << endl
           << "- dataset to train from" << endl
           << "- target column of the dataset (name)" << endl;
           // << "- number of generations in training (int)" << endl
           // << "- how many members are in each gen (int)" << endl;
      return;
    }
    geneticTrain(c[1], c[2], c[3]);
  }
  */
  else if (c[0] == "clear")
    cout << string(50, '\n');
  else
  {
    cout << "error: \"" + c[0] + "\" was not recognized as a command\n";
  }
  // unrComm;
}

void setups()
{
  activations["tanh"] = tanh;
  activations["logistic"] = logistic;
  activations["relu"] = relu;
}

int main()
{
  setups();
  cout << "NeuralNetwork 0.0.1 console (testing workflow). Work by Atellas23." << endl
       << endl;
  cout << "> ";
  string s;
  while (getline(cin, s) and s != "end")
  {
    if (s.length() < 1)
      ;
    if (s == "exit")
      break;
    log(s);
    if (s[0] == '#')
      cout << '#' << endl; // Commentary check
    else
    { // Separate in different words (considered different when separated by a space ' ')
      vector<string> command = tokenize(s);
      main_stream(command); // Run command through the main command line stream
    }
    cout << "> ";
  }
  save_log("console.log");
}
