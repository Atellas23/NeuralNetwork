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

map<string,int> inModelList;
vector<NNet*> modelList;
map<string,int> inDatasets;
vector<dataFrame*> datasets;
map<string,activation*> activations;

void main_stream(const vs& c);

void save_log(string filename, const vs& where = consoleLog) {
  ofstream logfile(filename);
  for (auto s : where) logfile << s << endl;
  logfile.close();
}

void log(string& s, vs& where = consoleLog) { where.push_back(s); }

void loadModel(const string& filename) { inProgress; }

void saveModel(const string& modelName, const string& filename) { inProgress; }

void loadInstructions(const string& filename) {
  ifstream file(filename);
  if (not file.good()) {
    cerr << "error: the file \"" << filename << "\" could not be loaded." << endl;
    return;
  }
  string temp;
  while (getline(file, temp) and temp != "end") {
    vs comm = tokenize(temp);
    main_stream(comm);
  }
}

void loadData(const string& name, const string& filename) {
  inDatasets[name] = (int)datasets.size();
  datasets.push_back(new dataFrame());
  datasets.back()->read(filename);
  datasets.back()->setName(name);
}

void createModel(const string& modelname, int inDim, int outDim, int numLayers) {
  if (inModelList.find(modelname) != inModelList.end()) {
    cout << "Please use a name that is not already a model." << endl;
    return;
  }
  cout << "Creating model "+modelname+"." << endl;
  inModelList[modelname] = (int)modelList.size();
  modelList.push_back(new NNet(modelname,inDim,outDim,numLayers));
  //cout << "the segmentation fault is somehow past here?" << endl;
}

void showModelDetail(const string& modelname) {
  if (inModelList.find(modelname) == inModelList.end()) {
    cout << "error: please request details from an existing model" << endl;
    return;
  }
  NNet *model = modelList[inModelList[modelname]];
  cout << endl << "MODEL NAME: "+model->getName() << endl
       << "  Input dimension: " << model->getDim()[0] << endl
       << "  Output dimension: " << model->getDim()[1] << endl
       << "  Model depth: " << model->getDepth() << endl
       << "  Model parameters:" << endl;
  print(model->getWeights());
}

void printModelNames() {
  int modelNum = 0;
  for (auto p : inModelList) cout << ++modelNum << ": "+p.first << endl;
}

void modelEditSubconsole(const string& modelname) {
  if (inModelList.find(modelname) == inModelList.end()) {
    cout << "error: please use an existing model" << endl;
    return;
  }
  cout << "Editing model "+modelname << endl;
  cout << "[ME]> ";
  string temp;
  while (getline(cin,temp) and temp != "end") {
    vs comm = tokenize(temp);
    if (comm[0] == "addlayer") {
      if (comm.size() >= 3)
        modelList[inModelList[modelname]]->addLayer(stoi(comm[1]),activations[comm[2]]);
      else
        modelList[inModelList[modelname]]->addLayer();
    }
    else if (comm[0] == "editlayer") {
      if (comm.size() < 2)
        cout << "error: you must specify which layer to update" << endl;
      else if (stoi(comm[1]) > modelList[inModelList[modelname]]->getDepth() or stoi(comm[1]) < 0)
        cout << "error: please refer to an existing layer in this model" << endl;
      else {
        cout << "Editing layer "+comm[1] << endl;
        Layer *l = modelList[inModelList[modelname]]->operator[](stoi(comm[1]));
        cout << "[L"+comm[1]+"E]> ";
        while (getline(cin,temp) and temp != "end") {
          vs subcomm = tokenize(temp);
          if (subcomm[0] == "set") {
            if (subcomm.size() < 2)
              cout << "error: invalid setting operation" << endl;
            else if (subcomm[1] == "name") {
              if (subcomm.size() < 3)
                cout << "error: please enter a valid name for this layer" << endl;
              else l->setName(subcomm[2]);
            }
            else if (subcomm[1] == "activation") {
              if (subcomm.size() < 3)
                cout << "error: please enter a valid activation function for this layer" << endl;
              else l->setLayerActivation(activations[subcomm[2]]);
            }
            else if (subcomm[1] == "neurons") {
              if (subcomm.size() < 5)
                cout << "error: you need to specify the number of neurons, the input dimension of the layer, and the layer-wide activaiton function" << endl;
              else l->setupNeurons(stoi(subcomm[2]),stoi(subcomm[3]),activations[subcomm[4]]);
            }
            else unrComm;
          }
          cout << "[L"+comm[1]+"E]> ";
        }
      }
    }
    else unrComm;
    cout << "[ME]> ";
  }
}

void main_stream(const vs& c) {
  if (c.size() < 1) return;
  if (c[0] == "load") {
    if (c.size() < 2) unrComm;
    else if (c[1] == "model") loadModel(c[2]);
    else if (c[1] == "instructions") loadInstructions(c[2]);
    else if (c[1] == "data") {
      if (c.size() < 4) {
        cout << "error: load data needs a name for the dataset and a filepath" << endl;
        return;
      }
      loadData(c[2],c[3]);
    }
    else unrComm;
  }
  else if (c[0] == "save") {
    if (c.size() < 2) unrComm;
    else if (c[1] == "model") saveModel(c[2], c[3]);
    else unrComm;
  }
  else if (c[0] == "record") {
    if (recordTrigger) cout << "already recording!" << endl;
    else {
      recordTrigger = true;
      string d;
      cout << "[R]> ";
      while (getline(cin, d) and d != "end") {
        log(d, recordLog);
        vector<string> comm = tokenize(d);
        main_stream(comm);
        cout << "[R]> ";
      }
      recordTrigger = false;
      save_log((c.size() == 2 ? c[1] : "recording"), recordLog);
    }
  }
  else if (c[0] == "create") {
    if (c.size() < 2) {
      cout << "error: incomplete command. Please specify what you want to create" << endl;
      return;
    }
    else if (c[1] == "model") {
      if (modelList.size() >= MAX_NETS) {
        cout << "The model memory is full! Do you want to evict the last model in the queue? (y/n)" << endl
             << "The model to be evicted is "+modelList.front()->getName()+"." << endl;
        char c;
        while (cin >> c and c != 'y' and c != 'n') cout << "Please use 'y' for YES and 'n' for NO" << endl;
        if (c == 'n') {
          cout << "Did not create the model." << endl;
          return;
        }
        cout << "Erasing said model." << endl;
        inModelList.erase(modelList.front()->getName());
        // delete modelList.front();
        modelList.erase(modelList.begin());
      }
      if (c.size() < 6) {
        cout << "error: must give the new model a name, input and output dimensions, and number of layers" << endl;
        return;
      }
      createModel(c[2],stoi(c[3]),stoi(c[4]),stoi(c[5]));
      return;
    }
    else {
      cout << "error: invalid object creation" << endl;
      return;
    }
  }
  else if (c[0] == "show") {
    if (c[1] == "models")
      printModelNames();
    else if (c[1] == "data") {
      if (c.size() < 3) {
        cout << "error: incomplete call to show data" << endl;
        return;
      }
      if (inDatasets.find(c[2]) == inDatasets.end()) {
        cout << "error: please use an existing dataset name" << endl;
        return;
      }
      datasets[inDatasets[c[2]]]->print();
    }
    else if (c[1] == "model")
      showModelDetail(c[2]);
    else cout << "error: invalid object to show" << endl;
  }
  else if (c[0] == "edit") {
    if (c.size() < 2) {
      cout << "error: no model object to edit" << endl;
      return;
    }
    modelEditSubconsole(c[1]);
  }
  else unrComm;
}


void setups() {
  activations["tanh"] = tanh;
  activations["logistic"] = logistic;
  // activations["relu"] = relu;
}

int main() {
  setups();
  cout << "NeuralNetwork 0.0.0 console. Work by Atellas23." << endl << endl;
  cout << "> ";
  string s;
  while (getline(cin, s) and s != "end") {
    if (s.length() < 1) {
      unrComm;
      continue;
    }
    log(s);
    if (s[0] == '#') cout << '#' << endl; // Commentary check
    else {                                // Separate in different words (considered different when separated by a space ' ')
      vector<string> command = tokenize(s);
      main_stream(command);                   // Run command through the main command line stream
    }
    cout << "> ";
  }
  save_log("console.log");
}
