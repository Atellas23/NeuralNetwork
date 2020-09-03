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
#define MAX_NETS 10
vs consoleLog;
vs recordLog;
bool recordTrigger = false;

map<string,int> inModelList;
vector<NNet*> modelList;
map<string,int> inDatasets;
vector<dataFrame*> datasets;

void main_stream(const vs& c);

void save_log(string filename, const vs& where = consoleLog) {
  ofstream logfile(filename);
  for (auto s : where) logfile << s << endl;
  logfile.close();
}

void log(string& s, vs& where = consoleLog) {
  where.push_back(s);
}

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

void createModel(const string& modelname) {
  if (inModelList.find(modelname) != inModelList.end()) {
    cout << "Please use a name that is not already a model." << endl;
    return;
  }
  cout << "Creating model "+modelname+". Please answer a few questions." << endl
       << "Do you have knowledge of the input and output dimensions of the model (y/n)? ";
  char option;
  while (cin >> option and option != 'y' and option != 'n') cout << "Please use 'y' for YES and 'n' for NO.\n";
  if (option == 'n') {
    inModelList[modelname] = (int)modelList.size();
    modelList.push_back(new NNet(modelname));
    return;
  }
  int in,out;
  cout << "Please let me know first the input and second the output dimension: ";
  cin >> in >> out;
  cout << "Do you have knowledge of the number of layers you want to use (y/n)? ";
  while (cin >> option and option != 'y' and option != 'n') cout << "Please use 'y' for YES and 'n' for NO.\n";
  if (option == 'n') {
    inModelList[modelname] = (int)modelList.size();
    modelList.push_back(new NNet(modelname,in,out));
    return;
  }
  cout << "Please let me know the desired number of layers: ";
  int layrs;
  cin >> layrs;
  inModelList[modelname] = (int)modelList.size();
  modelList.push_back(new NNet(modelname,in,out,layrs));
}

void printModelNames() {
  int modelNum = 0;
  for (auto p : inModelList) cout << ++modelNum << ": "+p.first << endl;
}

void main_stream(const vs& c) {
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
      cout << "error: incomplete command. Please specify what you want to create." << endl;
      return;
    }
    else if (c[1] == "model") {
      if (modelList.size() >= MAX_NETS) {
        cout << "The model memory is full! Do you want to evict the last model in the queue? (y/n)" << endl
             << "The model to be evicted is "+modelList.front()->getName()+"." << endl;
        char c;
        while (cin >> c and c != 'y' and c != 'n') cout << "Please use 'y' for yes and 'n' for no." << endl;
        if (c == 'n') {
          cout << "Did not create the model." << endl;
          return;
        }
        cout << "Erasing said model." << endl;
        inModelList.erase(modelList.front()->getName());
        // delete modelList.front();
        modelList.erase(modelList.begin());
      }
      createModel(c[3]);
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
    else cout << "error: invalid object to show" << endl;
  }
  else unrComm;
}


int main() {
  cout << "NeuralNetwork 0.0.0 console. Work by Atellas23." << endl << endl;
  cout << "> ";
  string s;
  while (getline(cin, s)) {
    log(s);
    if (s[0] == '#') cout << '#' << endl; // Commentary check
    else {                                // Separate in different words (considered different when separated by a space ' ')
      vector<string> command = tokenize(s);
      if (command[0] == "end") break;
      else main_stream(command);                   // Run command through the main command line stream
    }
    cout << "> ";
  }
  save_log("console.log");
}
