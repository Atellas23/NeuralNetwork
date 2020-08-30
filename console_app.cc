#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <queue>
#include "lib/neural"
using namespace std;
#define inProgress cout << "in progress" << endl
#define unrComm cout << "error: unrecognized command" << endl
#define MAX_NETS 10
vector<string> consoleLog;
vector<string> recordLog;
bool recordTrigger = false;

vector<NNet*> modelList;

set<string> names;

void save_log(string filename, vector<string>& where = consoleLog) {
  ofstream logfile(filename);
  for (auto s : where) logfile << s << endl;
  logfile.close();
}

void log(string& s, vector<string>& where = consoleLog) {
  where.push_back(s);
}

vector<string> tokenize(string& s) {
  vector<string> command;
  string word = "";
  int l = s.length();
  for (int j = 0; j <= l; ++j) {
    if (s[j] == ' ' or j == l) {
      command.push_back(word);
      word = "";
    }
    else word = word + s[j];
  }
  return command;
}

void loadModel(const string& filename) { inProgress; }

void loadInstructions(const string& filename) { inProgress; }

void saveModel(const string& modelName, const string& filename) { inProgress; }

void createModel(const string& modelname) { modelList.push_back(new NNet(modelname)); }

void printModelNames() {
  for (set<string>::iterator it = names.begin(); it != names.end(); ++it) cout << *it << endl;
}

void main_stream(const vector<string>& c) {
  if (c[0] == "load") {
    if (c.size() < 2) unrComm;
    else if (c[1] == "model") loadModel(c[3]);
    else if (c[1] == "instructions") loadInstructions(c[3]);
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
      while (getline(cin, d)) {
        log(d, recordLog);
        vector<string> comm = tokenize(d);
        if (comm[0] == "end") break;
        else main_stream(comm);
      }
      recordTrigger = false;
      save_log("recording", recordLog);
    }
  }
  else if (c[0] == "create") {
    if (c[1] == "model") {
      if (modelList.size() >= MAX_NETS) {
        cout << "The model heap is full! Do you want to evict the last model in the queue? (y/n)" << endl
             << "The model to be evicted is " << modelList.front()->getName() << '.' << endl;
        char c;
        while (cin >> c and c != 'y' and c != 'n') cout << "Please use 'y' for yes and 'n' for no." << endl;
        if (c == 'n') {
          cout << "Did not create the model." << endl;
          return;
        }
        cout << "Erasing the said model." << endl;
        delete modelList.front();
        modelList.erase(modelList.begin());
      }
      else {
        if (c[2] == "empty")
          modelList.push_back(new NNet());
        else {
          names.insert(c[3]);
          //modelList.push_back(new NNet(c[3]));
          createModel(c[3]);
        }
      }
    }
  }
  else if (c[0] == "show") {
    if (c[1] == "modelheap")
      printModelNames();
  }
  else unrComm;
}


int main() {
  cout << "NeuralNetwork 0.0.0 console. Work by Atellas23." << endl;
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
