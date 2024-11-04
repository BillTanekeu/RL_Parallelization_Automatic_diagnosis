#ifndef mlp_h
#define mlp_h

#include "Layers.hpp"
#include "Reader.hpp"
 
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <vector>
#include <thread>

using namespace std;

typedef unique_ptr<Reader> pReader;
typedef unique_ptr<Layer> pLayer;
typedef unique_ptr<Dense> pDense;



 
class MLP {
  
public:
  int verbosity;
  int id_up;
  mutex mtx;
  bool sync_upd = false;
  int synch_update = 0;
  int policy_synch_update = 0;
  vector<int> position_th;
  //pair<int, int> dims;
  bool CrossEntropy_And_Softmax;
  vector<pLayer> layers;

  pReader reader;

  ofstream logFile;
  ofstream saveFile;
  ifstream loadFile;

  float learningRate, momentum, lmse;
  vector<float> error_rl;
  vector<float> error_cl;
  MLP(int verbosity = 0);
  void copy_mlp(const MLP &network);
  
  pair<int, int> dims;

  template <class ltype>
  void addLayer(int outputN, int inputN, Activation afuncType, int nb_thread);

  template <class ltype> void addLayer(int outputN, Activation afuncType, int nb_thread);

  template <class ltype> void addLayer(float rate);

  void network_copy(MLP *network);
  bool readImages(string filename, vector<vector<float>> &vec);
  bool readLabels(string filename, vector<float> &vec);
  bool read_data(string filename, vector<vector<float>> &train_features, vector<vector<float>> &test_features,
                               vector<float> &train_targets, vector<float> &test_targets);
  void compile(const float &learningRate, const float &momentum, int nb_threads);

  void reset(int nb_threads);
  float implicite_parTrainClassifier(vector<vector<float>> &train_x, vector<float> &train_y, int nb_thread);
  float Imp_parTrainClassifier(vector<vector<float>> train_x, vector<float> train_y, int nb_thread, int thread_id, bool mutexMat);
  
  float trainClassifier(vector<vector<float>> &train_x, vector<float> &train_y, int id_thread);
  float par_trainClassifier(vector<vector<float>> &train_x, vector<float> &train_y, bool usePos, int nb_thread);
  

  float update_params_rl(vector<vector<float>> &states,vector<vector<float>> &nextstates,
                         vector<float> &actions, vector<bool> &dones, 
                                vector<float> &rewards, float lamda, int id_thread);
  
  float Implicite_Agg_update_RL(vector<vector<float>> &states,vector<vector<float>> &nextstates,
                         vector<float> &actions, vector<bool> &dones, 
                                vector<float> &rewards, float lamda, int id_thread);
  
  float par_update_params_rl(vector<vector<float>> &states,vector<vector<float>> &nextstates,
                         vector<float> &actions, vector<bool> &dones, 
                                vector<float> &rewards, float lamda, int nb_thread);


  float synch_update_params_rl(vector<vector<float>> &states,vector<vector<float>> &nextstates,
                         vector<float> &actions, vector<bool> &dones, 
                                vector<float> &rewards, float lamda, int nb_threads, int id_th);

  float Imp_update_params_rl(vector<vector<float>> states,vector<vector<float>> nextstates,
                         vector<float> actions, vector<bool> dones, 
                                vector<float> rewards, float lamda, int nb_threads, int id_th, bool mutexMat);

  void Exp_synch_classifier(vector<vector<float>> &train_x, vector<float> &train_y,int nb_threads, int id_th);
  bool computeNetwork(vector<float> &image, int id_thread, float label = -1.0, bool usePrevParams = false);
  void updatePrevParams();
  int predict(vector<float> x, int nb_thread, bool choice =false);
  
  pair<int, float> getMostProbableAndEntropy(vector<float> x, int id_thread);

  void validateNetwork(vector<vector<float>> &test_x, vector<float> &test_y, int id_thread);

  void createLog(string fileName);
  void saveNetwork(string fileName);
  void loadNetwork(string fileName);

  

  template <typename Arg> void Msg(const Arg &arg) {
    cout << arg << endl;
    if (logFile.is_open()) {
      logFile << arg << endl;
    }
  }

  template <typename Arg, typename... Args>
  void Msg(const Arg &arg, const Args &... args) {
    cout << arg << " ";
    if (logFile.is_open()) {
      logFile << arg << " ";
    }
    Msg(args...);
  }
};
 
template <class ltype>
void MLP::addLayer(int outputN, int inputN, Activation afuncType, int id_thread) {
  if (layers.size() == 0 || layers.back()->getOutputN() == inputN)
    layers.push_back(make_unique<ltype>(outputN, inputN, afuncType, logFile, id_thread));
  else {
    Msg("Error: creating", layers.size() + 1, ". layer: input dimensions");
  }
}

template <class ltype> void MLP::addLayer(int outputN, Activation afuncType, int id_thread) {

  if (layers.size() != 0)
    layers.push_back(make_unique<ltype>(outputN, layers.back()->getOutputN(),
                                        afuncType, logFile, id_thread));
  else if (dims.first != 0 && dims.second != 0)
    layers.push_back(make_unique<ltype>(outputN, dims.second, afuncType, logFile, id_thread));
  else {
    //cout<<"dim first "<<dims.first<<" dim second "<<dims.second<<endl;
    Msg("iError: creating", layers.size() + 1, ". layer: input dimensions");
    
  }
}

template <class ltype> void MLP::addLayer(float rate) {

  if (layers.size() != 0)
    layers.push_back(make_unique<ltype>(rate, layers.back()->getOutputN(),
                                        layers.back()->getOutputN(), logFile));
  else {
    Msg("Error: creating", layers.size() + 1, ". layer: input dimensions");
  }
}


 struct Params_RL{
    int thread_id;
    int nb_thread;
    vector<vector<float>> states;
    vector<vector<float>> nextstates;
    vector<float> actions;
    vector<bool> dones; 
    vector<float> rewards;
    float lamda;
    int limit;
    MLP *mlp;
  } typedef Params_RL;
  
  struct Params_CL{
    int thread_id;
    int nb_thread;
    int limit;
    vector<vector<float>> train_x;
    vector<float> train_y;
    MLP *mlp;
  }typedef Params_CL;


#endif