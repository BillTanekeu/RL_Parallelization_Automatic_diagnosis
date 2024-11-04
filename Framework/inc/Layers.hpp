#ifndef layers_h
#define layers_h
 
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>
#include <iomanip>
#include <mutex>
#include "ActivationFunctions.hpp"

using namespace std;

typedef unique_ptr<IActivationFunction> pActFunc;

enum class LayerType { dense, dropout };

class Layer {
protected:
  int inputN, outputN, Nthread;
  vector<vector<float>> weights;
  vector<vector<float>> weight_AdamParamsMT;
  vector<vector<float>> weight_AdamParamsVT;
  vector<float> biais_AdamParamsMT;
  vector<float> biais_AdamParamsVT;
  float AdamParamsBeta1;
  float AdamParamsBeta2;
  float AdamParamsEps;
  vector<vector<float>> init_weights;
  vector<float> init_biais;
  
  vector<vector<float>> prWeights;
  vector<float> biais;
  vector<float> prBiais;
  vector<vector<float>> outputNeurons;
  vector<vector<float>> neuronDelta;
  //float sum_exp_output;
  string s_afunc;

  random_device rng;

  function<float(const float &)> activation;
  function<float(const float &)> derivativeactivation;

  function<float(const float &, const float &, float &)> s_activation;
  function<float(const float &, const float &, float &)> s_derivativeactivation;
  pActFunc actfunc;

public:
  Layer();
  void copy_layer( const Layer &layer);
  float max_output;
  vector<vector<float>> Accumelated_grad;
  vector<float> AccNeuronDelta; 
  LayerType type;
  Activation atype;
  vector<vector<float>> no_activate_outputNeurons;
  mutex mtx;

  uniform_real_distribution<float> dist{-0.5, 0.5};
  int getOutputN();
  int getInputN();
  void saveParams(ofstream &saveFile);
  
  
  vector<vector<float>> &getWeights();

  void loadParams(vector<vector<float>> iWeights, vector<float> iBiais);

  int getMostProbable(int id_thread);
  pair<int, float> getMostProbableAndEntropy(int id_thread);
  int getParamNum();
  vector<float> getOutputsNeurons(int id_thread);
  float getNeuronVal(const int &index, int id_thread);
  void resetNeuronDelta();
  float activationFunction(const float &input);
  float derivativeactivationFunction(const float &input);
  float getMaxNeuronlVal(int id_thread);
  float getValWeight(int &iNeuron, int &iInput);
  float getValNeuronDelta(int &iNeuron, int id_thread);


  //vector<float> &operator()() { return outputNeurons; }
  //const vector<float> &operator()() const { return outputNeurons; }

  virtual void init(int nb_threads) = 0;
  virtual void fillInput(vector<float> &input, int id_thread, bool usePrevParams = false) = 0;
  virtual void calculateLayer(Layer &prevLayer, int id_thread, bool usePrevParams = false) = 0;
  virtual void rescaleWeights(const float &momentum, const float &rate, int batch_size) = 0;
  virtual void implicite_rescaleWeights(vector<vector<float>> &grad, vector<float> &delta, const float &momentum, const float &rate, int batch_size) = 0;
  virtual void reset(int nb_threads)=0;
  virtual void setNeuronDelta(const int &index, const int &target,int id_thread, const int &batch_size=1) = 0;
  virtual void updatePrevParams() = 0;
  virtual void addGradLastLayerUseCrossEntropy(Layer &prevLayer,const int &index, const float &target, int id_thread) = 0;
  virtual void addGradLastLayer(Layer &prevLayer,const int &index, const float &target, bool crossEntropy_softmax, int id_thread, float val = 1)=0;
  virtual void addGradHiddenLayer(Layer &prevLayer,Layer &nextLayer,int &index, int id_thread )=0;
  virtual void addGradFirstLayer(vector<float> &observation, Layer &nextLayer,int &index, int id_thread) =0;
  virtual void implicite_addGradLastLayerUseCrossEntropy(vector<vector<float>> &grad, vector<float> &delta ,Layer &prevLayer,const int &index, const float &target, int id_th)=0;
  virtual void implicite_addGradLastLayer(vector<vector<float>> &grad, vector<float> &delta, Layer &prevLayer,const int &index, const float &target, bool crossEntropy_softmax, int id_thread, float val = 1)=0;
  virtual void implicite_addGradHiddenLayer(vector<vector<float>> &grad, vector<float> &delta, Layer &prevLayer,Layer &nextLayer,int &index, int id_thread )=0;
  virtual void implicite_addGradFirstLayer(vector<vector<float>> &grad, vector<float> &delta, vector<float> &observation, Layer &nextLayer,int &index, int id_thread)=0;
  virtual void implicite_rescaleWeights_mutex_matrix(vector<vector<float>> &acc_grad, vector<float> &acc_delta, const float &momentum, const float &rate, int batch_size) = 0;

};

class Dense : public Layer {

public:
  Dense(int outputN, int inputN, Activation afuncType, ofstream &logFile, int id_thread);
  //void copy_layer( Layer &Dlayer);
  void reset(int nb_threads);
  void init(int nb_threads);
  void fillInput(vector<float> &input, int id_thread, bool usePrevParams = false);
  void calculateLayer(Layer &prevLayer, int id_thread, bool usePrevParams = false);
  void rescaleWeights(const float &momentum, const float &rate, int batch_size);
  void setNeuronDelta(const int &index, const int &target, int id_thread, const int &batch_size=1);
  
  //void set_sum_exp_output();
  void updatePrevParams();
  void addGradLastLayerUseCrossEntropy(Layer &prevLayer,const int &index, const float &target, int id_thread);
  void addGradLastLayer(Layer &prevLayer,const int &index, const float &target, bool crossEntropy_softmax, int id_thread, float val = 1);
  void addGradHiddenLayer(Layer &prevLayer,Layer &nextLayer,int &index, int id_thread );
  void addGradFirstLayer(vector<float> &observation, Layer &nextLayer,int &index, int id_thread);
  void implicite_addGradLastLayerUseCrossEntropy(vector<vector<float>> &grad, vector<float> &delta ,Layer &prevLayer,const int &index, const float &target, int id_th);
  void implicite_addGradLastLayer(vector<vector<float>> &grad, vector<float> &delta, Layer &prevLayer, const int &index, const float &target, bool crossEntropy_softmax, int id_thread, float val = 1);
  void implicite_addGradHiddenLayer(vector<vector<float>> &grad, vector<float> &delta, Layer &prevLayer,Layer &nextLayer,int &index, int id_thread );
  void implicite_addGradFirstLayer(vector<vector<float>> &grad, vector<float> &delta, vector<float> &observation, Layer &nextLayer,int &index, int id_thread);
  void implicite_rescaleWeights(vector<vector<float>> &grad, vector<float> &delta, const float &momentum, const float &rate, int batch_size);

  void implicite_rescaleWeights_mutex_matrix(vector<vector<float>> &acc_grad, vector<float> &acc_delta, const float &momentum, const float &rate, int batch_size);
  };
 

#endif