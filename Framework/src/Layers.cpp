#include "Layers.hpp"
 
Layer::Layer(){
  
} 
void Layer::copy_layer( const Layer &layer){

  
  inputN = layer.inputN;
  if(outputN != layer.outputN){
    cout<<endl<<"Error impossible de copier le layer, les sorties sont differentes"<<endl;
    exit(0);
  }
  
  weights.assign(layer.weights.begin(), layer.weights.end());
  weight_AdamParamsMT.assign(layer.weight_AdamParamsMT.begin(), layer.weight_AdamParamsMT.end());
  weight_AdamParamsVT.assign(layer.weight_AdamParamsVT.begin(), layer.weight_AdamParamsVT.end());
  biais_AdamParamsMT.assign(layer.biais_AdamParamsMT.begin(), layer.biais_AdamParamsMT.end());
  biais_AdamParamsVT.assign(layer.biais_AdamParamsVT.begin(), layer.biais_AdamParamsVT.end());
  prWeights.assign(layer.prWeights.begin(), layer.prWeights.end());
  neuronDelta.assign(layer.neuronDelta.begin(), layer.neuronDelta.end());
  biais.assign(layer.biais.begin(), layer.biais.end());
  prBiais.assign(layer.prBiais.begin(), layer.prBiais.end());
  outputNeurons.assign(layer.outputNeurons.begin(), layer.outputNeurons.end());
  Accumelated_grad.assign(layer.Accumelated_grad.begin(), layer.Accumelated_grad.end());
  AccNeuronDelta.assign(layer.AccNeuronDelta.begin(), layer.AccNeuronDelta.end());
  no_activate_outputNeurons.assign(layer.no_activate_outputNeurons.begin(), layer.no_activate_outputNeurons.end());

  AdamParamsBeta1 = layer.AdamParamsBeta1;
  AdamParamsBeta2 = layer.AdamParamsBeta2;
  AdamParamsEps = layer.AdamParamsEps;
  max_output = layer.max_output;
  //sum_exp_output = layer.sum_exp_output;

  //s_afunc = layer.s_afunc;  
  //random_device rng;
  //activation = layer.activation;
  //derivativeactivation = layer.derivativeactivation;
  //s_activation = layer.s_activation;
  //s_derivativeactivation = layer.s_derivativeactivation;
  //type = layer.type;
  //atype = layer.atype;

  //actfunc = layer.actfunc;
  //pActFunc actfunc;
  
  //uniform_real_distribution<float> dist{-0.5, 0.5};
  
}




int Layer::getOutputN() { return outputN; }

int Layer::getInputN() { return inputN; }

void Layer::saveParams(ofstream &saveFile) {

  for (int out = 0; out < outputN; out++) {
    for (int inp = 0; inp < inputN; inp++) {
          saveFile << weights[out][inp] << " ";
    }
    saveFile<< biais[out]<<" ";
    saveFile << endl;
  }
} 

vector<float> Layer::getOutputsNeurons(int id_th){
  vector<float> vec;
  for (vector<float> line : outputNeurons){
    vec.push_back(line[id_th]);
  }
  return vec;
}

vector<vector<float>> &Layer::getWeights() { return weights; }

void Layer::loadParams(vector<vector<float>> iWeights, vector<float> iBiais){
  weights = iWeights;
  biais = iBiais;
  init_weights.assign(weights.begin(), weights.end());
  init_biais.assign(biais.begin(), biais.end());
  
  }


int Layer::getParamNum() {

  int params = 0;
  for (auto &row : weights) {
    params += row.size();
  }
    params += biais.size();
  return params;
}

float Layer::getNeuronVal(const int &index, int id_th) { return outputNeurons[index][id_th]; }

int Layer::getMostProbable(int id_th) {
  int id_max = 0;
  float max = outputNeurons[0][id_th];

  for(int i =0; i<outputNeurons.size(); i++){
    if( outputNeurons[i][id_th] > max){
      id_max = i;
      max = outputNeurons[i][id_th];
    }
  }

  return id_max;
}

pair<int, float> Layer::getMostProbableAndEntropy(int id_th){
  float entropy = 0;
  int id_max = 0;
  float max = outputNeurons[0][id_th];
  for(int i =0; i< outputNeurons.size(); i++){
    //cout<<" "<<outputNeurons[i][id_th];
    if(outputNeurons[i][id_th] > max){
      max = outputNeurons[i][id_th];
      id_max = i;
    }

    float end = outputNeurons[i][id_th];

    //cout<<endl<<"id th "<<id_th<<" out neurone "<<outputNeurons[i][id_th]<<" id "<<i<<endl;
    entropy -= (outputNeurons[i][id_th]*log(outputNeurons[i][id_th]+0.0000001));
    
    if(isnan(entropy)){
      cout<<"ERROR nan value getMostProbableAndEntropy : outputneurons "<<outputNeurons[i][id_th]
          <<" epy "<<entropy<<" end "<<end<<" id_th "<<id_th<<endl;
      exit(-1);
    }
    
  }

  return pair<int, float> (id_max, entropy);

}

float Layer::getMaxNeuronlVal( int id_th){
  return outputNeurons[getMostProbable(id_th)][id_th];
}

float Layer::getValWeight(int &iNeuron, int &iInput){return weights[iNeuron][iInput];}
float Layer::getValNeuronDelta(int &iNeuron, int id_th){return neuronDelta[iNeuron][id_th];}


float Layer::activationFunction(const float &input) {
  if(s_afunc == "Softmax"){
    return s_activation(input, 1, max_output);
  }
  return activation(input);
}

float Layer::derivativeactivationFunction(const float &input){
  if(s_afunc == "Softmax"){
    return s_derivativeactivation(input, 1, max_output);
  }
  return derivativeactivation(input);
}

void Layer::resetNeuronDelta() {
  fill(AccNeuronDelta.begin(), AccNeuronDelta.end(), 0);
  fill(neuronDelta.begin(), neuronDelta.end(), vector<float>(Nthread, 0));
  for (auto& inner_vec : Accumelated_grad) {
    std::fill(inner_vec.begin(), inner_vec.end(), 0);
  }
}
