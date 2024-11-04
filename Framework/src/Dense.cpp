#include "Layers.hpp"

  
Dense::Dense(int oN, int iN, Activation afunc, ofstream &logFile, int nb_thread) {

  inputN = iN;
  outputN = oN;
  Nthread = nb_thread;
  type = LayerType::dense; 
  atype = afunc;


  if (afunc == Activation::relu) {
    actfunc = make_unique<Relu>();
    s_afunc = "Relu";
  } else if (afunc == Activation::sigmoid) {
    actfunc = make_unique<Sigmoid>();
    s_afunc = "Sigmoid";
  } else if (afunc == Activation::tanh) {
    actfunc = make_unique<Tanh>();
    s_afunc = "Tanh";
  } else if (afunc == Activation::gauss) {
    actfunc = make_unique<Gauss>();
    s_afunc = "Gauss";
  } else if (afunc == Activation::bent) {
    actfunc = make_unique<Bent>();
    s_afunc = "Bent";
  } else if (afunc == Activation::softplus) {
    actfunc = make_unique<SoftPlus>();
    s_afunc = "SoftPlus";
  } else if (afunc == Activation::sinusoid) {
    actfunc = make_unique<Sinusoid>();
    s_afunc = "Sinusoid";
  } else if (afunc == Activation::identity) {
    actfunc = make_unique<Identity>();
    s_afunc = "Identity";
  } else if (afunc == Activation::isrlu) {
    actfunc = make_unique<ISRLU>();
    s_afunc = "ISRLU";
  }else if (afunc == Activation::softmax) {
    actfunc = make_unique<Softmax>();
    s_afunc = "Softmax";
  }else {
    exit(-1);
  }

  if(s_afunc == "Softmax"){
    s_activation = [=](const float &input, const float &sum_exp, float &max_output)
          { return actfunc->s_activation(input, sum_exp, max_output); };
    s_derivativeactivation = [=](const float &input,  const float &sum_exp, float &max_output) {
    return actfunc->s_derivativeactivation(input, sum_exp, max_output);
    };
  }else{
    activation = [=](const float &input) { return actfunc->activation(input); };
    derivativeactivation = [=](const float &input) {
    return actfunc->derivativeactivation(input);
  };
  }
 

  string msg = "Dense layer created with I/O dimensions " + to_string(inputN) +
               " " + to_string(outputN) +
               ", the activation function: " + s_afunc;
  cout << msg << endl;
  if (logFile.is_open())
    logFile << msg << endl;
}


void Dense::reset(int nb_threads){
  weights.clear();
  AccNeuronDelta.clear();
  Accumelated_grad.clear();
  weight_AdamParamsMT.clear();
  weight_AdamParamsVT.clear();
  biais.clear();
  biais_AdamParamsMT.clear();
  biais_AdamParamsVT.clear();
  outputNeurons.clear();
  no_activate_outputNeurons.clear();
  neuronDelta.clear();

  for (int inp = 0; inp < outputN; inp++){
    weights.push_back(vector<float>(inputN));
    Accumelated_grad.push_back(vector<float>(inputN));
    weight_AdamParamsMT.push_back(vector<float>(inputN));
    weight_AdamParamsVT.push_back(vector<float>(inputN));
    biais_AdamParamsMT.push_back(0);
    biais_AdamParamsVT.push_back(0);
    outputNeurons.push_back(vector<float>(nb_threads));
    no_activate_outputNeurons.push_back(vector<float>(nb_threads,0));
    neuronDelta.push_back(vector<float>(nb_threads, 0));
  }

    
  //neuronDelta.resize(outputN);
  AccNeuronDelta.resize(outputN);
  //outputNeurons.resize(outputN);
  //no_activate_outputNeurons.resize(outputN);
  biais.resize(outputN);

  weights.assign(init_weights.begin(), init_weights.end());
  biais.assign(init_biais.begin(), init_biais.end());
  
  AdamParamsBeta1 = 0.9;
  AdamParamsBeta2 = 0.999;
  AdamParamsEps = 10e-8;

  prWeights.assign(weights.begin(), weights.end());
  prBiais.assign(biais.begin(), biais.end());

}

void Dense::init(int nb_threads) {

  weights.clear();
  AccNeuronDelta.clear();
  Accumelated_grad.clear();
  weight_AdamParamsMT.clear();
  weight_AdamParamsVT.clear();
  biais.clear();
  biais_AdamParamsMT.clear();
  biais_AdamParamsVT.clear();
  outputNeurons.clear();
  no_activate_outputNeurons.clear();
  neuronDelta.clear();

  for (int inp = 0; inp < outputN; inp++){
    weights.push_back(vector<float>(inputN));
    Accumelated_grad.push_back(vector<float>(inputN));
    weight_AdamParamsMT.push_back(vector<float>(inputN));
    weight_AdamParamsVT.push_back(vector<float>(inputN));
    biais_AdamParamsMT.push_back(0);
    biais_AdamParamsVT.push_back(0);
    outputNeurons.push_back(vector<float>(nb_threads));
    no_activate_outputNeurons.push_back(vector<float>(nb_threads));
    neuronDelta.push_back(vector<float>(nb_threads, 0));
  }

    
  //neuronDelta.resize(outputN);
  AccNeuronDelta.resize(outputN);
  //outputNeurons.resize(outputN);
  //no_activate_outputNeurons.resize(outputN);
  biais.resize(outputN);

  for_each(weights.begin(), weights.end(), [&](auto &row) {
    generate(row.begin(), row.end(), [&]() { return dist(rng); });
  });

  for(auto &b : biais){
    b = dist(rng);
  }

  
  AdamParamsBeta1 = 0.9;
  AdamParamsBeta2 = 0.999;
  AdamParamsEps = 10e-8;

  prWeights.assign(weights.begin(), weights.end());
  prBiais.assign(biais.begin(), biais.end());

  init_weights.assign(weights.begin(), weights.end());
  init_biais.assign(biais.begin(), biais.end());

}

void Dense::fillInput(vector<float> &input,int id_th, bool usePrevParams) {

  vector<vector<float>> t_weights;
  vector<float> t_biais;

  if(usePrevParams){
    t_biais = prBiais;
    t_weights = prWeights;
  }else{
    t_biais = biais;
    t_weights = weights;
  }

  for (int iNeuron = 0; iNeuron < outputNeurons.size(); iNeuron++) {
    float val = 0.0;
    for (int iInput = 0; iInput < input.size(); iInput++) {
      val += input[iInput] * t_weights[iNeuron][iInput];
    }
    no_activate_outputNeurons[iNeuron][id_th] = val+ t_biais[iNeuron];
    outputNeurons[iNeuron][id_th] = activationFunction(val);
  }
}

void Dense::calculateLayer(Layer &prevLayer, int id_th, bool usePrevParams) {


  vector<vector<float>> t_weights;
  vector<float> t_biais;


  if(usePrevParams){
    t_biais = prBiais;
    t_weights = prWeights;
  }else{
    t_biais = biais;
    t_weights = weights;
  }

  if(s_afunc == "Softmax"){
    float max_out = 0;
    for (int iNeuron = 0; iNeuron < outputNeurons.size(); iNeuron++) {
      float val = 0.0;

      for (int iPrev = 0; iPrev < prevLayer.getOutputN(); iPrev++) {
        val += prevLayer.getNeuronVal(iPrev, id_th) * t_weights[iNeuron][iPrev];
      }
      no_activate_outputNeurons[iNeuron][id_th] = val+ t_biais[iNeuron];
      outputNeurons[iNeuron][id_th] = val+ t_biais[iNeuron];
     
      if(outputNeurons[iNeuron][id_th] > max_out) max_out = outputNeurons[iNeuron][id_th];
    }
    float sum_exp =0.001;
    for(int i=0; i< outputNeurons.size(); i++){
      sum_exp += exp(outputNeurons[i][id_th]- max_out);
    }

    for(int i = 0; i<outputNeurons.size(); i++){
      outputNeurons[i][id_th] = exp(outputNeurons[i][id_th] - max_out)/ sum_exp;
    }


  }else{
    for (int iNeuron = 0; iNeuron < outputNeurons.size(); iNeuron++) {
    float val = 0.0;

    for (int iPrev = 0; iPrev < prevLayer.getOutputN(); iPrev++) {
      val += prevLayer.getNeuronVal(iPrev, id_th) * t_weights[iNeuron][iPrev];
    }
    no_activate_outputNeurons[iNeuron][id_th] = val + t_biais[iNeuron];
    outputNeurons[iNeuron][id_th] = activationFunction(val + t_biais[iNeuron]);
   }

  }
  
  
}


void Dense::implicite_rescaleWeights(vector<vector<float>> &acc_grad, vector<float> &acc_delta, const float &momentum, const float &rate, int batch_size) {
  
  //cout<<"modification poids"<<endl;
  
  for (int out = 0; out < outputN; out++) {

    for (int inp = 0; inp < inputN; inp++) {
      float grad = acc_grad[out][inp]/batch_size;
      weight_AdamParamsMT[out][inp] = AdamParamsBeta1*weight_AdamParamsMT[out][inp] +
                                      (1 -AdamParamsBeta1)*grad;
      
      weight_AdamParamsVT[out][inp] = AdamParamsBeta2*weight_AdamParamsVT[out][inp] +
                                      (1 -AdamParamsBeta2)*pow(grad, 2);

      float mt, vt;
      mt = weight_AdamParamsMT[out][inp] /(1 - AdamParamsBeta1);
      vt = weight_AdamParamsVT[out][inp] /(1 - AdamParamsBeta2);

      mtx.lock();
      weights[out][inp] = weights[out][inp] -(rate*mt)/(sqrt(vt) + AdamParamsEps);
      mtx.unlock();

      //weights[out][inp] = weights[out][inp] -(rate*grad);
 
 
    }

    float grad = acc_delta[out]/batch_size;
    biais_AdamParamsMT[out] = AdamParamsBeta1*biais_AdamParamsMT[out] +
                                      (1 -AdamParamsBeta1)*grad;
      
    biais_AdamParamsVT[out] = AdamParamsBeta2*biais_AdamParamsVT[out] +
                                      (1 -AdamParamsBeta2)*pow(grad, 2);

    float mt, vt;
    mt = biais_AdamParamsMT[out] /(1 - AdamParamsBeta1);
    vt = biais_AdamParamsVT[out] /(1 - AdamParamsBeta2);
    mtx.lock();
    biais[out] = biais[out] -(rate*mt)/(sqrt(vt) + AdamParamsEps);
    mtx.unlock();
  }
  
}



void Dense::implicite_rescaleWeights_mutex_matrix(vector<vector<float>> &acc_grad, vector<float> &acc_delta, const float &momentum, const float &rate, int batch_size) {
  
  //cout<<"modification poids"<<endl;
  mtx.lock();  
  for (int out = 0; out < outputN; out++) {

    for (int inp = 0; inp < inputN; inp++) {
      float grad = acc_grad[out][inp]/batch_size;
      weight_AdamParamsMT[out][inp] = AdamParamsBeta1*weight_AdamParamsMT[out][inp] +
                                      (1 -AdamParamsBeta1)*grad;
      
      weight_AdamParamsVT[out][inp] = AdamParamsBeta2*weight_AdamParamsVT[out][inp] +
                                      (1 -AdamParamsBeta2)*pow(grad, 2);

      float mt, vt;
      mt = weight_AdamParamsMT[out][inp] /(1 - AdamParamsBeta1);
      vt = weight_AdamParamsVT[out][inp] /(1 - AdamParamsBeta2);

      weights[out][inp] = weights[out][inp] -(rate*mt)/(sqrt(vt) + AdamParamsEps);

      //weights[out][inp] = weights[out][inp] -(rate*grad);
 
 
    }

    float grad = acc_delta[out]/batch_size;
    biais_AdamParamsMT[out] = AdamParamsBeta1*biais_AdamParamsMT[out] +
                                      (1 -AdamParamsBeta1)*grad;
      
    biais_AdamParamsVT[out] = AdamParamsBeta2*biais_AdamParamsVT[out] +
                                      (1 -AdamParamsBeta2)*pow(grad, 2);

    float mt, vt;
    mt = biais_AdamParamsMT[out] /(1 - AdamParamsBeta1);
    vt = biais_AdamParamsVT[out] /(1 - AdamParamsBeta2);
    biais[out] = biais[out] -(rate*mt)/(sqrt(vt) + AdamParamsEps);
    mtx.unlock();
  }
  
}



void Dense::rescaleWeights(const float &momentum, const float &rate, int batch_size) {
  
  //cout<<"modification poids"<<endl;
  
  for (int out = 0; out < outputN; out++) {

    for (int inp = 0; inp < inputN; inp++) {
      float grad = Accumelated_grad[out][inp]/batch_size;
      weight_AdamParamsMT[out][inp] = AdamParamsBeta1*weight_AdamParamsMT[out][inp] +
                                      (1 -AdamParamsBeta1)*grad;
      
      weight_AdamParamsVT[out][inp] = AdamParamsBeta2*weight_AdamParamsVT[out][inp] +
                                      (1 -AdamParamsBeta2)*pow(grad, 2);

      float mt, vt;
      mt = weight_AdamParamsMT[out][inp] /(1 - AdamParamsBeta1);
      vt = weight_AdamParamsVT[out][inp] /(1 - AdamParamsBeta2);


      mtx.lock();
      weights[out][inp] = weights[out][inp] -(rate*mt)/(sqrt(vt) + AdamParamsEps);
      mtx.unlock();

      //weights[out][inp] = weights[out][inp] -(rate*grad);
 
 
    }

    float grad = AccNeuronDelta[out]/batch_size;
    biais_AdamParamsMT[out] = AdamParamsBeta1*biais_AdamParamsMT[out] +
                                      (1 -AdamParamsBeta1)*grad;
      
    biais_AdamParamsVT[out] = AdamParamsBeta2*biais_AdamParamsVT[out] +
                                      (1 -AdamParamsBeta2)*pow(grad, 2);

      float mt, vt;
      mt = biais_AdamParamsMT[out] /(1 - AdamParamsBeta1);
      vt = biais_AdamParamsVT[out] /(1 - AdamParamsBeta2);
    mtx.lock();
    biais[out] = biais[out] -(rate*mt)/(sqrt(vt) + AdamParamsEps);
    mtx.unlock();
  }
  
}



void Dense::updatePrevParams(){

  mtx.lock();
  prBiais.clear();
  prWeights.clear();
  prWeights.assign(weights.begin(), weights.end());
  prBiais.assign(biais.begin(), biais.end());
  mtx.unlock();
}


void Dense::addGradLastLayerUseCrossEntropy(Layer &prevLayer,const int &index, const float &target, int id_th){
  if(int(target) == index){
        neuronDelta[index][id_th] = outputNeurons[index][id_th] -1; //classification
  }else{
        neuronDelta[index][id_th] = outputNeurons[index][id_th]; // classification
  }

  if(neuronDelta[index][id_th] !=0){
    mtx.lock();
    for(int iNeuron=0; iNeuron< prevLayer.getOutputN(); iNeuron++){
      Accumelated_grad[index][iNeuron] += neuronDelta[index][id_th]* prevLayer.getNeuronVal(iNeuron, id_th);
    }
    AccNeuronDelta[index]+= neuronDelta[index][id_th];
    mtx.unlock();
  }
  
}



void Dense::implicite_addGradLastLayerUseCrossEntropy(vector<vector<float>> &grad, vector<float> &delta ,Layer &prevLayer,const int &index, const float &target, int id_th){
  if(int(target) == index){
        neuronDelta[index][id_th] = outputNeurons[index][id_th] -1; //classification
  }else{
        neuronDelta[index][id_th] = outputNeurons[index][id_th]; // classification
  }

  if(neuronDelta[index][id_th] !=0){
    
    for(int iNeuron=0; iNeuron< prevLayer.getOutputN(); iNeuron++){
      grad[index][iNeuron] += neuronDelta[index][id_th]* prevLayer.getNeuronVal(iNeuron, id_th);
    }
    delta[index]+=neuronDelta[index][id_th];
    
  }
  
}


void Dense::implicite_addGradLastLayer(vector<vector<float>> &grad, vector<float> &delta, Layer &prevLayer, const int &index, const float &target, bool crossEntropy_softmax, int id_th, float val){
  if(crossEntropy_softmax){ // classification with loss = crossEntropy and activation = softmax
    if(int(target) == index){
        neuronDelta[index][id_th] = (outputNeurons[index][id_th] -1)*val; // policy gradient
        //neuronDelta[index][id_th] = outputNeurons[index] -val; //classification
    }else{
        neuronDelta[index][id_th] = outputNeurons[index][id_th] *val;  //policy gradient
        //neuronDelta[index] = outputNeurons[index]; // classification
    }
  }else{
    if(int(target) == index){ // loss = mean square error
        //neuronDelta[index] = (outputNeurons[index]-val);
        neuronDelta[index][id_th] = (outputNeurons[index][id_th]-val)*derivativeactivationFunction(no_activate_outputNeurons[index][id_th]);
    }else{
        neuronDelta[index][id_th] = 0;
    }
  }
  
  if(neuronDelta[index][id_th] != 0){
    for(int iNeuron=0; iNeuron< prevLayer.getOutputN(); iNeuron++){
    grad[index][iNeuron] += neuronDelta[index][id_th]* prevLayer.getNeuronVal(iNeuron, id_th);

    }
    delta[index]+=neuronDelta[index][id_th];
  }
  
}


void Dense::implicite_addGradHiddenLayer(vector<vector<float>> &grad, vector<float> &delta, Layer &prevLayer,Layer &nextLayer,int &index, int id_th ){
  float sum =  0.0;
  for(int iNeuron=0; iNeuron<nextLayer.getOutputN(); iNeuron++ ){
    sum += nextLayer.getValNeuronDelta(iNeuron, id_th)*nextLayer.getValWeight(iNeuron, index);
  }
  neuronDelta[index][id_th] = derivativeactivationFunction(no_activate_outputNeurons[index][id_th])*sum;

  if(neuronDelta[index][id_th] != 0){
    for(int iNeuron=0; iNeuron< prevLayer.getOutputN(); iNeuron++){
    grad[index][iNeuron] += neuronDelta[index][id_th]* prevLayer.getNeuronVal(iNeuron, id_th);
    } 
    delta[index]+=neuronDelta[index][id_th];
    
  }
  
}

void Dense::implicite_addGradFirstLayer(vector<vector<float>> &grad, vector<float> &delta, vector<float> &observation, Layer &nextLayer,int &index, int id_th){
  float sum =  0.0;
  for(int iNeuron=0; iNeuron<nextLayer.getOutputN(); iNeuron++ ){
    sum += nextLayer.getValNeuronDelta(iNeuron, id_th)*nextLayer.getValWeight(iNeuron, index);
  }
  neuronDelta[index][id_th] = derivativeactivationFunction(no_activate_outputNeurons[index][id_th])*sum;

  if(neuronDelta[index][id_th] != 0){
      
    for(int iCaract=0; iCaract< observation.size(); iCaract++){
      grad[index][iCaract] += neuronDelta[index][id_th]* observation[iCaract];
    } 
    delta[index]+=neuronDelta[index][id_th];
    
  }
 
}
  

void Dense::addGradLastLayer(Layer &prevLayer,const int &index, const float &target,
           bool crossEntropy_softmax, int id_th, float val){
  
  if(crossEntropy_softmax){ // classification with loss = crossEntropy and activation = softmax
    if(int(target) == index){
        neuronDelta[index][id_th] = (outputNeurons[index][id_th] -1)*val; // policy gradient
        //neuronDelta[index][id_th] = outputNeurons[index] -val; //classification
    }else{
        neuronDelta[index][id_th] = outputNeurons[index][id_th] *val;  //policy gradient
        //neuronDelta[index] = outputNeurons[index]; // classification
    }
  }else{
    if(int(target) == index){ // loss = mean square error
        //neuronDelta[index] = (outputNeurons[index]-val);
        neuronDelta[index][id_th] = (outputNeurons[index][id_th]-val)*derivativeactivationFunction(no_activate_outputNeurons[index][id_th]);
    }else{
        neuronDelta[index][id_th] = 0;
    }
  }
  
  if(neuronDelta[index][id_th] != 0){
    mtx.lock();
    for(int iNeuron=0; iNeuron< prevLayer.getOutputN(); iNeuron++){
    Accumelated_grad[index][iNeuron] += neuronDelta[index][id_th]* prevLayer.getNeuronVal(iNeuron, id_th);

    }
    AccNeuronDelta[index]+=neuronDelta[index][id_th];
    mtx.unlock();
  }
   
}

void Dense::addGradHiddenLayer(Layer &prevLayer,Layer &nextLayer, int &index, int id_th ){
  float sum =  0.0;
  for(int iNeuron=0; iNeuron<nextLayer.getOutputN(); iNeuron++ ){
    sum += nextLayer.getValNeuronDelta(iNeuron, id_th)*nextLayer.getValWeight(iNeuron, index);
  }
  neuronDelta[index][id_th] = derivativeactivationFunction(no_activate_outputNeurons[index][id_th])*sum;

  if(neuronDelta[index][id_th] != 0){
    mtx.lock();
    for(int iNeuron=0; iNeuron< prevLayer.getOutputN(); iNeuron++){
    Accumelated_grad[index][iNeuron] += neuronDelta[index][id_th]* prevLayer.getNeuronVal(iNeuron, id_th);
    } 
    AccNeuronDelta[index]+=neuronDelta[index][id_th];
    mtx.unlock();
  }
  
}

void Dense::addGradFirstLayer(vector<float> &observation, Layer &nextLayer, int &index, int id_th){
  float sum =  0.0;
  for(int iNeuron=0; iNeuron<nextLayer.getOutputN(); iNeuron++ ){
    sum += nextLayer.getValNeuronDelta(iNeuron, id_th)*nextLayer.getValWeight(iNeuron, index);
  }
  neuronDelta[index][id_th] = derivativeactivationFunction(no_activate_outputNeurons[index][id_th])*sum;

  if(neuronDelta[index][id_th] != 0){
    mtx.lock();
    for(int iCaract=0; iCaract< observation.size(); iCaract++){
      Accumelated_grad[index][iCaract] += neuronDelta[index][id_th]* observation[iCaract];
    } 
    AccNeuronDelta[index]+=neuronDelta[index][id_th];
    mtx.unlock();
  }
 
}

void Dense::setNeuronDelta(const int &index, const int &target, int id_th, const int &batch_size) {

  if(int(target) == index){
          neuronDelta[index][id_th] += (outputNeurons[index][id_th]-1)/batch_size;
  }else{
          neuronDelta[index][id_th] += (outputNeurons[index][id_th])/batch_size;
  }
  
 //neuronDelta[index] = -target*log(outputNeurons[index])*derivativeactivationFunction(outputNeurons[index]); 
}
