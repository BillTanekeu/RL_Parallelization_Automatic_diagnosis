 
#include "MLP.hpp"
  
MLP::MLP(int verb) {
  
  verbosity = verb;
  dims.first = 0;
  dims.second = 0;

  pReader reader = make_unique<Reader>();

  reader->setVerbosity(verb);
}

void MLP::copy_mlp(const MLP &network){
  learningRate = network.learningRate;
  momentum = network.momentum;
  lmse = network.lmse;
  
  for (int i =0; i<layers.size(); i++){
    layers[i]->copy_layer(*network.layers[i]);
    
  }

}


void MLP::reset(int nb_threads){

  for_each(layers.begin(), layers.end(), [&](pLayer &layer) { layer->reset(nb_threads); });
 
}
void MLP::compile(const float &lR, const float &m, int nb_threads) {
  learningRate = lR;
  momentum = m;
  error_cl.resize(nb_threads);
  error_rl.resize(nb_threads);
  Msg("Setting up the network...");
  for_each(layers.begin(), layers.end(), [&](pLayer &layer) { layer->init(nb_threads); });
 
  Msg("Done.");
}

bool MLP::readImages(string fileName, vector<vector<float>> &vec) {

  pair<int, int> t_dims = reader->read_data(fileName, vec);
  
  Msg("dims ->>",dims.first," ", dims.second,"tdims-->", t_dims.first, " ", t_dims.second);
  if (t_dims.first == -1 && t_dims.second == -1) {
    Msg("Error: open image file ", fileName);
    return false;
  } else if ((dims.second != 0 && dims.second != t_dims.second)) {
    Msg("Error: different image dimensions");
    return false;
  } else if (dims.first == 0 && dims.second == 0)
    dims = t_dims;

  if (verbosity > 0) {
    Msg("Number of images: ", vec.size());
    Msg("Dimension of images: ", vec[0].size(), " (", dims.first, "x",
        dims.second, ")");
  }
  return true;
}

bool MLP::readLabels(string filename, vector<float> &vec) {

  if (!reader->read_data_Label(filename, vec)) {
    Msg("Error: reading train labels");
    return false;
  }

  if (verbosity > 0) {
    Msg("Number of labels: ", vec.size());
  }

  return true;
}

bool MLP::read_data(string filename,  vector<vector<float>> &train_features, vector<vector<float>> &test_features,
                               vector<float> &train_targets, vector<float> &test_targets){
  //ReturnOneHotEncoding res;
  MzReturnLoadCSV res;

  reader->Mz_load_csv(filename, res);

  //reader->One_hot_encoding(res_csv, res);

  vector<vector<float>>  features = res.features;
  vector<float> targets = res.targets;
 
  int n_train = int(features.size()*0.8);
  copy(features.begin(), features.begin() + n_train, back_inserter(train_features));
  copy(features.begin()+ n_train+1, features.end(), back_inserter(test_features));
  copy(targets.begin(), targets.begin() + n_train, back_inserter(train_targets));
  copy(targets.begin() + n_train+1, targets.end(), back_inserter(test_targets));

  dims.first = train_features.size();
  dims.second = train_features[1].size();
  return true;
}


void MLP::createLog(string fileName) {
  logFile.open(fileName);
  if (logFile.is_open()) {
    Msg("Logfile", fileName, "created");
  }
}

void MLP::saveNetwork(string fileName) {

  saveFile.open(fileName);

  int ndropouts = 0;
  for_each(layers.begin(), layers.end(), [&](auto &l) {
    if (l->type == LayerType::dropout) {
      ndropouts++;
    }
  });

  saveFile << layers.size() << " " << momentum << " " << learningRate << endl;
  
  for_each(layers.begin(), layers.end(), [&](auto &l) {
    if (l->type == LayerType::dropout) {
      return;
    }
    saveFile << "#LB" << endl;
    saveFile << int(l->type) << " " << int(l->atype) << " " << l->getInputN()
             << " " << l->getOutputN() << endl;
    l->saveParams(saveFile);
    saveFile << "#LE" << endl;
  });

  saveFile << "END" << endl;

  saveFile.close();

  Msg("Saving trained network to", fileName, "complete.");
}

void MLP::loadNetwork(string fileName) {

  loadFile.open(fileName);
  if (!loadFile.is_open()) {
    Msg("Error: failed to load network from file", fileName);
    exit(-1);
  }
  int lnum;
  loadFile >> lnum >> momentum >> learningRate;
  for (int iLayer = 0; iLayer < lnum; iLayer++) {
    char id[4];
    loadFile >> id;
    if (id == "END")
      break;
    int ltype;
    int atype;
    int t_inputN;
    int t_outputN;
    loadFile >> ltype >> atype >> t_inputN >> t_outputN;
    auto e_atype = static_cast<Activation>(atype);
    if (ltype == 0)
      addLayer<Dense>(t_outputN, t_inputN, e_atype, 1);
    
    vector<vector<float>> iWeights(t_outputN, vector<float>(t_inputN));
    vector<float> iBiais(t_outputN);
    
    for (int out = 0; out < t_outputN; out++)  {
      for (int inp = 0; inp < t_inputN; inp++) {
        loadFile >> iWeights[out][inp];
      }
      loadFile >> iBiais[out];
    }

    layers.back()->loadParams(iWeights, iBiais);
    loadFile >> id;
  }

  loadFile.close();

  Msg("Loading trained network to", fileName, "complete.");
}


void par_train_policy(Params_RL p){
  
  bool CrossEntropy_And_Softmax = false;
  float error = 0.0;
  int id_th = p.thread_id;
  
  for(int id_obs=p.thread_id; id_obs<p.states.size(); id_obs+=p.nb_thread ){
    
    float target;
    int id_bestValue;
  
    p.mlp->computeNetwork(p.nextstates[id_obs], id_th ,-1, true);
  
    id_bestValue = p.mlp->layers.back()->getMostProbable(id_th);
    if(p.dones[id_obs]){
          target = p.rewards[id_obs];
    }
    else{
      target = p.rewards[id_obs] + p.lamda* p.mlp->layers.back()->getNeuronVal(id_bestValue, id_th);

    }
    
    float qValue;
    p.mlp->computeNetwork(p.states[id_obs], id_th);
    qValue = p.mlp->layers.back()->getNeuronVal(p.actions[id_obs], id_th);
    error += pow((target - qValue), 2)/2;
  
    if(isnan(error)){
        cout<<"NAN error Update params, target: "<<target<<" qvalue: " <<qValue<< " reward: "<< p.rewards[id_obs]<<
           " neuronVal "<<p.mlp->layers.back()->getNeuronVal(id_bestValue, id_th)<<endl;
        exit(0);
      }


    for(int iNeuron=0; iNeuron < p.mlp->layers[p.mlp->layers.size()-1]->getOutputN(); iNeuron++  ){
      p.mlp->layers.back()->addGradLastLayer(*p.mlp->layers[p.mlp->layers.size()-2], iNeuron,
             p.actions[id_obs], CrossEntropy_And_Softmax, id_th, target);
    }
    
    for(int iLayer=p.mlp->layers.size()-2; iLayer>0; iLayer--){
      for(int iNeuron = 0; iNeuron < p.mlp->layers[iLayer]->getOutputN(); iNeuron++){
        p.mlp->layers[iLayer]->addGradHiddenLayer(*p.mlp->layers[iLayer-1], *p.mlp->layers[iLayer+1], iNeuron, id_th);
      }
    }
    for(int iNeuron = 0; iNeuron< p.mlp->layers[0]->getOutputN(); iNeuron++){
        p.mlp->layers[0]->addGradFirstLayer(p.states[id_obs], *p.mlp->layers[1], iNeuron, id_th);
    }

  }
  p.mlp->error_rl[id_th] = error;
  
}



float MLP::par_update_params_rl(vector<vector<float>> &states,vector<vector<float>> &nextstates,
                         vector<float> &actions, vector<bool> &dones, 
                                vector<float> &rewards, float lamda, int nb_thread){

    Params_RL p;

    p.states = states;
    p.nextstates = nextstates;
    p.actions = actions;
    p.dones = dones;
    p.rewards = rewards;
    p.lamda = lamda;
    p.nb_thread = nb_thread;
    p.mlp = this;

    vector<thread> threads;
    threads.resize(nb_thread);
    error_rl.resize(nb_thread);
    
    for( int t =0; t < nb_thread; t++){
      p.thread_id = t;
      threads[t] = thread(par_train_policy, p);
    }

    for(int t=0; t<nb_thread; t++) {
		  threads[t].join();
	  }
    for (int iLayer = layers.size() - 1; iLayer >= 0; iLayer--) {

      layers[iLayer]->rescaleWeights(momentum, learningRate, states.size());
      layers[iLayer]->resetNeuronDelta();

    }
  
  float error = 0;
  for( float err : error_rl) error += err;

  return error/states.size();
  
  }



float MLP::synch_update_params_rl(vector<vector<float>> &states,vector<vector<float>> &nextstates,
                         vector<float> &actions, vector<bool> &dones, 
                                vector<float> &rewards, float lamda, int nb_threads, int id_th){
  
  CrossEntropy_And_Softmax = false;
  float error = 0.0;
  int nEpoch = 2;
  
  for(int id_obs=id_th; id_obs<states.size(); id_obs+=nb_threads ){
   
    float target;
    int id_bestValue;
    computeNetwork(nextstates[id_obs], id_th ,-1, true);
    id_bestValue = layers.back()->getMostProbable(id_th);
    if(dones[id_obs]){
          target = rewards[id_obs];
    }
    else{
      target = rewards[id_obs] + lamda*layers.back()->getNeuronVal(id_bestValue, id_th);

    }
    
    float qValue;
    computeNetwork(states[id_obs], id_th);
    qValue = layers.back()->getNeuronVal(actions[id_obs], id_th);
    this->error_rl[id_th] += pow((target - qValue), 2)/2;
    //cout<<endl<<"target "<<target<<" qval "<<qValue<<" error "<<error<<endl;

    if(isnan(error)){
        cout<<"NAN error Update params, target: "<<target<<" qvalue: " <<qValue<< " reward: "<< rewards[id_obs]<<
           " neuronVal "<<layers.back()->getNeuronVal(id_bestValue, id_th)<<endl;
        exit(0);
      }

    
    for(int iNeuron=0; iNeuron < layers[layers.size()-1]->getOutputN(); iNeuron++  ){
      layers.back()->addGradLastLayer(*layers[layers.size()-2], iNeuron,
             actions[id_obs], CrossEntropy_And_Softmax, id_th, target);
    }
    
    for(int iLayer=layers.size()-2; iLayer>0; iLayer--){
      for(int iNeuron = 0; iNeuron < layers[iLayer]->getOutputN(); iNeuron++){
        layers[iLayer]->addGradHiddenLayer(*layers[iLayer-1], *layers[iLayer+1], iNeuron, id_th);
      }
    }
    for(int iNeuron = 0; iNeuron< layers[0]->getOutputN(); iNeuron++){
        layers[0]->addGradFirstLayer(states[id_obs], *layers[1], iNeuron, id_th);
    }

    

  }
  
  this->mtx.lock();
  this->policy_synch_update += 1;
  this->mtx.unlock();
  //cout<<"sync policy "<< this->policy_synch_update<< " nb threads "<<nb_threads<<endl;

  if(this->policy_synch_update == nb_threads){
     
    for (int iLayer = layers.size() - 1; iLayer >= 0; iLayer--) {

      layers[iLayer]->rescaleWeights(momentum, learningRate, states.size());
      layers[iLayer]->resetNeuronDelta();

    } 
    error = 0;
    for( float err : error_rl) error += err;
    error_rl.clear();
    error_rl.resize(nb_threads);

    this->policy_synch_update = 0;
    //return error;
      //cout<<"Epoch "<<i<<" error "<<error/states.size()<<" nbthread "<<nb_threads<<endl;
  

  }
  
  //exit(0);
  return error/states.size();

}



void implicite_Agg_par_train_policy(Params_RL p){
  
  bool CrossEntropy_And_Softmax = false;
  float error = 0.0;
  int id_th = p.thread_id;
  int Nlayer =p.mlp->layers.size();
  int batch_size = 10;
  vector<vector<vector<float>>> grad;
  vector<vector<float>> delta;
   
  grad.resize(Nlayer);
  delta.resize(Nlayer);

  for( int i =0; i < Nlayer; i++){
    grad[i].assign(p.mlp->layers[i]->Accumelated_grad.begin(),p.mlp->layers[i]->Accumelated_grad.end());
    delta[i].assign(p.mlp->layers[i]->AccNeuronDelta.begin(), p.mlp->layers[i]->AccNeuronDelta.end());
  }


  for(int id_obs=p.thread_id; id_obs<p.states.size(); id_obs+=p.nb_thread ){
    //cout<<"thread "<<p.thread_id<<" id obs "<<id_obs<<endl;  
    float target;
    int id_bestValue;
    p.mlp->computeNetwork(p.nextstates[id_obs], id_th ,-1, true);
    id_bestValue = p.mlp->layers.back()->getMostProbable(id_th);
    if(p.dones[id_obs]){
          target = p.rewards[id_obs];
    }
    else{
      target = p.rewards[id_obs] + p.lamda* p.mlp->layers.back()->getNeuronVal(id_bestValue, id_th);

    }
    
    float qValue;
    p.mlp->computeNetwork(p.states[id_obs], id_th);
    qValue = p.mlp->layers.back()->getNeuronVal(p.actions[id_obs], id_th);
    error += pow((target - qValue), 2)/2;
    //cout<<endl<<"target "<<target<<" qval "<<qValue<<" error "<<endl;
    if(isnan(error)){
        cout<<"NAN error Update params, target: "<<target<<" qvalue: " <<qValue<< " reward: "<< p.rewards[id_obs]<<
           " neuronVal "<<p.mlp->layers.back()->getNeuronVal(id_bestValue, id_th)<<endl;
        exit(0);
      }

    for(int iNeuron=0; iNeuron < p.mlp->layers[p.mlp->layers.size()-1]->getOutputN(); iNeuron++  ){
      p.mlp->layers.back()->implicite_addGradLastLayer(grad.back(), delta.back(), *p.mlp->layers[p.mlp->layers.size()-2], iNeuron,
             p.actions[id_obs], CrossEntropy_And_Softmax, id_th, target);
    }

    for(int iLayer=p.mlp->layers.size()-2; iLayer>0; iLayer--){
      for(int iNeuron = 0; iNeuron < p.mlp->layers[iLayer]->getOutputN(); iNeuron++){
        p.mlp->layers[iLayer]->implicite_addGradHiddenLayer(grad[iLayer], delta[iLayer],*p.mlp->layers[iLayer-1], *p.mlp->layers[iLayer+1], iNeuron, id_th);
      }
    }

    for(int iNeuron = 0; iNeuron< p.mlp->layers[0]->getOutputN(); iNeuron++){
        p.mlp->layers[0]->implicite_addGradFirstLayer(grad[0], delta[0], p.states[id_obs], *p.mlp->layers[1], iNeuron, id_th);
        
    }
    
  }

  for (int iLayer = Nlayer - 1; iLayer >= 0; iLayer--) {

      p.mlp->layers[iLayer]->implicite_rescaleWeights(grad[iLayer], delta[iLayer], p.mlp->momentum, p.mlp->learningRate, p.states.size()/p.nb_thread);
        //layers[iLayer]->resetNeuronDelta();
      grad[iLayer].assign(p.mlp->layers[iLayer]->Accumelated_grad.begin(),p.mlp->layers[iLayer]->Accumelated_grad.end());
      delta[iLayer].assign(p.mlp->layers[iLayer]->AccNeuronDelta.begin(), p.mlp->layers[iLayer]->AccNeuronDelta.end());
    }


  p.mlp->error_rl[id_th] = error;
  
}



float MLP::Imp_update_params_rl(vector<vector<float>> states,vector<vector<float>> nextstates,
                         vector<float> actions, vector<bool> dones, 
                                vector<float> rewards, float lamda, int nb_threads, int id_th, bool mutexMat){

  bool CrossEntropy_And_Softmax = false;
  float error = 0.0;
  int Nlayer =this->layers.size();
  int batch_size = 10;
  vector<vector<vector<float>>> grad;
  vector<vector<float>> delta;
   
  grad.resize(Nlayer);
  delta.resize(Nlayer);

  for( int i =0; i < Nlayer; i++){
    grad[i].assign(this->layers[i]->Accumelated_grad.begin(),this->layers[i]->Accumelated_grad.end());
    delta[i].assign(this->layers[i]->AccNeuronDelta.begin(), this->layers[i]->AccNeuronDelta.end());
  }


  for(int id_obs=id_th; id_obs<states.size(); id_obs+=nb_threads ){
    /*
    cout<<"thread "<<id_th<<" id obs "<<id_obs<<" len state "
    <<states.size()<<" n "<<nextstates.size()<<" a "<<actions.size()<<" d "<<dones.size()
    << " r "<<rewards.size()  <<endl;  */
    
    float target;
    int id_bestValue;
    this->computeNetwork(nextstates[id_obs], id_th ,-1, true);
    id_bestValue = this->layers.back()->getMostProbable(id_th);
    if(dones[id_obs]){
          target = rewards[id_obs];
    }
    else{
      target = rewards[id_obs] + lamda* this->layers.back()->getNeuronVal(id_bestValue, id_th);

    }
    
    float qValue;
    this->computeNetwork(states[id_obs], id_th);
    qValue = this->layers.back()->getNeuronVal(actions[id_obs], id_th);
    error += pow((target - qValue), 2)/2;
    //cout<<endl<<"target "<<target<<" qval "<<qValue<<" error "<<endl;
    if(isnan(error)){
        cout<<"NAN error Update params, target: "<<target<<" qvalue: " <<qValue<< " reward: "<< rewards[id_obs]<<
           " neuronVal "<<this->layers.back()->getNeuronVal(id_bestValue, id_th)<<endl;
        exit(0);
      }

    for(int iNeuron=0; iNeuron < this->layers[this->layers.size()-1]->getOutputN(); iNeuron++  ){
      this->layers.back()->implicite_addGradLastLayer(grad.back(), delta.back(), *this->layers[this->layers.size()-2], iNeuron,
             actions[id_obs], CrossEntropy_And_Softmax, id_th, target);
    }

    for(int iLayer=this->layers.size()-2; iLayer>0; iLayer--){
      for(int iNeuron = 0; iNeuron < this->layers[iLayer]->getOutputN(); iNeuron++){
        this->layers[iLayer]->implicite_addGradHiddenLayer(grad[iLayer], delta[iLayer],*this->layers[iLayer-1], *this->layers[iLayer+1], iNeuron, id_th);
      }
    }

    for(int iNeuron = 0; iNeuron< this->layers[0]->getOutputN(); iNeuron++){
        this->layers[0]->implicite_addGradFirstLayer(grad[0], delta[0], states[id_obs], *this->layers[1], iNeuron, id_th);
        
    }
    
  }

  if(mutexMat){
    for (int iLayer = Nlayer - 1; iLayer >= 0; iLayer--) {

      this->layers[iLayer]->implicite_rescaleWeights_mutex_matrix(grad[iLayer], delta[iLayer], this->momentum, this->learningRate, states.size()/nb_threads);
        //layers[iLayer]->resetNeuronDelta();
      grad[iLayer].assign(this->layers[iLayer]->Accumelated_grad.begin(),this->layers[iLayer]->Accumelated_grad.end());
      delta[iLayer].assign(this->layers[iLayer]->AccNeuronDelta.begin(), this->layers[iLayer]->AccNeuronDelta.end());
    }


  }else{
    for (int iLayer = Nlayer - 1; iLayer >= 0; iLayer--) {

      this->layers[iLayer]->implicite_rescaleWeights(grad[iLayer], delta[iLayer], this->momentum, this->learningRate, states.size()/nb_threads);
        //layers[iLayer]->resetNeuronDelta();
      grad[iLayer].assign(this->layers[iLayer]->Accumelated_grad.begin(),this->layers[iLayer]->Accumelated_grad.end());
      delta[iLayer].assign(this->layers[iLayer]->AccNeuronDelta.begin(), this->layers[iLayer]->AccNeuronDelta.end());
    }

  }
  

  this->error_rl[id_th] = error;
  
  return error;
}
  


float MLP::Implicite_Agg_update_RL(vector<vector<float>> &states,vector<vector<float>> &nextstates,
                         vector<float> &actions, vector<bool> &dones, 
                                vector<float> &rewards, float lamda, int nb_thread){
    
      Params_RL p;

      p.states = states;
      p.nextstates = nextstates;
      p.actions = actions;
      p.dones = dones;
      p.rewards = rewards;
      p.lamda = lamda;
      p.nb_thread = nb_thread;
      p.mlp = this;

      vector<thread> threads;
      threads.resize(nb_thread);
      error_rl.resize(nb_thread);
      
      for( int t =0; t < nb_thread; t++){
        p.thread_id = t;
        threads[t] = thread(implicite_Agg_par_train_policy, p);
      }

      for(int t=0; t<nb_thread; t++) {
        threads[t].join();
      }
      /*
      for (int iLayer = layers.size() - 1; iLayer >= 0; iLayer--) {

        layers[iLayer]->rescaleWeights(momentum, learningRate, states.size());
        layers[iLayer]->resetNeuronDelta();

      }*/
    
    float error = 0;
    for( float err : error_rl) error += err;

    return error/states.size();
  
  
  }



float MLP::update_params_rl(vector<vector<float>> &states,vector<vector<float>> &nextstates,
                         vector<float> &actions, vector<bool> &dones, 
                                vector<float> &rewards, float lamda, int id_th){
  
  CrossEntropy_And_Softmax = false;
  float error = 0.0;
  
  for(int id_obs=0; id_obs<states.size(); id_obs++ ){
   
    float target;
    int id_bestValue;
    computeNetwork(nextstates[id_obs], id_th ,-1, true);
    id_bestValue = layers.back()->getMostProbable(id_th);
    if(dones[id_obs]){
          target = rewards[id_obs];
    }
    else{
      target = rewards[id_obs] + lamda*layers.back()->getNeuronVal(id_bestValue, id_th);

    }
    
    float qValue;
    computeNetwork(states[id_obs], id_th);
    qValue = layers.back()->getNeuronVal(actions[id_obs], id_th);
    error += pow((target - qValue), 2)/2;
    //cout<<endl<<"target "<<target<<" qval "<<qValue<<" error "<<error<<endl;

    if(isnan(error)){
        cout<<"NAN error Update params, target: "<<target<<" qvalue: " <<qValue<< " reward: "<< rewards[id_obs]<<
           " neuronVal "<<layers.back()->getNeuronVal(id_bestValue, id_th)<<endl;
        exit(0);
      }

    
    for(int iNeuron=0; iNeuron < layers[layers.size()-1]->getOutputN(); iNeuron++  ){
      layers.back()->addGradLastLayer(*layers[layers.size()-2], iNeuron,
             actions[id_obs], CrossEntropy_And_Softmax, id_th, target);
    }
    
    for(int iLayer=layers.size()-2; iLayer>0; iLayer--){
      for(int iNeuron = 0; iNeuron < layers[iLayer]->getOutputN(); iNeuron++){
        layers[iLayer]->addGradHiddenLayer(*layers[iLayer-1], *layers[iLayer+1], iNeuron, id_th);
      }
    }
    for(int iNeuron = 0; iNeuron< layers[0]->getOutputN(); iNeuron++){
        layers[0]->addGradFirstLayer(states[id_obs], *layers[1], iNeuron, id_th);
    }

    

  }



  for (int iLayer = layers.size() - 1; iLayer >= 0; iLayer--) {

      layers[iLayer]->rescaleWeights(momentum, learningRate, states.size());
      layers[iLayer]->resetNeuronDelta();

    } 

  //cout<<endl<<"end"<<endl; 
  return error/states.size();

}



void implicite_Par_trainClassifier(Params_CL p){

  bool CrossEntropy_And_Softmax = true;
  float accuracy = 0.0;
  float error = 0.0;
  int id_th = p.thread_id;
  int batch_size = 10;
  vector<vector<vector<float>>> grad;
  vector<vector<float>> delta;
  int Nlayer =p.mlp->layers.size(); 
  grad.resize(Nlayer);
  delta.resize(Nlayer);

  for( int i =0; i < Nlayer; i++){
    grad[i].assign(p.mlp->layers[i]->Accumelated_grad.begin(),p.mlp->layers[i]->Accumelated_grad.end());
    delta[i].assign(p.mlp->layers[i]->AccNeuronDelta.begin(), p.mlp->layers[i]->AccNeuronDelta.end());
  }

  int cpt = 0;
  for(int id_obs=p.thread_id; id_obs<p.train_x.size(); id_obs+= p.nb_thread){

    
    if(p.mlp->computeNetwork(p.train_x[id_obs],id_th, p.train_y[id_obs], false)) accuracy +=1;
    float target = p.train_y[id_obs];

    for (int iNeuron = 0; iNeuron < p.mlp->layers.back()->getOutputN(); iNeuron++) {
              
      if(target == iNeuron){
        error += (-log(p.mlp->layers.back()->getNeuronVal(iNeuron, id_th)+ 0.0000005));
      }  
      p.mlp->layers.back()->implicite_addGradLastLayerUseCrossEntropy(grad.back(), delta.back(),*p.mlp->layers[p.mlp->layers.size()-2], iNeuron, target, id_th);
    }

    for(int iLayer=p.mlp->layers.size()-2; iLayer>0; iLayer--){
      for(int iNeuron = 0; iNeuron < p.mlp->layers[iLayer]->getOutputN(); iNeuron++){
        p.mlp->layers[iLayer]->implicite_addGradHiddenLayer(grad[iLayer], delta[iLayer],*p.mlp->layers[iLayer-1], *p.mlp->layers[iLayer+1], iNeuron, id_th);
      }
    }
    
    for(int iNeuron = 0; iNeuron< p.mlp->layers[0]->getOutputN(); iNeuron++){
        p.mlp->layers[0]->implicite_addGradFirstLayer(grad[0], delta[0], p.train_x[id_obs], *p.mlp->layers[1], iNeuron, id_th);
    }

    cpt +=1;

    if(cpt % batch_size == 0){

      for (int iLayer = Nlayer - 1; iLayer >= 0; iLayer--) {
        p.mlp->layers[iLayer]->implicite_rescaleWeights(grad[iLayer], delta[iLayer], p.mlp->momentum, p.mlp->learningRate, batch_size);
        grad[iLayer].assign(p.mlp->layers[iLayer]->Accumelated_grad.begin(),p.mlp->layers[iLayer]->Accumelated_grad.end());
        delta[iLayer].assign(p.mlp->layers[iLayer]->AccNeuronDelta.begin(), p.mlp->layers[iLayer]->AccNeuronDelta.end());
          //layers[iLayer]->resetNeuronDelta();
      }
    }
    

  }
  

  p.mlp->error_cl[p.thread_id] = error;


}



float MLP::Imp_parTrainClassifier(vector<vector<float>> train_x, vector<float> train_y, int nb_thread, int thread_id, bool mutexMat){


  bool CrossEntropy_And_Softmax = true;
  float accuracy = 0.0;
  float error = 0.0;
  int id_th = thread_id;
  int batch_size = 10;

  vector<vector<vector<float>>> grad;
  vector<vector<float>> delta;
  int Nlayer =this->layers.size(); 
  grad.resize(Nlayer);
  delta.resize(Nlayer);

  for( int i =0; i < Nlayer; i++){
    grad[i].assign(this->layers[i]->Accumelated_grad.begin(),this->layers[i]->Accumelated_grad.end());
    delta[i].assign(this->layers[i]->AccNeuronDelta.begin(), this->layers[i]->AccNeuronDelta.end());
  }

  int cpt = 0;
  for(int id_obs=thread_id; id_obs<train_x.size(); id_obs+= nb_thread){

    //cout<<"thread "<<thread_id<<" size data "<<train_x.size()<<" idobs "<<id_obs<<endl;    
    if(this->computeNetwork(train_x[id_obs],id_th, train_y[id_obs], false)) accuracy +=1;
    float target = train_y[id_obs];

    for (int iNeuron = 0; iNeuron < this->layers.back()->getOutputN(); iNeuron++) {
              
      if(target == iNeuron){
        error += (-log(this->layers.back()->getNeuronVal(iNeuron, id_th)+ 0.0000005));
      }  
      this->layers.back()->implicite_addGradLastLayerUseCrossEntropy(grad.back(), delta.back(),*this->layers[this->layers.size()-2], iNeuron, target, id_th);
    }

    for(int iLayer=this->layers.size()-2; iLayer>0; iLayer--){
      for(int iNeuron = 0; iNeuron < this->layers[iLayer]->getOutputN(); iNeuron++){
        this->layers[iLayer]->implicite_addGradHiddenLayer(grad[iLayer], delta[iLayer],*this->layers[iLayer-1], *this->layers[iLayer+1], iNeuron, id_th);
      }
    }
    
    for(int iNeuron = 0; iNeuron< this->layers[0]->getOutputN(); iNeuron++){
        this->layers[0]->implicite_addGradFirstLayer(grad[0], delta[0], train_x[id_obs], *this->layers[1], iNeuron, id_th);
    }

    cpt +=1;

    if(cpt % batch_size == 0){

      if(mutexMat){

        for (int iLayer = Nlayer - 1; iLayer >= 0; iLayer--) {
        this->layers[iLayer]->implicite_rescaleWeights_mutex_matrix(grad[iLayer], delta[iLayer], this->momentum, this->learningRate, batch_size);
        grad[iLayer].assign(this->layers[iLayer]->Accumelated_grad.begin(),this->layers[iLayer]->Accumelated_grad.end());
        delta[iLayer].assign(this->layers[iLayer]->AccNeuronDelta.begin(), this->layers[iLayer]->AccNeuronDelta.end());
          //layers[iLayer]->resetNeuronDelta();
        }

      }else{
        for (int iLayer = Nlayer - 1; iLayer >= 0; iLayer--) {
        this->layers[iLayer]->implicite_rescaleWeights(grad[iLayer], delta[iLayer], this->momentum, this->learningRate, batch_size);
        grad[iLayer].assign(this->layers[iLayer]->Accumelated_grad.begin(),this->layers[iLayer]->Accumelated_grad.end());
        delta[iLayer].assign(this->layers[iLayer]->AccNeuronDelta.begin(), this->layers[iLayer]->AccNeuronDelta.end());
          //layers[iLayer]->resetNeuronDelta();
        }

      }
      
    }

    

  }
  

  this->error_cl[thread_id] = error;

  return error;

}
  


float MLP::implicite_parTrainClassifier(vector<vector<float>> &train_x, vector<float> &train_y, int nb_thread){
   Params_CL p;

    p.train_x = train_x;
    p.train_y = train_y;
    p.nb_thread = nb_thread;
    p.mlp = this;
    
    vector<thread> threads;
    threads.resize(nb_thread);
    error_cl.resize(nb_thread);
    
    for( int t =0; t < nb_thread; t++){
      p.thread_id = t;
      threads[t] = thread(implicite_Par_trainClassifier, p);
    }


    for(int t=0; t<nb_thread; t++) {
		  threads[t].join();
	  }

   
  
  float error = 0;
  for( float err : error_cl) error += err;

  return error/train_x.size();



}





//classifier



void par_train_classifier(Params_CL p){
  float accuracy = 0.0;
  float error = 0.0;
  //cout<<"begin classifier thread "<<p.thread_id<<" state size "<<p.train_x.size()<<endl;
  for(int id_obs= p.mlp->position_th[p.thread_id]; id_obs<p.limit; id_obs+= p.nb_thread){
    //cout<<"thread "<<p.thread_id<<"id obs "<<id_obs<<" len train "<<p.train_x.size()<<endl;
    p.mlp->position_th[p.thread_id] = id_obs;
    if(p.mlp->computeNetwork(p.train_x[id_obs],p.thread_id, p.train_y[id_obs], false)) accuracy +=1;
    float target = p.train_y[id_obs];

    for (int iNeuron = 0; iNeuron < p.mlp->layers.back()->getOutputN(); iNeuron++) {
              
      if(target == iNeuron){
        error += (-log(p.mlp->layers.back()->getNeuronVal(iNeuron, p.thread_id)+ 0.0000005));
        //cout<<"error data "<<id_obs<<" thread "<<p.thread_id<<" error "<<error<<endl;
      }  
      p.mlp->layers.back()->addGradLastLayerUseCrossEntropy(*p.mlp->layers[p.mlp->layers.size()-2], iNeuron, target, p.thread_id);
    }

    for(int iLayer=p.mlp->layers.size()-2; iLayer>0; iLayer--){
      for(int iNeuron = 0; iNeuron < p.mlp->layers[iLayer]->getOutputN(); iNeuron++){
        p.mlp->layers[iLayer]->addGradHiddenLayer(*p.mlp->layers[iLayer-1], *p.mlp->layers[iLayer+1], iNeuron, p.thread_id);
      }
    }
    
    for(int iNeuron = 0; iNeuron< p.mlp->layers[0]->getOutputN(); iNeuron++){
        p.mlp->layers[0]->addGradFirstLayer(p.train_x[id_obs], *p.mlp->layers[1], iNeuron, p.thread_id);
    }

  }
  
  p.mlp->error_cl[p.thread_id] = error;

}



void MLP::Exp_synch_classifier(vector<vector<float>> &train_x, vector<float> &train_y,
  int nb_threads, int id_th){

  float accuracy = 0.0;
  float error = 0.0;
  int batch_size = 10;
  int batch_per_thread = batch_size / nb_threads;
  int End = batch_per_thread + id_th +1;
  //cout<<"begin classifier thread "<<p.thread_id<<" state size "<<p.train_x.size()<<endl;
  for(int id_obs= id_th; id_obs<End; id_obs+= nb_threads){
    //cout<<"thread "<<p.thread_id<<"id obs "<<id_obs<<" len train "<<p.train_x.size()<<endl;
    this->position_th[id_th] = id_obs;
    if(this->computeNetwork(train_x[id_obs],id_th, train_y[id_obs], false)) accuracy +=1;
    float target = train_y[id_obs];

    for (int iNeuron = 0; iNeuron < this->layers.back()->getOutputN(); iNeuron++) {
              
      if(target == iNeuron){
        error += (-log(this->layers.back()->getNeuronVal(iNeuron, id_th)+ 0.0000005));
        //cout<<"error data "<<id_obs<<" thread "<<thread_id<<" error "<<error<<endl;
      }  
      this->layers.back()->addGradLastLayerUseCrossEntropy(*this->layers[this->layers.size()-2], iNeuron, target, id_th);
    }

    for(int iLayer=this->layers.size()-2; iLayer>0; iLayer--){
      for(int iNeuron = 0; iNeuron < this->layers[iLayer]->getOutputN(); iNeuron++){
        this->layers[iLayer]->addGradHiddenLayer(*this->layers[iLayer-1], *this->layers[iLayer+1], iNeuron, id_th);
      }
    }
    
    for(int iNeuron = 0; iNeuron< this->layers[0]->getOutputN(); iNeuron++){
        this->layers[0]->addGradFirstLayer(train_x[id_obs], *this->layers[1], iNeuron, id_th);
    }

  }
  
  this->error_cl[id_th] = error;

  }
  


float MLP::par_trainClassifier(vector<vector<float>> &train_x, vector<float> &train_y, bool usePos, int nb_thread){

    Params_CL p;
    p.train_x = train_x;
    p.train_y = train_y;
    p.nb_thread = nb_thread;
    p.mlp = this;
    id_up = 0;
    int batch_size = 10;
    float error = 0;
    if(!usePos){
      this->position_th.resize(nb_thread);
      for(int i =0; i<nb_thread; i++) this->position_th[i] = i;

    }
    else{
      for (int iLayer = layers.size() - 1; iLayer >= 0; iLayer--) {
        layers[iLayer]->rescaleWeights(momentum, learningRate, batch_size);
        layers[iLayer]->resetNeuronDelta();
      }
    }

    
    vector<thread> threads;
    threads.resize(nb_thread);
    error_cl.resize(nb_thread);
    
    for(int data_used =0; data_used<train_x.size(); data_used+=batch_size){
      if(data_used + batch_size > train_x.size()){
        p.limit = train_x.size();    
      }
      else{
        p.limit = data_used + batch_size;
      }
      
      for( int t =0; t < nb_thread; t++){
        p.thread_id = t;
        threads[t] = thread(par_train_classifier, p);
      }

      for(int t=0; t<nb_thread; t++) {
        threads[t].join();
      }

      
      for (int iLayer = layers.size() - 1; iLayer >= 0; iLayer--) {
        layers[iLayer]->rescaleWeights(momentum, learningRate, batch_size);
        layers[iLayer]->resetNeuronDelta();
      }
      
      for( float err : error_cl) error += err;

    }
    
  return error/train_x.size();

}


float MLP::trainClassifier(vector<vector<float>> &train_x, vector<float> &train_y, int id_th){

  float accuracy = 0.0;
  float error = 0.0;
  int batch_size = 10;
  //int batch_size = train_x.size();
  int cpt =0;
  for(int id_obs=0; id_obs<train_x.size(); id_obs++){
    //cout<<"id_obs "<<id_obs<<" train_x_size "<<train_x.size()<<" target size "<<train_y.size()<<endl;
    if(computeNetwork(train_x[id_obs],id_th, train_y[id_obs], false)) accuracy +=1;
    float target = train_y[id_obs];

    for (int iNeuron = 0; iNeuron < layers.back()->getOutputN(); iNeuron++) {
              
      if(target == iNeuron){
        error += (-log(layers.back()->getNeuronVal(iNeuron, id_th)+ 0.0000005));
      }  
      layers.back()->addGradLastLayerUseCrossEntropy(*layers[layers.size()-2], iNeuron, target, id_th);
    }

    for(int iLayer=layers.size()-2; iLayer>0; iLayer--){
      for(int iNeuron = 0; iNeuron < layers[iLayer]->getOutputN(); iNeuron++){
        layers[iLayer]->addGradHiddenLayer(*layers[iLayer-1], *layers[iLayer+1], iNeuron, id_th);
      }
    }
    
    for(int iNeuron = 0; iNeuron< layers[0]->getOutputN(); iNeuron++){
        layers[0]->addGradFirstLayer(train_x[id_obs], *layers[1], iNeuron, id_th);
    }
    cpt +=1;
    if(cpt % batch_size ==0){
       for (int iLayer = layers.size() - 1; iLayer >= 0; iLayer--) {

        layers[iLayer]->rescaleWeights(momentum, learningRate, train_x.size());
        layers[iLayer]->resetNeuronDelta();
    }
  
    }

   
  }

 
   
  return error/train_x.size();

}






void MLP::updatePrevParams(){
  for (int iLayer = layers.size() - 1; iLayer >= 0; iLayer--) {

  layers[iLayer]->updatePrevParams();

  }  
}


pair<int, float> MLP::getMostProbableAndEntropy(vector<float> x, int id_th){
  computeNetwork(x, id_th);
  return layers.back()->getMostProbableAndEntropy(id_th);
}

int MLP::predict(vector<float> x, int id_th, bool choice ){
  computeNetwork(x, id_th);
  if(choice == false){
    return layers.back()->getMostProbable(id_th);
  
  }
  else{
    random_device rd;
    mt19937 gen;

    vector<float> probs = layers.back()->getOutputsNeurons(id_th);
    
    discrete_distribution<int> dist(probs.begin(), probs.end());

    return dist(gen);
  }
}


bool MLP::computeNetwork(vector<float> &features, int id_th, float label, bool usePrevParams) {

  layers[0]->fillInput(features, id_th, usePrevParams);

  for (int iLayer = 1; iLayer < layers.size(); iLayer++) {
    layers[iLayer]->calculateLayer(*layers[iLayer - 1], id_th, usePrevParams);
  }

  if (label == -1.0)
    return true;

  int predicted = layers.back()->getMostProbable(id_th);

  if (int(label) == predicted) {
    return true;
  } else
    return false;
}




void MLP::validateNetwork(vector<vector<float>> &test_x,
                          vector<float> &test_y, int id_th) {
  int total = test_y.size();
  int passed = 0;

  int barWidth = 42;
  string eq;
  for (int i = 0; i < barWidth; ++i)
    eq += "=";

  Msg(eq);
  Msg("Start validating network on", test_x.size(), "samples.");
  Msg(eq);

  auto start_testing = std::chrono::high_resolution_clock::now();

  for (int id_obs = 0; id_obs < test_x.size(); id_obs++) {

    cout << id_obs << " / " << test_x.size() << " [ ";
    float progress = float(id_obs) / float(test_x.size());
    int pos = barWidth * progress;

    if (computeNetwork(test_x[id_obs], id_th, test_y[id_obs])) {
      passed++;
    }

    for (int i = 0; i < barWidth; ++i) {
      if (i < pos)
        cout << "=";
      else if (i == pos)
        cout << ">";
      else
        cout << "_";
    }

    cout << " ] " << int(progress * 100.0) << " %\r";
    cout.flush();
  }

  auto stop_testing = std::chrono::high_resolution_clock::now();
  auto elapsed_testing = std::chrono::duration_cast<std::chrono::seconds>(
                             stop_testing - start_testing)
                             .count();

  Msg(test_x.size(), "/", test_x.size(), "[", eq, "] 100 %");

  Msg("Validation result: ", passed, "/", total, "passed (",
      float(passed) / float(total) * 100.0, "% accuracy),", elapsed_testing,
      "s elapsed");
}