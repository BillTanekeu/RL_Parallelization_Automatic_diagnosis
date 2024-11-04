#include "agent.hpp"
#include <mutex>
 
void agent::reset(int nb_threads){
    
    this->mlp->reset(nb_threads);
    this->classifier->reset(nb_threads);
    this->states.clear();
    this->rewards.clear();
    this->next_states.clear();
    this->dones.clear();
    this->targets.clear();
    this->actions.clear();
    this->seuils.clear();
    this->avg_per_Epoch.clear();
    this->prec_per_Epoch.clear();
    this->end_iEpoch.clear();

}



void agent::init_networks(int nb_action, int nb_disease, bool seq, int nb_thread){
    
    //this->classifier->loadNetwork(fileClassifier);
    cout<<"==== Initialize Classifier Network ===="<<endl;
    
    this->classifier->dims.first = 1;
    this->classifier->dims.second = nb_action;
    this->classifier->addLayer<Dense>(30, Activation::relu, nb_thread);
    this->classifier->addLayer<Dense>(20, Activation::relu, nb_thread);
    this->classifier->addLayer<Dense>(nb_disease, Activation::softmax, nb_thread);
    this->classifier->compile(l_rate, momemtum, nb_thread);

    cout<<"==== Initialize policy Network ===="<<endl;
    this->mlp->dims.second = nb_action;
    this->mlp->dims.first = 1;    
    this->mlp->addLayer<Dense>(110, Activation::relu, nb_thread);
    this->mlp->addLayer<Dense>(110, Activation::relu, nb_thread);
    this->mlp->addLayer<Dense>(nb_action, Activation::identity, nb_thread);
    this->mlp->compile(l_rate, momemtum, nb_thread);
    cout<<"==== Initialize threads networks ("<< nb_thread<<") ===="<<endl;
    
}

void agent::init_env(int nb_thread, string filedata, int size_UD){
    //cout<<"init env nEpoch "<<this->N_epoch<<endl;

    this->nb_threads = nb_thread;
    this->glob_env.init_environment(filedata);
    to_move.resize(nb_thread);
    position.resize(nb_thread);
    this->limit_UD = size_UD;
    this->real_len_UD = 0;
    this->states.resize(size_UD);
    this->rewards.resize(size_UD);
    this->next_states.resize(size_UD);
    this->dones.resize(size_UD);
    this->targets.resize(size_UD);
    this->actions.resize(size_UD);
    
    this->avg_per_Epoch.resize(nb_thread);
    this->prec_per_Epoch.resize(nb_thread);
    this->test_avg_per_Epoch.resize(nb_thread);
    this->test_prec_per_Epoch.resize(nb_thread);

    for(int i =0; i< nb_thread; i++){
        this->position[i] = i;
        this->to_move[i] = i;
    } 
    
    vector<float> temp (nb_thread, 0);
    capture.assign(temp.begin(), temp.end());
    precision.assign(temp.begin(), temp.end());
    average_turn.assign(temp.begin(), temp.end());
    
    end_iEpoch.resize(this->N_epoch);
    for(int i =0; i <end_iEpoch.size(); i++){
        end_iEpoch[i] = 0;
    }  
    for( int i =0; i<nb_thread; i++){
        this->envs.push_back(env(glob_env));
    }
    seuils.resize(this->glob_env.nb_diseases);
    for(float &val : seuils) val = 0.5;
    
    for(int i = 0; i< this->glob_env.dims.first; i++){
        this->vec_gen.push_back(mt19937(i));
    }
}

void agent::reset_metrics(int nb_thread){
    capture.clear();
    precision.clear();
    average_turn.clear();
    position.clear();
    endEnv_threads = 0;
    stop_env_threads = false;
    Error_P = 0.0;
    nb_modif = 0;
    UpdateP = false;
    bool_convg = false;
    //end_iEpoch = 0;
    vector<float> temp (nb_thread, 0);
    
    for(int i =0; i< nb_thread; i++){
        position[i] = i;
    } 
    
    capture.assign(temp.begin(), temp.end());
    precision.assign(temp.begin(), temp.end());
    average_turn.assign(temp.begin(), temp.end());
    
    
}



float agent::update_params(vector<vector<float>> &states,vector<vector<float>> &nextstates,
                         vector<float> &actions, vector<bool> &dones, 
                                vector<float> &rewards, vector<float> &targets, bool parallel, int nb_th, bool implicite ){
                                    
    float error, pre;
    
    /*
     for(int i : batch_states[0]){
        cout<<" "<<i;
    }
    cout<<endl;
    exit(0);
    */
    if(parallel){
        if(implicite){
            mlp->Implicite_Agg_update_RL(states, nextstates,actions, dones,rewards, this->Lamda, nb_th);
            //error = classifier->trainClassifier(states, targets, 0);
            error = classifier->implicite_parTrainClassifier(states, targets, nb_th);
        }
        else{
            mlp->par_update_params_rl(states, nextstates,actions, dones,rewards, this->Lamda, nb_th);
            //error = classifier->par_trainClassifier(states, targets, false, nb_th);
            error = classifier->trainClassifier(states, targets, 0);
        }
        
    }
    else{
        mlp->update_params_rl(states, nextstates,actions, dones,rewards, this->Lamda, 0);
        error = classifier->trainClassifier(states, targets, 0);
    }
    
    
    
    return error;
}


float agent::Exp_update_params(vector<vector<float>> &states,vector<vector<float>> &nextstates,
                         vector<float> &actions, vector<bool> &dones, 
                                vector<float> &rewards, vector<float> &targets,bool parallel, int nb_th){
    float error, pre;


    mlp->par_update_params_rl(states, nextstates,actions, dones,rewards, this->Lamda, nb_th);
    //error = classifier->par_trainClassifier(states, targets, true, nb_th);
    error = classifier->trainClassifier(states, targets, 0);

    return error;

}


void agent::Exp_synch_update( int nb_th, int th_id ){
    
    bool last = false;
    
    if(this->endEnv_threads == this->env_nbThreads){
        this->endEnv_threads =0;
        
        
    }
    

    
    mlp->synch_update_params_rl(this->batch_states, this->batch_nextstates, this->batch_actions,
                this->batch_dones, this->batch_rewards, this->Lamda, nb_th, th_id);
    
    
    classifier->Exp_synch_classifier(this->batch_states,
                            this->batch_targets, nb_th, th_id);
}


void agent::Imp_update( int nb_th, int th_id ){
    
    
    
    batch b;
    b.size = this->batch_trainning/nb_th;
    
    this->create_batch_V2(b);
    


    //cout<<"deb update thread "<<th_id<<" batch_size "<<b.batch_states.size()<<endl;
    this-> policy_error_test = mlp->Imp_update_params_rl(b.batch_states, b.batch_nextstates, b.batch_actions,
                b.batch_dones, b.batch_rewards, this->Lamda, nb_th, th_id, this->mutex_Mat);
    
    //cout<<"end thread "<<th_id<<" nbthread "<<nb_th<<endl;
    
    this->synch_loss = classifier->Imp_parTrainClassifier(b.batch_states,
                            b.batch_targets, nb_th, th_id, this->mutex_Mat);
    
    //cout<<"end2"<<endl;
    //cout<<"Fin update thread "<<th_id<<" batch_size "<<this->batch_states.size()<<" End env thread "<<endEnv_threads<<endl;

   
    
}


void agent::update_prevParams(bool seq){
    mlp->updatePrevParams();

    /*if(!seq){
        for(int i =0; i<this->nb_threads; i++){
            this->mlp_models[i]->updatePrevParams();
        }
    }*/
    
}

void agent::save_agent(string filename_classifier, string filename_policy){
    this->mlp->saveNetwork(filename_policy);
    this->classifier->saveNetwork(filename_classifier);
}

void agent::load_agent(string filename_classifier, string filename_policy){
    this->mlp->loadNetwork(filename_policy);
    this->classifier->loadNetwork(filename_classifier);
}

pair<int, float> agent::choose_diagnostic(vector<float> state, int id_thread, bool origin){
    if(origin){
       return this->classifier->getMostProbableAndEntropy(state, id_thread); 
    }
    return this->classifier_models[id_thread]->getMostProbableAndEntropy(state, id_thread);
}


int agent::choose_action(vector<float> &state, float eps, bool eGreedy, int id_thread, bool origin, int id_obs){
     if(eGreedy){
        random_device rd;
        //mt19937 gen(1);
        uniform_int_distribution<int> dist_action(0, this->glob_env.dims.second-1);
        uniform_real_distribution<float> dist_eps(0,1);

        if(eps > dist_eps(this->vec_gen[id_obs])){
            return dist_action(this->vec_gen[id_obs]);
        }
        if(origin){
            return this->mlp->predict(state, id_thread, false);    
        }
        return this->mlp_models[id_thread]->predict(state, id_thread, false);
    }
    else{
        if(origin){
            return this->mlp->predict(state, id_thread, true);    
        }
        return this->mlp_models[id_thread]->predict(state, id_thread, true);
    }
}

pair<float, float> agent::test(){

    int deb = this->glob_env.end_train + 1;
    int end = this->glob_env.dims.first;
    int test_size = (end -deb) +1;
    int max_turn = this->glob_env.MAX_TURN;
    float acc = 0.0;
    float acc_turn = 0.0;
    pair<float, float> res;

    for (int id_obs = deb ; id_obs < end; id_obs++) {//begin batch
        
                vector<float> current_state(this->glob_env.dims.second,0);
                vector<float> t_symp(this->glob_env.dims.second,0);
                vector<int> all_symp;
                int action;
                int t_disease, dt;
                pair<int, float> diag;
                envs[0].initial_state(id_obs, current_state, t_symp, t_disease, all_symp);
                diag = choose_diagnostic(current_state, 0, true);
                envs[0].H_s0 = diag.second;
                //cout<<endl<<"AP init state"<<endl;
                for(int turn = 0; turn < max_turn; turn++){
                    float reward;
                    
                    float H_st = diag.second;
                    
                    dt = diag.first;
                    //this->states[to_move[0]].assign(current_state.begin(), current_state.end());
                    action = choose_action(current_state, 0, true, 0, true, id_obs);
                    envs[0].step(current_state, t_symp, all_symp, reward, action, turn, t_disease, diag);
                    diag = choose_diagnostic(current_state, 0 , true);
            
                    float rh = (H_st -  diag.second )/envs[0].H_s0;
                   
                    if( rh < 0) rh = 0;
                    reward = reward + (2.5*rh);
                
                    if( (H_st < seuils[dt]) or (turn == max_turn -1) ){
                    
                        if(dt == t_disease and turn != (max_turn-1)){
                            acc += 1;
                            if( (seuils[t_disease] - H_st) < 0.2){
                                float l = 0.3;
                                seuils[t_disease] = l*seuils[t_disease] + (1-l)*H_st;
                            }
                        } 
                        acc_turn += turn+1; 
                        break;
                    }
                    
                } //end Dialog
    }

    res.first = acc/test_size;
    res.second = acc_turn/test_size;

    return res;
     
}


void agent::test_par(int id_th){

    int Nb_TH = this->env_nbThreads;

    int deb = this->glob_env.end_train + 1;
    int end = this->glob_env.dims.first;
    int test_size = (end -deb) +1;
    int max_turn = this->glob_env.MAX_TURN;
    float acc = 0.0;
    float acc_turn = 0.0;
    pair<float, float> res;

    int id_t;

    if(this->env_nbThreads == 1){
        id_t = this->nb_threads -this->env_nbThreads;
    }else{
        id_t = id_th + (this->nb_threads -this->env_nbThreads);
    }

    for (int id_obs = deb + id_th ; id_obs < end; id_obs += Nb_TH) {//begin batch
        
        vector<float> current_state(this->glob_env.dims.second,0);
        vector<float> t_symp(this->glob_env.dims.second,0);
        vector<int> all_symp;
        int action;
        int t_disease, dt;
        pair<int, float> diag;
        envs[id_th].initial_state(id_obs, current_state, t_symp, t_disease, all_symp);
        diag = choose_diagnostic(current_state, id_t, true);
        envs[id_th].H_s0 = diag.second;
        //cout<<endl<<"AP init state"<<endl;
        for(int turn = 0; turn < max_turn; turn++){
            float reward;
            
            float H_st = diag.second;
            
            dt = diag.first;
            //this->states[to_move[0]].assign(current_state.begin(), current_state.end());
            action = choose_action(current_state, 0, true, id_t, true, id_obs);
            envs[id_th].step(current_state, t_symp, all_symp, reward, action, turn, t_disease, diag);
            diag = choose_diagnostic(current_state, id_t , true);
    
            float rh = (H_st -  diag.second )/envs[id_th].H_s0;
            
            if( rh < 0) rh = 0;
            reward = reward + (2.5*rh);
        
            if( (H_st < seuils[dt]) or (turn == max_turn -1) ){
            
                if(dt == t_disease and turn != (max_turn-1)){
                    acc += 1;
                    if( (seuils[t_disease] - H_st) < 0.2){
                        float l = 0.3;
                        seuils[t_disease] = l*seuils[t_disease] + (1-l)*H_st;
                    }
                } 
                acc_turn += turn+1; 
                break;
            }
            
        } //end Dialog
    }

    this->test_avg_per_Epoch[id_th].push_back(acc_turn);
    this->test_prec_per_Epoch[id_th].push_back(acc);

     
}

void  dialog(params p){

    int train_size = p.ag->glob_env.end_train;
    int barWidth = 23;
    int max_turn = p.ag->glob_env.MAX_TURN;
    for (int id_obs = p.ag->position[p.num_thread]  ; id_obs < p.ag->limit; id_obs += p.ag->env_nbThreads) {//begin batch
        //cout<<endl<<"thread "<<p.num_thread<<" started"<<" id_obs "<<id_obs<<" env _thread "<<p.ag->env_nbThreads<<endl;
        //cout<<"thread "<<p.num_thread<<" limit "<<p.ag->limit<<" env thread "<<p.ag->env_nbThreads<<" id "<<id_obs<<endl;
        //if(id_obs % 200 ==0)  p.ag->update_prevParams();
        //if(iEpoch % 2 ==0)  update_prevParams();
        //float progress = 100 * float(id_obs) / float(train_size);
        vector<float> current_state(p.ag->glob_env.dims.second,0);
        vector<float> t_symp(p.ag->glob_env.dims.second,0);
        vector<int> all_symp;
        int action;
        int t_disease, dt;
        pair<int, float> diag;

        p.ag->envs[p.num_thread].initial_state(id_obs, current_state, t_symp, t_disease, all_symp);
        diag = p.ag->choose_diagnostic(current_state, p.num_thread, true);
        p.ag->envs[p.num_thread].H_s0 = diag.second;
        //cout<<endl<<"av turn thread "<<p.num_thread<<" id obs "<<id_obs<<endl;
       
        for(int turn = 0; turn < max_turn; turn++){
            /*
            if(p.ag->to_move[p.num_thread] >= p.ag->limit_UD){
                cout<<endl<<"thread "<<p.num_thread<<" nb_threas "<<p.ag->env_nbThreads<<" move "<<p.ag->to_move[p.num_thread]<<
                " limit ud "<< p.ag->limit_UD<<" real ud "<<p.ag->real_len_UD<<endl;
                exit(0);
            }
            */


            float reward;
            float H_st = diag.second;
            
            dt = diag.first;
            p.ag->states[p.ag->to_move[p.num_thread]].assign(current_state.begin(), current_state.end());
            action = p.ag->choose_action(current_state, p.ag->eps, true, p.num_thread, true, id_obs);
            p.ag->envs[p.num_thread].step(current_state, t_symp, all_symp, reward, action, turn, t_disease, diag);
            diag = p.ag->choose_diagnostic(current_state, p.num_thread, true);
    
            float rh = (H_st -  diag.second )/p.ag->envs[p.num_thread].H_s0;
            

            if( rh < 0) rh = 0;
            reward = reward + (2.5*rh);

            

            p.ag->rewards[p.ag->to_move[p.num_thread]] = reward;
            p.ag->next_states[p.ag->to_move[p.num_thread]].assign(current_state.begin(), current_state.end());
            p.ag->actions[p.ag->to_move[p.num_thread]] = action;
            p.ag->targets[p.ag->to_move[p.num_thread]] = t_disease;
                p.ag->dones[p.ag->to_move[p.num_thread]] = false;
            int t_mv = p.ag->to_move[p.num_thread];

            p.ag->real_len_UD = min(p.ag->limit_UD, p.ag->real_len_UD +1);
            

            if((p.ag->to_move[p.num_thread] + p.ag->env_nbThreads) >= p.ag->limit_UD){
                p.ag->to_move[p.num_thread] = p.num_thread;

            }else{
                p.ag->to_move[p.num_thread] += p.ag->env_nbThreads;
            }
            

            if( (H_st < p.ag->seuils[dt]) or (turn == max_turn -1) ){
                //cout<<"av done true t_mv: "<< t_mv<<" p.ag.tomove numthread "<<p.ag->to_move[p.num_thread]<<" thread_id "<<p.num_thread<<endl;
                p.ag->dones[t_mv] = true;
                //cout<<"ap done true "<<endl;
                if(dt == t_disease and turn != (max_turn-1)){
                    p.ag->precision[p.num_thread] += 1;
                   // if(p.ag->precision[p.num_thread] == train_size) cout<<endl<<"id obs "<<id_obs<<endl;
                    if( (p.ag->seuils[t_disease] - H_st) < 0.2){
                        float l = 0.3;
                        p.ag->seuils[t_disease] = l*p.ag->seuils[t_disease] + (1-l)*H_st;
                    }
                } 
                p.ag->average_turn[p.num_thread] += turn+1; 
                break;
            }
            
            //cout<<endl<<"Reward "<<reward<<" rh "<<rh<<endl;
           
            
            
        } //end Dialog

        p.ag->capture[p.num_thread] += p.ag->envs[p.num_thread].count_sympt/all_symp.size();
        
        
        if((id_obs + p.ag->env_nbThreads) >= train_size){
            p.ag->position[p.num_thread] = p.num_thread;
        }else{
            p.ag->position[p.num_thread] = id_obs + p.ag->env_nbThreads;
        }
        
        
    }//end Batch
            
}


void dialog_v2(params p){
   
   int train_size = p.ag->glob_env.end_train;
    int max_turn = p.ag->glob_env.MAX_TURN;
    float epsilon_min = 0.0;
    float end_epsilon = p.ag->N_epoch/10;
    float epsilon_decay_value = p.ag->eps / end_epsilon;
    
    int id_th;
    if(p.ag->env_nbThreads == 1 ){
        id_th = p.ag->nb_threads -p.ag->env_nbThreads;

    }else{
        id_th = p.num_thread + (p.ag->nb_threads - p.ag->env_nbThreads);
    }

    //cout<<endl<<"thread "<<id_th<<" started"<<endl;
    for(int iEpoch = 0; iEpoch < p.ag->N_epoch; iEpoch ++ ){
        if(p.ag->stop_env_threads){
            p.ag->mtx.lock();
            p.ag->endEnv_threads +=1;
            //int endE = p.ag->endEnv_threads;
            p.ag->mtx.unlock();
            //cout<<"stop thread "<<p.num_thread<<" endE "<<endE<<endl;
            return;
        }
        float acc = 0;
        float AT = 0;
        for (int id_obs = p.num_thread  ; id_obs < train_size; id_obs += p.ag->env_nbThreads) {//begin batch
            //cout<<endl<<"thread "<<p.num_thread<<" started"<<" id_obs "<<id_obs<<" env _thread "<<p.ag->env_nbThreads<<endl;
                    
            vector<float> current_state(p.ag->glob_env.dims.second,0);
            vector<float> t_symp(p.ag->glob_env.dims.second,0);
            vector<int> all_symp;
            int action;
            int t_disease, dt;
            pair<int, float> diag;

            p.ag->envs[p.num_thread].initial_state(id_obs, current_state, t_symp, t_disease, all_symp);
            diag = p.ag->choose_diagnostic(current_state, id_th, true);
            p.ag->envs[p.num_thread].H_s0 = diag.second;
        

            for(int turn = 0; turn < max_turn; turn++){            
                float reward;
                float H_st = diag.second;
                
                dt = diag.first;
                p.ag->states[p.ag->to_move[p.num_thread]].assign(current_state.begin(), current_state.end());
                action = p.ag->choose_action(current_state, p.ag->eps, true, id_th, true, id_obs);
                p.ag->envs[p.num_thread].step(current_state, t_symp, all_symp, reward, action, turn, t_disease, diag);
                diag = p.ag->choose_diagnostic(current_state, id_th, true);
        
                float rh = (H_st -  diag.second )/p.ag->envs[p.num_thread].H_s0;
                

                if( rh < 0) rh = 0;
                reward = reward + (2.5*rh);


                p.ag->rewards[p.ag->to_move[p.num_thread]] = reward;
                p.ag->next_states[p.ag->to_move[p.num_thread]].assign(current_state.begin(), current_state.end());
                p.ag->actions[p.ag->to_move[p.num_thread]] = action;
                p.ag->targets[p.ag->to_move[p.num_thread]] = t_disease;
                p.ag->dones[p.ag->to_move[p.num_thread]] = false;
                int t_mv = p.ag->to_move[p.num_thread];

                p.ag->mtx.lock();
                p.ag->real_len_UD = min(p.ag->limit_UD, p.ag->real_len_UD +1);
                p.ag->mtx.unlock();

                if((p.ag->to_move[p.num_thread] + p.ag->env_nbThreads) >= p.ag->limit_UD){
                    p.ag->to_move[p.num_thread] = p.num_thread;

                }else{
                    p.ag->to_move[p.num_thread] += p.ag->env_nbThreads;
                }

                if( (H_st < p.ag->seuils[dt]) or (turn == max_turn -1) ){
                    p.ag->dones[t_mv] = true;

                    if(dt == t_disease and turn != (max_turn-1)){
                        acc += 1;
                    // if(p.ag->precision[p.num_thread] == train_size) cout<<endl<<"id obs "<<id_obs<<endl;
                        if( (p.ag->seuils[t_disease] - H_st) < 0.2){
                            float l = 0.3;
                            p.ag->seuils[t_disease] = l*p.ag->seuils[t_disease] + (1-l)*H_st;
                        }
                    } 
                    AT += turn+1; 
                    break;
                }
                
                
            } //end Dialog

            //p.ag->capture[p.num_thread] += p.ag->envs[p.num_thread].count_sympt/all_symp.size();
            
            if((id_obs + p.ag->env_nbThreads) >= train_size){
                p.ag->position[p.num_thread] = p.num_thread;
            }else{
                p.ag->position[p.num_thread] = id_obs + p.ag->env_nbThreads;
            }
            //cout<<"thread "<<p.num_thread<<" ap move"<<endl;
            
        }//end Epoch

        p.ag->test_par(p.num_thread);
        //if( p.ag->eps > epsilon_min)  p.ag->eps -= epsilon_decay_value;
        //p.ag->update_prevParams(true);

        p.ag->prec_per_Epoch[p.num_thread].push_back(acc);
        p.ag->avg_per_Epoch[p.num_thread].push_back(AT);
        p.ag->mtx.lock();
        p.ag->end_iEpoch[iEpoch] +=1;
        p.ag->mtx.unlock();
       
    }

    p.ag->mtx.lock();
    p.ag->endEnv_threads +=1;
    p.ag->mtx.unlock();

   
}


void dialog_Epoch(params p){

    int train_size = p.ag->glob_env.end_train;
    int max_turn = p.ag->glob_env.MAX_TURN;
    int cpt = 0;
    for (int id_obs = p.num_thread  ; id_obs < p.ag->glob_env.end_train; id_obs += p.ag->env_nbThreads) {//begin batch
   
        vector<float> current_state(p.ag->glob_env.dims.second,0);
        vector<float> t_symp(p.ag->glob_env.dims.second,0);
        vector<int> all_symp;
        int action;
        int t_disease, dt;
        pair<int, float> diag;

        p.ag->envs[p.num_thread].initial_state(id_obs, current_state, t_symp, t_disease, all_symp);
        diag = p.ag->choose_diagnostic(current_state, p.num_thread + p.ag->env_nbThreads, true);
        p.ag->envs[p.num_thread].H_s0 = diag.second;

        //cout<<endl<<"av turn thread "<<p.num_thread<<" id obs "<<id_obs<<" len "<<p.ag->real_len_UD<<endl;
       
        for(int turn = 0; turn < max_turn; turn++){
           
            float reward;
            float H_st = diag.second;
            
            dt = diag.first;
            p.ag->states[p.ag->to_move[p.num_thread]].assign(current_state.begin(), current_state.end());
            action = p.ag->choose_action(current_state, p.ag->eps, true, p.num_thread+ p.ag->env_nbThreads, true, id_obs);
            p.ag->envs[p.num_thread].step(current_state, t_symp, all_symp, reward, action, turn, t_disease, diag);
            diag = p.ag->choose_diagnostic(current_state, p.num_thread+ p.ag->env_nbThreads, true);
    
            float rh = (H_st -  diag.second )/p.ag->envs[p.num_thread].H_s0;

            if( rh < 0) rh = 0;
            reward = reward + (2.5*rh);

            p.ag->rewards[p.ag->to_move[p.num_thread]] = reward;
            p.ag->next_states[p.ag->to_move[p.num_thread]].assign(current_state.begin(), current_state.end());
            p.ag->actions[p.ag->to_move[p.num_thread]] = action;
            p.ag->targets[p.ag->to_move[p.num_thread]] = t_disease;
            p.ag->dones[p.ag->to_move[p.num_thread]] = false;
            int t_mv = p.ag->to_move[p.num_thread];

            p.ag->mtx.lock();
            p.ag->real_len_UD = min(p.ag->limit_UD, p.ag->real_len_UD +1);
            p.ag->mtx.unlock();

            if((p.ag->to_move[p.num_thread] + p.ag->env_nbThreads) >= p.ag->limit_UD){
                p.ag->to_move[p.num_thread] = p.num_thread;

            }else{
                p.ag->to_move[p.num_thread] += p.ag->env_nbThreads;
            }

            if( (H_st < p.ag->seuils[dt]) or (turn == max_turn -1) ){
                p.ag->dones[t_mv] = true;

                if(dt == t_disease and turn != (max_turn-1)){
                    p.ag->precision[p.num_thread] += 1;
                   // if(p.ag->precision[p.num_thread] == train_size) cout<<endl<<"id obs "<<id_obs<<endl;
                    if( (p.ag->seuils[t_disease] - H_st) < 0.2){
                        float l = 0.3;
                        p.ag->seuils[t_disease] = l*p.ag->seuils[t_disease] + (1-l)*H_st;
                    }
                } 
                p.ag->average_turn[p.num_thread] += turn+1; 
                break;
            }
            
            
        } //end Dialog

        p.ag->capture[p.num_thread] += p.ag->envs[p.num_thread].count_sympt/all_symp.size();
        
        
        if((id_obs + p.ag->env_nbThreads) >= train_size){
            p.ag->position[p.num_thread] = p.num_thread;
        }else{
            p.ag->position[p.num_thread] = id_obs + p.ag->env_nbThreads;
        }
        cpt++;
        /*
        if(((cpt +1) % 10) == 0){
            if(p.ag->UpdateP == false){
                p.ag->mtx.lock();
                p.ag->UpdateP = true;
                p.ag->mtx.unlock();                
            }
        }*/
    
    }//end Batch
    p.ag->mtx.lock();
    p.ag->endEnv_threads +=1;
    p.ag->mtx.unlock();
    
}



void  Exp_synch_dialog(params p){

    int train_size = p.ag->glob_env.end_train;
    int barWidth = 23;
    int max_turn = p.ag->glob_env.MAX_TURN;
    for (int id_obs = p.ag->position[p.num_thread]  ; id_obs < p.ag->limit; id_obs += p.ag->env_nbThreads) {//begin batch
        //cout<<endl<<"thread "<<p.num_thread<<" started"<<" id_obs "<<id_obs<<" env _thread "<<p.ag->env_nbThreads<<endl;
        //cout<<"thread "<<p.num_thread<<" limit "<<p.ag->limit<<" env thread "<<p.ag->env_nbThreads<<" id "<<id_obs<<endl;
        //if(id_obs % 200 ==0)  p.ag->update_prevParams();
        //if(iEpoch % 2 ==0)  update_prevParams();
        //float progress = 100 * float(id_obs) / float(train_size);
        vector<float> current_state(p.ag->glob_env.dims.second,0);
        vector<float> t_symp(p.ag->glob_env.dims.second,0);
        vector<int> all_symp;
        int action;
        int t_disease, dt;
        pair<int, float> diag;

        p.ag->envs[p.num_thread].initial_state(id_obs, current_state, t_symp, t_disease, all_symp);
        diag = p.ag->choose_diagnostic(current_state, p.num_thread, true);
        p.ag->envs[p.num_thread].H_s0 = diag.second;
        //cout<<endl<<"av turn thread "<<p.num_thread<<" id obs "<<id_obs<<endl;
       
        for(int turn = 0; turn < max_turn; turn++){
            /*
            if(p.ag->to_move[p.num_thread] >= p.ag->limit_UD){
                cout<<endl<<"thread "<<p.num_thread<<" nb_threas "<<p.ag->env_nbThreads<<" move "<<p.ag->to_move[p.num_thread]<<
                " limit ud "<< p.ag->limit_UD<<" real ud "<<p.ag->real_len_UD<<endl;
                exit(0);
            }
            */


            float reward;
            float H_st = diag.second;
            
            dt = diag.first;
            p.ag->states[p.ag->to_move[p.num_thread]].assign(current_state.begin(), current_state.end());
            action = p.ag->choose_action(current_state, p.ag->eps, true, p.num_thread, true, id_obs);
            p.ag->envs[p.num_thread].step(current_state, t_symp, all_symp, reward, action, turn, t_disease, diag);
            diag = p.ag->choose_diagnostic(current_state, p.num_thread, true);
    
            float rh = (H_st -  diag.second )/p.ag->envs[p.num_thread].H_s0;
            

            if( rh < 0) rh = 0;
            reward = reward + (2.5*rh);

            

            p.ag->rewards[p.ag->to_move[p.num_thread]] = reward;
            p.ag->next_states[p.ag->to_move[p.num_thread]].assign(current_state.begin(), current_state.end());
            p.ag->actions[p.ag->to_move[p.num_thread]] = action;
            p.ag->targets[p.ag->to_move[p.num_thread]] = t_disease;
                p.ag->dones[p.ag->to_move[p.num_thread]] = false;
            int t_mv = p.ag->to_move[p.num_thread];

            p.ag->real_len_UD = min(p.ag->limit_UD, p.ag->real_len_UD +1);
            

            if((p.ag->to_move[p.num_thread] + p.ag->env_nbThreads) >= p.ag->limit_UD){
                p.ag->to_move[p.num_thread] = p.num_thread;

            }else{
                p.ag->to_move[p.num_thread] += p.ag->env_nbThreads;
            }
            

            if( (H_st < p.ag->seuils[dt]) or (turn == max_turn -1) ){
                //cout<<"av done true t_mv: "<< t_mv<<" p.ag.tomove numthread "<<p.ag->to_move[p.num_thread]<<" thread_id "<<p.num_thread<<endl;
                p.ag->dones[t_mv] = true;
                //cout<<"ap done true "<<endl;
                if(dt == t_disease and turn != (max_turn-1)){
                    p.ag->precision[p.num_thread] += 1;
                   // if(p.ag->precision[p.num_thread] == train_size) cout<<endl<<"id obs "<<id_obs<<endl;
                    if( (p.ag->seuils[t_disease] - H_st) < 0.2){
                        float l = 0.3;
                        p.ag->seuils[t_disease] = l*p.ag->seuils[t_disease] + (1-l)*H_st;
                    }
                } 
                p.ag->average_turn[p.num_thread] += turn+1; 
                break;
            }
            
            //cout<<endl<<"Reward "<<reward<<" rh "<<rh<<endl;
           
            
            
        } //end Dialog

        p.ag->capture[p.num_thread] += p.ag->envs[p.num_thread].count_sympt/all_symp.size();
        
        
        if((id_obs + p.ag->env_nbThreads) >= train_size){
            p.ag->position[p.num_thread] = p.num_thread;
        }else{
            p.ag->position[p.num_thread] = id_obs + p.ag->env_nbThreads;
        }
        
        
    }//end Batch
    
    //p.ag->begin_create_batch = false;
    
    cout<<"\r";

    
    p.ag->begin_create_batch = false;
    if(p.ag->start_create_batch){
        
        //cout<<"win thread "<<p.num_thread<<" begin create batch "<<p.ag->begin_create_batch<<" b"<<endl;
        p.ag->create_batch(p.ag->batch_trainning);
        p.ag->mtx.lock();
        p.ag->end_create_batch = true;
        p.ag->mtx.unlock();
        
        
    }
    
    
    while (true){
        cout<<"\r";
        if(p.ag->end_create_batch ){
            break;
        }
    }
    
    
    
    p.ag->mtx.lock();
    p.ag->endEnv_threads +=1;
    p.ag->mtx.unlock();
    //cout<<"Deb update thread i =  "<<p.num_thread<<endl;
    p.ag->Exp_synch_update(p.ag->env_nbThreads, p.num_thread);
    
            
}



void Imp_Epoch_dialog(params p){

    int train_size = p.ag->glob_env.end_train;
    int max_turn = p.ag->glob_env.MAX_TURN;
    int cpt = 0;
    int barWidth = 23;
    //cout<<"deb threads "<<p.num_thread<<" env threads "<<p.ag->env_nbThreads<<endl;
    //while(true){}
    for (int id_obs = p.num_thread  ; id_obs < p.ag->glob_env.end_train; id_obs += p.ag->env_nbThreads) {//begin batch
   
        vector<float> current_state(p.ag->glob_env.dims.second,0);
        vector<float> t_symp(p.ag->glob_env.dims.second,0);
        vector<int> all_symp;
        int action;
        int t_disease, dt;
        pair<int, float> diag;

        p.ag->envs[p.num_thread].initial_state(id_obs, current_state, t_symp, t_disease, all_symp);
        diag = p.ag->choose_diagnostic(current_state, p.num_thread , true);
        p.ag->envs[p.num_thread].H_s0 = diag.second;

        //cout<<endl<<"av turn thread "<<p.num_thread<<" id obs "<<id_obs<<" len "<<p.ag->real_len_UD<<endl;
       
        for(int turn = 0; turn < max_turn; turn++){
           
            float reward;
            float H_st = diag.second;
            
            dt = diag.first;
            p.ag->states[p.ag->to_move[p.num_thread]].assign(current_state.begin(), current_state.end());
            action = p.ag->choose_action(current_state, p.ag->eps, true, p.num_thread, true, id_obs);
            p.ag->envs[p.num_thread].step(current_state, t_symp, all_symp, reward, action, turn, t_disease, diag);
            diag = p.ag->choose_diagnostic(current_state, p.num_thread, true);
    
            float rh = (H_st -  diag.second )/p.ag->envs[p.num_thread].H_s0;

            if( rh < 0) rh = 0;
            reward = reward + (2.5*rh);

            p.ag->rewards[p.ag->to_move[p.num_thread]] = reward;
            p.ag->next_states[p.ag->to_move[p.num_thread]].assign(current_state.begin(), current_state.end());
            p.ag->actions[p.ag->to_move[p.num_thread]] = action;
            p.ag->targets[p.ag->to_move[p.num_thread]] = t_disease;
            p.ag->dones[p.ag->to_move[p.num_thread]] = false;
            int t_mv = p.ag->to_move[p.num_thread];

            p.ag->mtx.lock();
            p.ag->real_len_UD = min(p.ag->limit_UD, p.ag->real_len_UD +1);
            p.ag->mtx.unlock();

            if((p.ag->to_move[p.num_thread] + p.ag->env_nbThreads) >= p.ag->limit_UD){
                p.ag->to_move[p.num_thread] = p.num_thread;

            }else{
                p.ag->to_move[p.num_thread] += p.ag->env_nbThreads;
            }

            if( (H_st < p.ag->seuils[dt]) or (turn == max_turn -1) ){
                p.ag->dones[t_mv] = true;

                if(dt == t_disease and turn != (max_turn-1)){
                    p.ag->precision[p.num_thread] += 1;
                   // if(p.ag->precision[p.num_thread] == train_size) cout<<endl<<"id obs "<<id_obs<<endl;
                    if( (p.ag->seuils[t_disease] - H_st) < 0.2){
                        float l = 0.3;
                        p.ag->seuils[t_disease] = l*p.ag->seuils[t_disease] + (1-l)*H_st;
                    }
                } 
                p.ag->average_turn[p.num_thread] += turn+1; 
                break;
            }
            
            
        } //end Dialog

        p.ag->capture[p.num_thread] += p.ag->envs[p.num_thread].count_sympt/all_symp.size();
        
        
        if((id_obs + p.ag->env_nbThreads) >= train_size){
            p.ag->position[p.num_thread] = p.num_thread;
        }else{
            p.ag->position[p.num_thread] = id_obs + p.ag->env_nbThreads;
        }
        cpt++;
        
        //cout<<"thread "<<p.num_thread<<" idobs "<<id_obs<<endl;
        if(((cpt +1) % 18) == 0){
            //cout<<"Av update thread "<<p.num_thread<<" to move "<<p.ag->to_move[p.num_thread]<<endl;

            
            /*
            p.ag->begin_create_batch = false;
            if(p.ag->start_create_batch){
                p.ag->mtx.lock();
                p.ag->end_create_batch = false;
                p.ag->mtx.unlock();

                cout<<"win thread "<<p.num_thread<<" begin create batch "<<p.ag->begin_create_batch<<" b"<<endl;
                p.ag->create_batch(p.ag->batch_trainning);
                
                

            }
            */

            p.ag->Imp_update(p.ag->env_nbThreads, p.num_thread);

            
            
        
        }//end Batch

        //cout<<"Ap update thread "<<p.num_thread<<" to move "<<p.ag->to_move[p.num_thread]<<endl;

        /*
        if(p.num_thread == 0){
            float progress = 100 * float(p.ag->position[0]) / float(train_size);
            cout<<"Epoch "<<"--"<<"/"<<"--"<<": ";
            cout << p.ag->position[0] << "/" << train_size <<" [ ";
            int pos = barWidth * progress / 100;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos)
                    cout << "=";
                else if (i == pos)
                    cout << ">";
                else
                    cout << "_";
            }

            cout << " ] " << int(progress) << " %"<<"real_len UD "<<p.ag->real_len_UD;
            cout<<"\r";
        
        }
        */

        
    
    }

    p.ag->mtx.lock();
    p.ag->endEnv_threads +=1;
    p.ag->mtx.unlock();


}


void agent::par_trainParams(int Iepoch){
   
    int barWidth = 15;
    int train_size = this->glob_env.end_train;

    while (true){
        cout<<"\r";
        if( this->endEnv_threads == this->env_nbThreads){
            cout<<"\r";
            break;
        }
        /**/        
        /*
        float progress = 100 * float(position[0]) / float(train_size);
                cout<<"Epoch ..."<<Iepoch<<" ";
                cout << position[0] << "/" << train_size <<" [ ";
             int pos = barWidth * progress / 100;
                for (int i = 0; i < barWidth; ++i) {
                    if (i < pos)
                        cout << "=";
                    else if (i == pos)
                        cout << ">";
                    else
                        cout << "_";
                }

            cout << " ] " << int(progress) << " %"<<"real_len UD "<<this->real_len_UD;
            cout<<"\r";
          */  
            auto start_update = std::chrono::high_resolution_clock::now();
        //cout <<endl<<"epochError "<<epochError <<endl;
                   
        if(this->real_len_UD >0){
            //cout<<endl<<"up end " <<this->endEnv_threads<<" env "<<this->env_nbThreads<<endl;
            
            nb_modif +=1;
            this->batch_states.clear();
            this->batch_nextstates.clear();
            this->batch_rewards.clear();
            this->batch_actions.clear();
            this->batch_dones.clear();
            this->batch_targets.clear();
        
            this->create_batch( this->batch_trainning);
            //cout<<endl<<"AP create"<<endl;
            this->Error_P += this->update_params(this->batch_states, this->batch_nextstates, this->batch_actions, 
                this->batch_dones, this->batch_rewards, this->batch_targets, true, this->nb_threads - this->env_nbThreads, false );
                        
        }
    } 
    
}






void agent::Imp_synchronous_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread,  result &res, bool implicite, bool mutexMat ){
    this->mutex_Mat = mutexMat;
    auto start_train = std::chrono::high_resolution_clock::now();
    this->mlp->error_cl.resize(nb_thread);
    this->mlp->error_rl.resize(nb_thread);
    this->mlp->position_th.resize(nb_thread);

    this->classifier->error_cl.resize(nb_thread);
    this->classifier->error_rl.resize(nb_thread);
    this->classifier->position_th.resize(nb_thread);
    
    this->begin_create_batch = true;
    std::fixed;
    cout<<"=== Training started : IMP Explicite Synchonous Paralelization ...  "<<nb_thread<<" threads"<<endl;
    time_t maintenant = time(NULL);
    tm *ltm = localtime(&maintenant); 
    ofstream fic;
    ofstream fic2;
    string fileName = "resultats_TANEKEU_Imp_synchronousPar";
    string Time_fileName =  "Time_TANEKEU_Imp_synchronousPar";
    
    if(implicite){
        fileName += "_implicite";
        Time_fileName += "_implicite";
    }

    fileName = fileName + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    Time_fileName = Time_fileName + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    fic.open(this->path_res+fileName, ofstream::app);
    fic2.open(this->path_res+Time_fileName, ofstream::app);

    cout<<"SaveFile created : "<<this->path_res+fileName<<endl;

    
    cout << std::setprecision(4);
    int limit_UD = 100000;

    init_env(nb_thread, filedata, limit_UD);
    //init_networks(this->glob_env.dims.second, this->glob_env.nb_diseases, false, nb_thread);
    //this->eps = 0;
    float epsilon_min = 0.0;
    //float end_epsilon = Epoch/4;
    float end_epsilon = 4;
    
    float epsilon_decay_value = this->eps / end_epsilon;
    int max_turn = this->glob_env.MAX_TURN;
    int C = 400;
    int train_size = this->glob_env.end_train;
    int nBatch = train_size / batch_size;
    int nBatch_rem = train_size % batch_size;
    int barWidth = 23;
    vector<thread> threads;
    threads.resize(this->nb_threads);
    this->env_nbThreads = this->nb_threads;
   
    params pars;
    pars.ag = this;

    
    string eq;
    for (int i = 0; i < barWidth; ++i)
        eq += "=";

    
    for (int iEpoch = 0; iEpoch < Epoch; iEpoch++) {
        auto start_epoch = std::chrono::high_resolution_clock::now();
        float batch_error =0.0;
        double epochError = 0.0;
        int time_for_update=0;
       
        //if(this->limit % C ==0)  update_prevParams(false);
        //cout<<"avant create threads"<<endl;
        

        int nb_env = min(this->nb_threads, batch_size); 
        this->env_nbThreads = nb_env;
        for(int t=0; t < nb_env; t++) {
            pars.num_thread = t;                
            threads[t] = thread(Imp_Epoch_dialog, pars);
        }

        for(int t=0; t<nb_env; t++) {
            threads[t].join();
        }


        //cout<<"apres join"<<endl;
        

        auto start_update = std::chrono::high_resolution_clock::now();


      
        epochError += this->synch_loss;
        /*
        epochError += update_params(this->batch_states, this->batch_nextstates, this->batch_actions, 
                this->batch_dones, this->batch_rewards, this->batch_targets, true, nb_thread, implicite );
        */
        auto end_update = std::chrono::high_resolution_clock::now();

        

        //cout <<endl<<"epochError "<<epochError <<endl;
        
        

        auto stop_epoch = chrono::high_resolution_clock::now();
        auto elapsed_epoch = chrono::duration_cast<chrono::milliseconds>(stop_epoch - start_epoch).count();     
        
      
        float glob_averageTurn = 0;
        float glob_precision =0;
        for(int i=0; i<this->nb_threads; i++){
            glob_averageTurn += this->average_turn[i];
            glob_precision += this->precision[i];
        }

        glob_averageTurn = glob_averageTurn /train_size;
        glob_precision = glob_precision/ train_size;

        if( this->eps > epsilon_min)  this->eps -= epsilon_decay_value;

        pair<float, float> res_test = test();
        
        
        cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<epochError<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<res_test.first<< " | val_AT "<<res_test.second<<
                                             " | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
        
       
        fic<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<epochError<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<res_test.first<< " | val_AT "<<res_test.second<<
                                             " | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
        
        fic2<<iEpoch +1<<" "<<elapsed_epoch<<endl;
        /*for (auto s:seuils) cout<<" "<<s;
        cout<<endl;*/
        res.acc = glob_precision;
        res.at = glob_averageTurn;
        res.acc_epoch.push_back(glob_precision);
        res.at_epoch.push_back(glob_averageTurn);
        res.err = Error_P;
        res.time_elapsed += elapsed_epoch;
        res.time_epoch.push_back(elapsed_epoch);
        res.epoch_convg = iEpoch+1;

        if(res.acc >= this->convg ){
            auto elapse_conv = chrono::high_resolution_clock::now();
            auto elapsed_c = chrono::duration_cast<chrono::milliseconds>(elapse_conv - start_train).count();
                
            bool_convg = true;
            res.time_convg = elapsed_c;
            break;
        }
        reset_metrics(this->nb_threads);
        update_prevParams(true);

    }//endEpoch
    
    auto stop_train = chrono::high_resolution_clock::now();
    auto elapse = chrono::duration_cast<chrono::milliseconds>(stop_train - start_train).count();
    res.time_elapsed = elapse;
    

    cout<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"ms"<<endl;
    fic<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"ms"<<" nb threads "<<nb_thread<<endl;
    //cout<<"av close"<<" end env "<<endEnv_threads<<endl;
    fic.close();
    fic2.close();
    //cout<<"ap close"<<endl;


}


void agent::Exp_synchronous_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread,  result &res, bool implicite ){
    auto start_train = std::chrono::high_resolution_clock::now();
    this->mlp->error_cl.resize(nb_thread);
    this->mlp->error_rl.resize(nb_thread);
    this->mlp->position_th.resize(nb_thread);

    this->classifier->error_cl.resize(nb_thread);
    this->classifier->error_rl.resize(nb_thread);
    this->classifier->position_th.resize(nb_thread);
    
    this->begin_create_batch = true;
    std::fixed;
    cout<<"=== Training started : Explicite Synchonous Paralelization ... "<<endl;
    time_t maintenant = time(NULL);
    tm *ltm = localtime(&maintenant); 
    ofstream fic;
    ofstream fic2;
    string fileName = "resultats_TANEKEU_Exp_synchronousPar";
    string Time_fileName =  "Time_TANEKEU_Exp_synchronousPar";
    
    if(implicite){
        fileName += "_implicite";
        Time_fileName += "_implicite";
    }

    fileName = fileName + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    Time_fileName = Time_fileName + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    fic.open(this->path_res+fileName, ofstream::app);
    fic2.open(this->path_res+Time_fileName, ofstream::app);

    cout<<"SaveFile created : "<<this->path_res+fileName<<endl;

    
    cout << std::setprecision(4);
    int limit_UD = 100000;

    init_env(nb_thread, filedata, limit_UD);
    //init_networks(this->glob_env.dims.second, this->glob_env.nb_diseases, false, nb_thread);
    //this->eps = 0;
    float epsilon_min = 0.0;
    //float end_epsilon = Epoch/4;
    float end_epsilon = 4;
    
    float epsilon_decay_value = this->eps / end_epsilon;
    int max_turn = this->glob_env.MAX_TURN;
    int C = 400;
    int train_size = this->glob_env.end_train;
    int nBatch = train_size / batch_size;
    int nBatch_rem = train_size % batch_size;
    int barWidth = 23;
    vector<thread> threads;
    threads.resize(this->nb_threads);
    this->env_nbThreads = this->nb_threads;
   
    params pars;
    pars.ag = this;

    
    string eq;
    for (int i = 0; i < barWidth; ++i)
        eq += "=";

    
    for (int iEpoch = 0; iEpoch < Epoch; iEpoch++) {
        auto start_epoch = std::chrono::high_resolution_clock::now();
        float batch_error =0.0;
        double epochError = 0.0;
        int time_for_update=0;
        for (int iBatch = 0; iBatch < nBatch; iBatch++) {
            if (iBatch + 1 == nBatch)
                this->limit = train_size;
            else
                this->limit = iBatch * batch_size + batch_size;
            
            //if(this->limit % C ==0)  update_prevParams(false);
            //cout<<"avant create threads"<<endl;
            int nb_env = min(this->nb_threads, batch_size); 
            for(int t=0; t < nb_env; t++) {
                pars.num_thread = t;                
        	    threads[t] = thread(Exp_synch_dialog, pars);
	        }

	        for(int t=0; t<nb_env; t++) {
		        threads[t].join();
	        }
            //cout<<"apres join"<<endl;


            /*
            float progress = 100 * float(position[0]) / float(train_size);
                cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ";
                cout << position[0] << "/" << train_size <<" [ ";
             int pos = barWidth * progress / 100;
                for (int i = 0; i < barWidth; ++i) {
                    if (i < pos)
                        cout << "=";
                    else if (i == pos)
                        cout << ">";
                    else
                        cout << "_";
                }

            cout << " ] " << int(progress) << " %"<<"real_len UD "<<this->real_len_UD;
            cout<<"\r";
            */
            auto start_update = std::chrono::high_resolution_clock::now();


            /*
            cout<<"batch size "<<batch_states.size()<<endl;
            for (int t :batch_states[0]){
                cout<<" "<<t;
            }
            exit(0);
            */

           epochError += Exp_update_params(this->batch_states, this->batch_nextstates, this->batch_actions, 
                    this->batch_dones, this->batch_rewards, this->batch_targets, true, nb_thread );
            
            this->batch_states.clear();
            this->batch_nextstates.clear();
            this->batch_rewards.clear();
            this->batch_actions.clear();
            this->batch_dones.clear();
            this->batch_targets.clear();
            
            this->end_create_batch = false;
            this->begin_create_batch = true;
            this->start_create_batch = true;
            //cout<<"av ";
            
            //create_batch(this->batch_trainning);
            //epochError += this->synch_loss;
           
            
            auto end_update = std::chrono::high_resolution_clock::now();

           

            //cout <<endl<<"epochError "<<epochError <<endl;
           
        }

        auto stop_epoch = chrono::high_resolution_clock::now();
        auto elapsed_epoch = chrono::duration_cast<chrono::milliseconds>(stop_epoch - start_epoch).count();     
        
      
        float glob_averageTurn = 0;
        float glob_precision =0;
        for(int i=0; i<this->nb_threads; i++){
            glob_averageTurn += this->average_turn[i];
            glob_precision += this->precision[i];
        }

        glob_averageTurn = glob_averageTurn /train_size;
        glob_precision = glob_precision/ train_size;

        if( this->eps > epsilon_min)  this->eps -= epsilon_decay_value;

        pair<float, float> res_test = test();
        
        /*
        cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<epochError<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<res_test.first<< " | val_AT "<<res_test.second<<
                                             " | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
        */
       
        fic<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<epochError<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<res_test.first<< " | val_AT "<<res_test.second<<
                                             " | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
        
        fic2<<iEpoch +1<<" "<<elapsed_epoch<<endl;
        /*for (auto s:seuils) cout<<" "<<s;
        cout<<endl;*/
        res.acc = glob_precision;
        res.at = glob_averageTurn;
        res.acc_epoch.push_back(glob_precision);
        res.at_epoch.push_back(glob_averageTurn);
        res.err = Error_P;
        res.time_elapsed += elapsed_epoch;
        res.time_epoch.push_back(elapsed_epoch);
        res.epoch_convg = iEpoch+1;

        if(res.acc >= this->convg ){
            auto elapse_conv = chrono::high_resolution_clock::now();
            auto elapsed_c = chrono::duration_cast<chrono::milliseconds>(elapse_conv - start_train).count();
                
            bool_convg = true;
            res.time_convg = elapsed_c;
            break;
        }
        reset_metrics(this->nb_threads);
        update_prevParams(true);
    }//endEpoch
    
    auto stop_train = chrono::high_resolution_clock::now();
    auto elapse = chrono::duration_cast<chrono::milliseconds>(stop_train - start_train).count();
    res.time_elapsed = elapse;
    
    cout<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"ms"<<endl;
    fic<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"ms"<<" nb threads "<<nb_thread<<endl;
    fic.close();
    fic2.close();


}



void agent::AdaptativeImpPar_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread,  result &res, int envTH, bool implicite, float threshold, int max_thread){
    
    auto start_train = std::chrono::high_resolution_clock::now();
    this->mlp->error_cl.resize(nb_thread);
    this->mlp->error_rl.resize(nb_thread);
    this->mlp->position_th.resize(nb_thread);

    this->classifier->error_cl.resize(nb_thread);
    this->classifier->error_rl.resize(nb_thread);
    this->classifier->position_th.resize(nb_thread);
    
    cout<<"=== Training started Adaptative parallelization IMP  ... "<<nb_thread<<" threads"<<endl;

    time_t maintenant = time(NULL);
    tm *ltm = localtime(&maintenant); 
    ofstream fic;
    ofstream fic2;
    string fileName = "resultats_TANEKEU_AdaptativePar";
    string Time_fileName =  "Time_TANEKEU_AdaptativePar";
    if(implicite){

        fileName += "_implicite";
        Time_fileName += "_implicite";
    }

    fileName = fileName + "_" + to_string(threshold) + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    Time_fileName = Time_fileName + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    fic.open(this->path_res+fileName, ofstream::app);
    fic2.open(this->path_res+Time_fileName, ofstream::app);

    cout<<"SaveFile created : "<<this->path_res+fileName<<endl;
    std::fixed;
    ///

    cout << std::setprecision(4);
    int limit_UD = 100000;

    this->N_epoch = Epoch;
    this->nb_threads = nb_thread;
    this->env_nbThreads = envTH;
    init_env(nb_thread, filedata, limit_UD);
    cout<<"end init env"<<endl;
    //init_networks(this->glob_env.dims.second, this->glob_env.nb_diseases, false, nb_thread);
    
    //this->eps = 0;
    float epsilon_min = 0.0;
    float end_epsilon = 4;
    float epsilon_decay_value = this->eps / end_epsilon;
    int max_turn = this->glob_env.MAX_TURN;
    int C = 400;
    int train_size = this->glob_env.end_train;
    int test_size = this->glob_env.dims.first - train_size;

    int barWidth = 15;
    int nBatch = train_size / batch_size;
    int nBatch_rem = train_size % batch_size;
    
    //this->env_nbThreads = 1;
    vector<thread> threads;
    threads.resize(this->nb_threads);
    
   
    
    params pars;
    pars.ag = this;

    auto start_epoch = std::chrono::high_resolution_clock::now();

    for(int t=0; t<this->env_nbThreads; t++) {
        pars.num_thread = t;                
        threads[t] = thread(dialog_v2, pars);
        threads[t].detach();
    }

    string eq;
    for (int i = 0; i < barWidth; ++i)  eq += "=";
    float iEpoch = 0;
    float error = 0.0;
    float nb_modif = 0;
    //cout<<"deb while "<<" rela len "<<this->real_len_UD<<endl;
    
    while(true){
        cout<<"\r";

        if(this->endEnv_threads == this->env_nbThreads){
            break;
        }
        //cout<<"len end iEpoch "<<this->end_iEpoch.size()<<endl;
        if(this->end_iEpoch[iEpoch] == this->env_nbThreads){   
            //cout<<" verif "<<iEpoch+1<<" reql len "<<this->real_len_UD<<endl;

            float glob_averageTurn = 0.0;
            float glob_precision = 0.0;
            float test_prec = 0.0;
            float test_avg;
            
            for(int i=0; i<this->env_nbThreads; i++){
                glob_averageTurn += this->avg_per_Epoch[i][iEpoch];
                glob_precision += this->prec_per_Epoch[i][iEpoch];
                test_prec += this->test_prec_per_Epoch[i][iEpoch];
                test_avg += this->test_avg_per_Epoch[i][iEpoch];

            }

            glob_averageTurn = glob_averageTurn/train_size;
            glob_precision = glob_precision/train_size;
            test_avg = test_avg/test_size;
            test_prec = test_prec/test_size;

            if( this->eps > epsilon_min){
                this->mtx.lock();
                this->eps -= epsilon_decay_value;
                this->mtx.unlock();
            }  
            auto stop_epoch = chrono::high_resolution_clock::now();
            auto elapsed_epoch = chrono::duration_cast<chrono::milliseconds>(stop_epoch - start_epoch).count();     

            //cout<<"real len "<<this->real_len_UD<<endl;
            /*
            cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<error<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<test_prec<< " | val_AT "<<test_avg<<
                                             "| nb_modif "<<nb_modif<<" | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
            */
            fic<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<error<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<test_prec<< " | val_AT "<<test_avg<<
                                             "| nb_modif "<<nb_modif<<" | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
            
            fic2<<iEpoch +1 <<" "<<elapsed_epoch<<endl;
            res.acc = glob_precision;
            res.at = glob_averageTurn;
            res.acc_epoch.push_back(glob_precision);
            res.at_epoch.push_back(glob_averageTurn);
            res.err = error;
            res.time_elapsed += elapsed_epoch;
            res.time_epoch.push_back(elapsed_epoch);
            res.nb_modif.push_back(nb_modif);

            if(res.acc >= this->convg ){
                bool_convg = true;
                res.time_convg = res.time_elapsed;
                res.epoch_convg = iEpoch+1;
                this->stop_env_threads = true;
                iEpoch = N_epoch;
                while (true)
                {   
                    cout<<"\r";
                    if(this->endEnv_threads == this->env_nbThreads) break;
                    //cout<<" endEnv "<<this->endEnv_threads<<" env thr "<<env_nbThreads<<" nb thread "<<this->nb_threads<<endl;

                }
                break;
            }
            

            update_prevParams(true);
            nb_modif = 0;
            iEpoch +=1;
            start_epoch = std::chrono::high_resolution_clock::now();

            if(glob_precision >= threshold or iEpoch == N_epoch){
                this->stop_env_threads = true;
                
                while (true)
                {
                    cout<<"\r";
                    if(this->endEnv_threads == this->env_nbThreads) break;
                    /*
                    cout<<" endEnv "<<this->endEnv_threads<<" env thr "<<env_nbThreads<<" nb thread "
                    <<this->nb_threads<<" stop env "<<this->stop_env_threads<<
                    "Iepoch "<<iEpoch<<" nepoc "<<N_epoch<<endl;
                    */
                }
                break;
                
            }

            
        }

       
        if(this->real_len_UD > 0){
            nb_modif +=1;
            this->batch_states.clear();
            this->batch_nextstates.clear();
            this->batch_rewards.clear();
            this->batch_actions.clear();
            this->batch_dones.clear();
            this->batch_targets.clear();
        
            create_batch( this->batch_trainning);
            error += update_params(this->batch_states, this->batch_nextstates, this->batch_actions, 
                    this->batch_dones, this->batch_rewards, this->batch_targets, true, (this->nb_threads - this->env_nbThreads), implicite );

        }
        
    }

    
    for(int i = this->env_nbThreads; i< nb_thread; i++){
        this->position[i] = i;
        this->to_move[i] = i;
    }
    this->env_nbThreads = this->nb_threads;

    cout<<"Synchnous ..."<<endl;
    for (int iEpoch = 0; iEpoch < Epoch; iEpoch++) {
        auto start_epoch = std::chrono::high_resolution_clock::now();
        float batch_error =0.0;
        double epochError = 0.0;
        int time_for_update=0;
       
        //if(this->limit % C ==0)  update_prevParams(false);
        //cout<<"avant create threads"<<endl;
        
        
        int nb_env = min(this->nb_threads, max_thread); 
        this->env_nbThreads = nb_env;
        
        for(int t=0; t < nb_env; t++) {
            pars.num_thread = t;                
            threads[t] = thread(Imp_Epoch_dialog, pars);
        }

        for(int t=0; t<nb_env; t++) {
            threads[t].join();
        }


        //cout<<"apres join"<<endl;
        

        auto start_update = std::chrono::high_resolution_clock::now();


        /*
        cout<<"batch size "<<batch_states.size()<<endl;
        for (int t :batch_states[0]){
            cout<<" "<<t;
        }
        exit(0);
        */
        /*
        
        */
        //this->end_create_batch = false;
        //this->begin_create_batch = true;
        //cout<<"av ";
        
        //create_batch(this->batch_trainning);
        epochError += this->synch_loss;
        /*
        epochError += update_params(this->batch_states, this->batch_nextstates, this->batch_actions, 
                this->batch_dones, this->batch_rewards, this->batch_targets, true, nb_thread, implicite );
        */
        auto end_update = std::chrono::high_resolution_clock::now();

        

        //cout <<endl<<"epochError "<<epochError <<endl;
        
        

        auto stop_epoch = chrono::high_resolution_clock::now();
        auto elapsed_epoch = chrono::duration_cast<chrono::milliseconds>(stop_epoch - start_epoch).count();     
        
      
        float glob_averageTurn = 0;
        float glob_precision =0;
        for(int i=0; i<this->nb_threads; i++){
            glob_averageTurn += this->average_turn[i];
            glob_precision += this->precision[i];
        }

        glob_averageTurn = glob_averageTurn /train_size;
        glob_precision = glob_precision/ train_size;

        if( this->eps > epsilon_min)  this->eps -= epsilon_decay_value;

        pair<float, float> res_test = test();
        
        /*
        cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<epochError<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<res_test.first<< " | val_AT "<<res_test.second<<
                                             " | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
        */
       
        fic<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<epochError<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<res_test.first<< " | val_AT "<<res_test.second<<
                                             " | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
        
        fic2<<iEpoch +1<<" "<<elapsed_epoch<<endl;
        /*for (auto s:seuils) cout<<" "<<s;
        cout<<endl;*/
        res.acc = glob_precision;
        res.at = glob_averageTurn;
        res.acc_epoch.push_back(glob_precision);
        res.at_epoch.push_back(glob_averageTurn);
        res.err = Error_P;
        res.time_elapsed += elapsed_epoch;
        res.time_epoch.push_back(elapsed_epoch);
        res.epoch_convg = iEpoch+1;

        if(res.acc >= this->convg ){
            auto elapse_conv = chrono::high_resolution_clock::now();
            auto elapsed_c = chrono::duration_cast<chrono::milliseconds>(elapse_conv - start_train).count();
                
            bool_convg = true;
            res.time_convg = elapsed_c;
            break;
        }
        reset_metrics(this->nb_threads);
        update_prevParams(true);

    }//endEpoch
    




    auto stop_train = chrono::high_resolution_clock::now();
    auto elapse = chrono::duration_cast<chrono::milliseconds>(stop_train - start_train).count();
    res.time_elapsed = elapse;
    cout<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"ms"<<endl;
    fic<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"ms"<<" nb threads "<<nb_thread<<endl;

    fic.close();
    fic2.close();



}




void agent::AdaptativePar_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread,  result &res, int envTH, bool implicite, float threshold){
    auto start_train = std::chrono::high_resolution_clock::now();
    cout<<"=== Training started Adaptative parallelization ... "<<endl;

    time_t maintenant = time(NULL);
    tm *ltm = localtime(&maintenant); 
    ofstream fic;
    ofstream fic2;
    string fileName = "resultats_TANEKEU_AdaptativePar";
    string Time_fileName =  "Time_TANEKEU_AdaptativePar";
    if(implicite){

        fileName += "_implicite";
        Time_fileName += "_implicite";
    }
    fileName = fileName + "_" + to_string(threshold) + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    Time_fileName = Time_fileName + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    fic.open(this->path_res+fileName, ofstream::app);
    fic2.open(this->path_res+Time_fileName, ofstream::app);

    cout<<"SaveFile created : "<<this->path_res+fileName<<endl;
    std::fixed;
    ///

    cout << std::setprecision(4);
    int limit_UD = 100000;

    this->N_epoch = Epoch;
    this->nb_threads = nb_thread;
    this->env_nbThreads = envTH;
    init_env(nb_thread, filedata, limit_UD);
    cout<<"end init env"<<endl;
    //init_networks(this->glob_env.dims.second, this->glob_env.nb_diseases, false, nb_thread);
    
    //this->eps = 0;
    float epsilon_min = 0.0;
    float end_epsilon = 4;
    float epsilon_decay_value = this->eps / end_epsilon;
    int max_turn = this->glob_env.MAX_TURN;
    int C = 400;
    int train_size = this->glob_env.end_train;
    int test_size = this->glob_env.dims.first - train_size;

    int barWidth = 15;
    int nBatch = train_size / batch_size;
    int nBatch_rem = train_size % batch_size;
    
    //this->env_nbThreads = 1;
    vector<thread> threads;
    threads.resize(this->nb_threads);
    
   
    
    params pars;
    pars.ag = this;

    auto start_epoch = std::chrono::high_resolution_clock::now();

    for(int t=0; t<this->env_nbThreads; t++) {
        pars.num_thread = t;                
        threads[t] = thread(dialog_v2, pars);
        threads[t].detach();
    }

    string eq;
    for (int i = 0; i < barWidth; ++i)  eq += "=";
    float iEpoch = 0;
    float error = 0.0;
    float nb_modif = 0;
    //cout<<"deb while "<<" rela len "<<this->real_len_UD<<endl;
    
    while(true){
        cout<<"\r";

        if(this->endEnv_threads == this->env_nbThreads){
            break;
        }
        //cout<<"len end iEpoch "<<this->end_iEpoch.size()<<endl;
        if(this->end_iEpoch[iEpoch] == this->env_nbThreads){   
            //cout<<" verif "<<iEpoch+1<<" reql len "<<this->real_len_UD<<endl;

            float glob_averageTurn = 0.0;
            float glob_precision = 0.0;
            float test_prec = 0.0;
            float test_avg;
            
            for(int i=0; i<this->env_nbThreads; i++){
                glob_averageTurn += this->avg_per_Epoch[i][iEpoch];
                glob_precision += this->prec_per_Epoch[i][iEpoch];
                test_prec += this->test_prec_per_Epoch[i][iEpoch];
                test_avg += this->test_avg_per_Epoch[i][iEpoch];

            }

            glob_averageTurn = glob_averageTurn/train_size;
            glob_precision = glob_precision/train_size;
            test_avg = test_avg/test_size;
            test_prec = test_prec/test_size;

            if( this->eps > epsilon_min){
                this->mtx.lock();
                this->eps -= epsilon_decay_value;
                this->mtx.unlock();
            }  
            auto stop_epoch = chrono::high_resolution_clock::now();
            auto elapsed_epoch = chrono::duration_cast<chrono::milliseconds>(stop_epoch - start_epoch).count();     

            //cout<<"real len "<<this->real_len_UD<<endl;
            /*
            cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<error<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<test_prec<< " | val_AT "<<test_avg<<
                                             "| nb_modif "<<nb_modif<<" | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
            */
            fic<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<error<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<test_prec<< " | val_AT "<<test_avg<<
                                             "| nb_modif "<<nb_modif<<" | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
            
            fic2<<iEpoch +1 <<" "<<elapsed_epoch<<endl;
            res.acc = glob_precision;
            res.at = glob_averageTurn;
            res.acc_epoch.push_back(glob_precision);
            res.at_epoch.push_back(glob_averageTurn);
            res.err = error;
            res.time_elapsed += elapsed_epoch;
            res.time_epoch.push_back(elapsed_epoch);
            res.nb_modif.push_back(nb_modif);

            if(res.acc >= this->convg ){
                bool_convg = true;
                res.time_convg = res.time_elapsed;
                res.epoch_convg = iEpoch+1;
                this->stop_env_threads = true;
                iEpoch = N_epoch;
                while (true)
                {   
                    cout<<"\r";
                    if(this->endEnv_threads == this->env_nbThreads) break;
                    //cout<<" endEnv "<<this->endEnv_threads<<" env thr "<<env_nbThreads<<" nb thread "<<this->nb_threads<<endl;

                }
                break;
            }
            

            update_prevParams(true);
            nb_modif = 0;
            iEpoch +=1;
            start_epoch = std::chrono::high_resolution_clock::now();

            if(glob_precision >= threshold or iEpoch == N_epoch){
                this->stop_env_threads = true;
                
                while (true)
                {
                    cout<<"\r";
                    if(this->endEnv_threads == this->env_nbThreads) break;
                    /*
                    cout<<" endEnv "<<this->endEnv_threads<<" env thr "<<env_nbThreads<<" nb thread "
                    <<this->nb_threads<<" stop env "<<this->stop_env_threads<<
                    "Iepoch "<<iEpoch<<" nepoc "<<N_epoch<<endl;
                    */
                }
                break;
                
            }

            
        }

       
        if(this->real_len_UD > 0){
            nb_modif +=1;
            this->batch_states.clear();
            this->batch_nextstates.clear();
            this->batch_rewards.clear();
            this->batch_actions.clear();
            this->batch_dones.clear();
            this->batch_targets.clear();
        
            create_batch( this->batch_trainning);
            error += update_params(this->batch_states, this->batch_nextstates, this->batch_actions, 
                    this->batch_dones, this->batch_rewards, this->batch_targets, true, (this->nb_threads - this->env_nbThreads), implicite );

        }
        
    }

    
    for(int i = this->env_nbThreads; i< nb_thread; i++){
        this->position[i] = i;
        this->to_move[i] = i;
    }
    this->env_nbThreads = this->nb_threads;

    cout<<"Synchnous ..."<<endl;
    for (iEpoch = iEpoch; iEpoch < Epoch; iEpoch++) {

        if(res.acc >= this->convg ){
            auto elapse_conv = chrono::high_resolution_clock::now();
            auto elapsed_c = chrono::duration_cast<chrono::milliseconds>(elapse_conv - start_train).count();

            bool_convg = true;
            res.time_convg = elapsed_c;
            break;
        }
        auto start_epoch = std::chrono::high_resolution_clock::now();
        float batch_error =0.0;
        double epochError = 0.0;
        int time_for_update=0;
        for (int iBatch = 0; iBatch < nBatch; iBatch++) {
            if (iBatch + 1 == nBatch)
                this->limit = train_size;
            else
                this->limit = iBatch * batch_size + batch_size;
            
            //if(this->limit % C ==0)  update_prevParams(false);

            int nb_env = min(this->nb_threads, batch_size);
            this->env_nbThreads = nb_env;
            //cout<<"av start thread"<<endl; 
            for(int t=0; t<nb_env; t++) {
                pars.num_thread = t;                
        	    threads[t] = thread(dialog, pars);
	        }

	        for(int t=0; t<nb_env; t++) {
		        threads[t].join();
	        }
            //cout<<"ap start thread"<<endl; 
            

            /*
            float progress = 100 * float(position[0]) / float(train_size);
                cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ";
                cout << position[0] << "/" << train_size <<" [ ";
             int pos = barWidth * progress / 100;
                for (int i = 0; i < barWidth; ++i) {
                    if (i < pos)
                        cout << "=";
                    else if (i == pos)
                        cout << ">";
                    else
                        cout << "_";
                }

            cout << " ] " << int(progress) << " %"<<"real_len UD "<<this->real_len_UD;
            cout<<"\r";
            */
            this->batch_states.clear();
            this->batch_nextstates.clear();
            this->batch_rewards.clear();
            this->batch_actions.clear();
            this->batch_dones.clear();
            this->batch_targets.clear();
            
            //cout<<"av ";
            
            create_batch(this->batch_trainning);
            epochError += update_params(this->batch_states, this->batch_nextstates, this->batch_actions, 
                    this->batch_dones, this->batch_rewards, this->batch_targets, true, nb_thread, implicite );

            //cout <<endl<<"epochError "<<epochError <<endl;
            
            
        }
        auto stop_epoch = chrono::high_resolution_clock::now();
        auto elapsed_epoch = chrono::duration_cast<chrono::milliseconds>(stop_epoch - start_epoch).count();     
        
        float glob_averageTurn = 0;
        float glob_precision =0;
        for(int i=0; i<this->nb_threads; i++){
            glob_averageTurn += this->average_turn[i];
            glob_precision += this->precision[i];
        }

        glob_averageTurn = glob_averageTurn/train_size;
        glob_precision = glob_precision / train_size;

        if( this->eps > epsilon_min)  this->eps -= epsilon_decay_value;

        pair<float, float> res_test = test();
        
        /*
        cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<epochError<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<res_test.first<< " | val_AT "<<res_test.second<<
                                             " | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
        */
        fic<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<epochError<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<res_test.first<< " | val_AT "<<res_test.second<<
                                             " | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
        
        fic2<<iEpoch +1<<" "<<elapsed_epoch<<endl;
        /*for (auto s:seuils) cout<<" "<<s;
        cout<<endl;*/
        res.acc = glob_precision;
        res.at = glob_averageTurn;
        res.acc_epoch.push_back(glob_precision);
        res.at_epoch.push_back(glob_averageTurn);
        res.err = Error_P;
        res.time_elapsed += elapsed_epoch;
        res.time_epoch.push_back(elapsed_epoch);
        res.epoch_convg = iEpoch+1;
        
        reset_metrics(this->nb_threads);
        update_prevParams(true);

        
    }//endEpoch
    
    




    auto stop_train = chrono::high_resolution_clock::now();
    auto elapse = chrono::duration_cast<chrono::milliseconds>(stop_train - start_train).count();
    res.time_elapsed = elapse;
    cout<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"ms"<<endl;
    fic<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"ms"<<" nb threads "<<nb_thread<<endl;

    fic.close();
    fic2.close();



}




void agent::Asynch_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread,  result &res, int envTH, bool implicite){
    auto start_train = std::chrono::high_resolution_clock::now();
    cout<<"=== Training started pV2 ... "<<endl;

    
    time_t maintenant = time(NULL);
    tm *ltm = localtime(&maintenant); 
    ofstream fic;
    ofstream fic2;
    string fileName = "resultats_TANEKEU_AsynchronousSeq";
    
    string Time_fileName =  "Time_TANEKEU_AsynchronousSeq";

    if(implicite){

        fileName += "_implicite";
        Time_fileName += "_implicite";
    }
    fileName = fileName + "_env_" + to_string(envTH)+"_" +to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    Time_fileName = Time_fileName + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    fic.open(this->path_res+fileName, ofstream::app);
    fic2.open(this->path_res+Time_fileName, ofstream::app);

    cout<<"SaveFile created : "<<this->path_res+fileName<<endl;
    std::fixed;
    ///

    cout << std::setprecision(4);
    int limit_UD = 100000;

    this->N_epoch = Epoch;
    this->nb_threads = nb_thread;
    this->env_nbThreads = envTH;
    init_env(nb_thread, filedata, limit_UD);
    cout<<"end init env"<<endl;

    //cout<<"nbthread "<<this->nb_threads<< " thread env : "<<this->env_nbThreads<<endl;
    //init_networks(this->glob_env.dims.second, this->glob_env.nb_diseases, false, nb_thread);
    
    //this->eps = 0;
    float epsilon_min = 0.0;
    //float end_epsilon = Epoch/30;
    float end_epsilon = 4;
    float epsilon_decay_value = this->eps / end_epsilon;
    int max_turn = this->glob_env.MAX_TURN;
    int C = 400;
    int train_size = this->glob_env.end_train;
    int test_size = this->glob_env.dims.first - train_size;

    int barWidth = 15;
    
    //this->env_nbThreads = 1;
    vector<thread> threads;
    threads.resize(this->env_nbThreads);
    
    
    params pars;
    pars.ag = this;

    auto start_epoch = std::chrono::high_resolution_clock::now();

    for(int t=0; t<this->env_nbThreads; t++) {
        pars.num_thread = t;                
        threads[t] = thread(dialog_v2, pars);
        threads[t].detach();
    }

    string eq;
    for (int i = 0; i < barWidth; ++i)  eq += "=";
    float iEpoch = 0;
    float error = 0.0;
    float nb_modif = 0;
    //cout<<"deb while "<<" rela len "<<this->real_len_UD<<endl;
    bool_convg=false;
    while(true){
        cout<<"\r";
         if(this->endEnv_threads == this->env_nbThreads){
            
            break;
        }
        if(this->end_iEpoch[iEpoch] == this->env_nbThreads){   
            //cout<<" verif "<<iEpoch+1<<" reql len "<<this->real_len_UD<<endl;

            float glob_averageTurn = 0.0;
            float glob_precision = 0.0;
            float test_prec = 0.0;
            float test_avg;
            
            for(int i=0; i<this->env_nbThreads; i++){
                glob_averageTurn += this->avg_per_Epoch[i][iEpoch];
                glob_precision += this->prec_per_Epoch[i][iEpoch];
                test_prec += this->test_prec_per_Epoch[i][iEpoch];
                test_avg += this->test_avg_per_Epoch[i][iEpoch];

            }

            glob_averageTurn = glob_averageTurn/train_size;
            glob_precision = glob_precision/train_size;
            test_avg = test_avg/test_size;
            test_prec = test_prec/test_size;

            if( this->eps > epsilon_min){
                this->mtx.lock();
                this->eps -= epsilon_decay_value;
                this->mtx.unlock();
            }  

            auto stop_epoch = chrono::high_resolution_clock::now();
            auto elapsed_epoch = chrono::duration_cast<chrono::milliseconds>(stop_epoch - start_epoch).count();     

            //cout<<"real len "<<this->real_len_UD<<endl;
            /*
            cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<error<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<test_prec<< " | val_AT "<<test_avg<<
                                             "| nb_modif "<<nb_modif<<" | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
            */
            fic<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<error<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<test_prec<< " | val_AT "<<test_avg<<
                                             "| nb_modif "<<nb_modif<<" | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
            
            fic2<<iEpoch +1 <<" "<<elapsed_epoch<<endl;
            res.acc = glob_precision;
            res.at = glob_averageTurn;
            res.acc_epoch.push_back(glob_precision);
            res.at_epoch.push_back(glob_averageTurn);
            res.err = error;
            res.time_elapsed += elapsed_epoch;
            res.time_epoch.push_back(elapsed_epoch);
            res.nb_modif.push_back(nb_modif);
            
            if(res.acc >= this->convg ){
                /*
                cout<<"res acc"<<res.acc<<" bool_convg "<<bool_convg<<" this-cong "<<this->convg
                <<" endenv "<<this->endEnv_threads<<" env thr "<<this->env_nbThreads<<endl;
                */
                bool_convg = true;
                res.time_convg = res.time_elapsed;
                res.epoch_convg = iEpoch+1;
                
                this->stop_env_threads = true;
                 while (true)
                {
                    cout<<"\r";
                    if(this->endEnv_threads == this->env_nbThreads) break;
                }
                break;
            }
            
            
            update_prevParams(true);
            nb_modif = 0;
            iEpoch +=1;
            start_epoch = std::chrono::high_resolution_clock::now();

            if(iEpoch == N_epoch){
                res.epoch_convg = iEpoch+1;
                break;
            }   
        }

       

        if(this->real_len_UD > 0){
            nb_modif +=1;
            this->batch_states.clear();
            this->batch_nextstates.clear();
            this->batch_rewards.clear();
            this->batch_actions.clear();
            this->batch_dones.clear();
            this->batch_targets.clear();
        
            create_batch( this->batch_trainning);
            error += update_params(this->batch_states, this->batch_nextstates, this->batch_actions, 
                    this->batch_dones, this->batch_rewards, this->batch_targets, false, (this->nb_threads - this->env_nbThreads), implicite );

        }
        
    }

    this->reset_metrics(this->nb_threads);
    auto stop_train = chrono::high_resolution_clock::now();
    auto elapse = chrono::duration_cast<chrono::milliseconds>(stop_train - start_train).count();
    res.time_elapsed = elapse;
    fic<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"ms"<<" nb threads "<<nb_thread<<endl;
    cout<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"ms"<<endl;


    fic.close();
    fic2.close();

}




void agent::Asynch_Adap_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread,  result &res, int MaxEnvTH, bool implicite){
    auto start_train = std::chrono::high_resolution_clock::now();
    cout<<"=== Training started Asynch_adap ... "<<endl;

    
    time_t maintenant = time(NULL);
    tm *ltm = localtime(&maintenant); 
    ofstream fic;
    ofstream fic2;
    string fileName = "resultats_TANEKEU_AsynchronousAdapPar";
    
    string Time_fileName =  "Time_TANEKEU_AsynchronousAdapPar";

    if(implicite){

        fileName += "_implicite";
        Time_fileName += "_implicite";
    }
    fileName = fileName + "_env_" + to_string(MaxEnvTH)+"_" +to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    Time_fileName = Time_fileName + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    fic.open(this->path_res+fileName, ofstream::app);
    fic2.open(this->path_res+Time_fileName, ofstream::app);

    cout<<"SaveFile created : "<<this->path_res+fileName<<endl;
    std::fixed;
    ///

    cout << std::setprecision(4);
    int limit_UD = 100000;

    this->N_epoch = Epoch;
    this->nb_threads = nb_thread;
    this->env_nbThreads = MaxEnvTH;
    init_env(nb_thread, filedata, limit_UD);
    cout<<"end init env : "<<this->env_nbThreads<<endl;

    //cout<<"nbthread "<<this->nb_threads<< " thread env : "<<this->env_nbThreads<<endl;
    //init_networks(this->glob_env.dims.second, this->glob_env.nb_diseases, false, nb_thread);
    
    //this->eps = 0;
    float epsilon_min = 0.0;
    //float end_epsilon = Epoch/30;
    float end_epsilon = 4;
    float epsilon_decay_value = this->eps / end_epsilon;
    int max_turn = this->glob_env.MAX_TURN;
    int C = 400;
    int train_size = this->glob_env.end_train;
    int test_size = this->glob_env.dims.first - train_size;

    int barWidth = 15;
    
    //this->env_nbThreads = 1;
    vector<thread> threads;
    threads.resize(this->env_nbThreads);
    
    
    params pars;
    pars.ag = this;

    auto start_epoch = std::chrono::high_resolution_clock::now();

    for(int t=0; t<this->env_nbThreads; t++) {
        pars.num_thread = t;                
        threads[t] = thread(dialog_v2, pars);
        threads[t].detach();
    }

    string eq;
    for (int i = 0; i < barWidth; ++i)  eq += "=";
    float iEpoch = 0;
    float error = 0.0;
    float nb_modif = 0;
    //cout<<"deb while "<<" rela len "<<this->real_len_UD<<endl;
    bool_convg=false;
    while(true){
        cout<<"\r";
         if(this->endEnv_threads == this->env_nbThreads){
            
            break;
        }
        if(this->end_iEpoch[iEpoch] == this->env_nbThreads){   
            //cout<<" verif "<<iEpoch+1<<" reql len "<<this->real_len_UD<<endl;
            float glob_averageTurn = 0.0;
            float glob_precision = 0.0;
            float test_prec = 0.0;
            float test_avg;
            
            for(int i=0; i<this->env_nbThreads; i++){
                glob_averageTurn += this->avg_per_Epoch[i][iEpoch];
                glob_precision += this->prec_per_Epoch[i][iEpoch];
                test_prec += this->test_prec_per_Epoch[i][iEpoch];
                test_avg += this->test_avg_per_Epoch[i][iEpoch];

            }

            glob_averageTurn = glob_averageTurn/train_size;
            glob_precision = glob_precision/train_size;
            test_avg = test_avg/test_size;
            test_prec = test_prec/test_size;

            if( this->eps > epsilon_min){
                this->mtx.lock();
                this->eps -= epsilon_decay_value;
                this->mtx.unlock();
            }  

            auto stop_epoch = chrono::high_resolution_clock::now();
            auto elapsed_epoch = chrono::duration_cast<chrono::milliseconds>(stop_epoch - start_epoch).count();     

            //cout<<"real len "<<this->real_len_UD<<endl;
            /*
            cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<error<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<test_prec<< " | val_AT "<<test_avg<<
                                             "| nb_modif "<<nb_modif<<" | env_th "<<this->env_nbThreads<<" | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
            */
            fic<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<error<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<test_prec<< " | val_AT "<<test_avg<<
                                             "| nb_modif "<<nb_modif<<" | env_th "<<this->env_nbThreads<<" | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
            
            fic2<<iEpoch +1 <<" "<<elapsed_epoch<<endl;
            res.acc = glob_precision;
            res.at = glob_averageTurn;
            res.acc_epoch.push_back(glob_precision);
            res.at_epoch.push_back(glob_averageTurn);
            res.err = error;
            res.time_elapsed += elapsed_epoch;
            res.time_epoch.push_back(elapsed_epoch);
            res.nb_modif.push_back(nb_modif);
            
            if(res.acc >= this->convg ){
                /*
                cout<<"res acc"<<res.acc<<" bool_convg "<<bool_convg<<" this-cong "<<this->convg
                <<" endenv "<<this->endEnv_threads<<" env thr "<<this->env_nbThreads<<endl;
                */
                bool_convg = true;
                res.time_convg = res.time_elapsed;
                res.epoch_convg = iEpoch+1;
                
                this->stop_env_threads = true;
                 while (true)
                {
                    cout<<"\r";
                    if(this->endEnv_threads == this->env_nbThreads) break;
                }
                break;
            }
            
            
            update_prevParams(true);
            nb_modif = 0;
            iEpoch +=1;
            start_epoch = std::chrono::high_resolution_clock::now();

            if(iEpoch == N_epoch){
                res.epoch_convg = iEpoch+1;
                break;
            }

            //update number of threads for environement interaction 

            this->env_nbThreads = max(1, min(MaxEnvTH, int((1-res.acc)*this->nb_threads)));


        }

       

        if(this->real_len_UD > 0){
            nb_modif +=1;
            this->batch_states.clear();
            this->batch_nextstates.clear();
            this->batch_rewards.clear();
            this->batch_actions.clear();
            this->batch_dones.clear();
            this->batch_targets.clear();
        
            create_batch( this->batch_trainning);
            error += update_params(this->batch_states, this->batch_nextstates, this->batch_actions, 
                    this->batch_dones, this->batch_rewards, this->batch_targets, true, (this->nb_threads - this->env_nbThreads), implicite );

        }
        
    }

    this->reset_metrics(this->nb_threads);
    auto stop_train = chrono::high_resolution_clock::now();
    auto elapse = chrono::duration_cast<chrono::milliseconds>(stop_train - start_train).count();
    res.time_elapsed = elapse;
    fic<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"ms"<<" nb threads "<<nb_thread<<endl;
    cout<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"ms"<<endl;


    fic.close();
    fic2.close();

}





void agent::parV2_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread,  result &res, int envTH, bool implicite){
    auto start_train = std::chrono::high_resolution_clock::now();
    cout<<"=== Training started pV2 ... "<<endl;

    
    time_t maintenant = time(NULL);
    tm *ltm = localtime(&maintenant); 
    ofstream fic;
    ofstream fic2;
    string fileName = "resultats_TANEKEU_AsynchronousPar";
    
    string Time_fileName =  "Time_TANEKEU_AsynchronousPar";

    if(implicite){

        fileName += "_implicite";
        Time_fileName += "_implicite";
    }
    fileName = fileName + "_env_" + to_string(envTH)+"_" +to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    Time_fileName = Time_fileName + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    fic.open(this->path_res+fileName, ofstream::app);
    fic2.open(this->path_res+Time_fileName, ofstream::app);

    cout<<"SaveFile created : "<<this->path_res+fileName<<endl;
    std::fixed;
    ///

    cout << std::setprecision(4);
    int limit_UD = 100000;

    this->N_epoch = Epoch;
    this->nb_threads = nb_thread;
    this->env_nbThreads = envTH;
    init_env(nb_thread, filedata, limit_UD);
    cout<<"end init env : "<<this->env_nbThreads<<endl;

    //cout<<"nbthread "<<this->nb_threads<< " thread env : "<<this->env_nbThreads<<endl;
    //init_networks(this->glob_env.dims.second, this->glob_env.nb_diseases, false, nb_thread);
    
    //this->eps = 0;
    float epsilon_min = 0.0;
    //float end_epsilon = Epoch/30;
    float end_epsilon = 4;
    float epsilon_decay_value = this->eps / end_epsilon;
    int max_turn = this->glob_env.MAX_TURN;
    int C = 400;
    int train_size = this->glob_env.end_train;
    int test_size = this->glob_env.dims.first - train_size;

    int barWidth = 15;
    
    //this->env_nbThreads = 1;
    vector<thread> threads;
    threads.resize(this->env_nbThreads);
    
    
    params pars;
    pars.ag = this;

    auto start_epoch = std::chrono::high_resolution_clock::now();

    for(int t=0; t<this->env_nbThreads; t++) {
        pars.num_thread = t;                
        threads[t] = thread(dialog_v2, pars);
        threads[t].detach();
    }

    string eq;
    for (int i = 0; i < barWidth; ++i)  eq += "=";
    float iEpoch = 0;
    float error = 0.0;
    float nb_modif = 0;
    //cout<<"deb while "<<" rela len "<<this->real_len_UD<<endl;
    bool_convg=false;
    while(true){
        cout<<"\r";
         if(this->endEnv_threads == this->env_nbThreads){
            
            break;
        }
        if(this->end_iEpoch[iEpoch] == this->env_nbThreads){   
            //cout<<" verif "<<iEpoch+1<<" reql len "<<this->real_len_UD<<endl;

            float glob_averageTurn = 0.0;
            float glob_precision = 0.0;
            float test_prec = 0.0;
            float test_avg;
            
            for(int i=0; i<this->env_nbThreads; i++){
                glob_averageTurn += this->avg_per_Epoch[i][iEpoch];
                glob_precision += this->prec_per_Epoch[i][iEpoch];
                test_prec += this->test_prec_per_Epoch[i][iEpoch];
                test_avg += this->test_avg_per_Epoch[i][iEpoch];

            }

            glob_averageTurn = glob_averageTurn/train_size;
            glob_precision = glob_precision/train_size;
            test_avg = test_avg/test_size;
            test_prec = test_prec/test_size;

            if( this->eps > epsilon_min){
                this->mtx.lock();
                this->eps -= epsilon_decay_value;
                this->mtx.unlock();
            }  

            auto stop_epoch = chrono::high_resolution_clock::now();
            auto elapsed_epoch = chrono::duration_cast<chrono::milliseconds>(stop_epoch - start_epoch).count();     

            //cout<<"real len "<<this->real_len_UD<<endl;
            /*
            cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<error<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<test_prec<< " | val_AT "<<test_avg<<
                                             "| nb_modif "<<nb_modif<<" | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
            */
            fic<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<error<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<test_prec<< " | val_AT "<<test_avg<<
                                             "| nb_modif "<<nb_modif<<" | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
            
            fic2<<iEpoch +1 <<" "<<elapsed_epoch<<endl;
            res.acc = glob_precision;
            res.at = glob_averageTurn;
            res.acc_epoch.push_back(glob_precision);
            res.at_epoch.push_back(glob_averageTurn);
            res.err = error;
            res.time_elapsed += elapsed_epoch;
            res.time_epoch.push_back(elapsed_epoch);
            res.nb_modif.push_back(nb_modif);
            
            if(res.acc >= this->convg ){
                /*
                cout<<"res acc"<<res.acc<<" bool_convg "<<bool_convg<<" this-cong "<<this->convg
                <<" endenv "<<this->endEnv_threads<<" env thr "<<this->env_nbThreads<<endl;
                */
                bool_convg = true;
                res.time_convg = res.time_elapsed;
                res.epoch_convg = iEpoch+1;
                
                this->stop_env_threads = true;
                 while (true)
                {
                    cout<<"\r";
                    if(this->endEnv_threads == this->env_nbThreads) break;
                }
                break;
            }
            
            
            update_prevParams(true);
            nb_modif = 0;
            iEpoch +=1;
            start_epoch = std::chrono::high_resolution_clock::now();

            if(iEpoch == N_epoch){
                res.epoch_convg = iEpoch+1;
                break;
            }   
        }

       

        if(this->real_len_UD > 0){
            nb_modif +=1;
            this->batch_states.clear();
            this->batch_nextstates.clear();
            this->batch_rewards.clear();
            this->batch_actions.clear();
            this->batch_dones.clear();
            this->batch_targets.clear();
        
            create_batch( this->batch_trainning);
            error += update_params(this->batch_states, this->batch_nextstates, this->batch_actions, 
                    this->batch_dones, this->batch_rewards, this->batch_targets, true, (this->nb_threads - this->env_nbThreads), implicite );

        }
        
    }

    this->reset_metrics(this->nb_threads);
    auto stop_train = chrono::high_resolution_clock::now();
    auto elapse = chrono::duration_cast<chrono::milliseconds>(stop_train - start_train).count();
    res.time_elapsed = elapse;
    fic<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"ms"<<" nb threads "<<nb_thread<<endl;
    cout<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"ms"<<endl;


    fic.close();
    fic2.close();

}


void agent::parV3_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread,  result &res, bool implicite){
    auto start_train = std::chrono::high_resolution_clock::now();
    std::fixed;
    cout << std::setprecision(4);
    int limit_UD = 100000;
    this->nb_threads = nb_thread;
    this->env_nbThreads = this->nb_threads/2;
    vector<thread> threads;
    threads.resize(this->env_nbThreads);
    params pars;
    pars.ag = this;
    float epsilon_min = 0.0;
    float end_epsilon = Epoch/30;
    float epsilon_decay_value = this->eps / end_epsilon;
    int max_turn = this->glob_env.MAX_TURN;
    int C = 400;

    init_env(nb_thread, filedata, limit_UD);
    //init_networks(this->glob_env.dims.second, this->glob_env.nb_diseases, false, this->nb_threads);
    //this->eps = 0;
    
    int train_size = this->glob_env.end_train;
    int nBatch = train_size / batch_size;
    int nBatch_rem = train_size % batch_size;
    
    int barWidth = 15;
    
    string eq;
    for (int i = 0; i < barWidth; ++i)
        eq += "=";

    cout<<"=== Training started pV3 ... "<<endl;
    
    for (int iEpoch = 0; iEpoch < Epoch; iEpoch++) {
        auto start_epoch = std::chrono::high_resolution_clock::now();
        float batch_error =0.0;
        double epochError = 0.0;
        int time_for_update=0;
        //cout<<endl<<"deb"<<endl;
        for(int t=0; t<this->env_nbThreads; t++) {
                pars.num_thread = t;                
        	    threads[t] = thread(dialog_Epoch, pars);
                threads[t].detach();
	    }
        //threads[2] = thread(par_trainParams);
        par_trainParams(iEpoch);
            /*
	        for(int t=0; t<this->nb_threads; t++) {
		        threads[t].join();
	        }*/

        
        
        float glob_averageTurn = 0;
        float glob_precision =0;
        //cout<<"ok "<<" iEpoch "<<iEpoch<<endl;
        for(int i=0; i<this->env_nbThreads; i++){
            glob_averageTurn += this->average_turn[i];
            glob_precision += this->precision[i];
        }
        glob_averageTurn = glob_averageTurn/train_size;
        glob_precision = glob_precision/train_size;
        //cout<<"End ok "<<" iEpoch "<<iEpoch<<endl;



        if( this->eps > epsilon_min)  this->eps -= epsilon_decay_value;

        pair<float, float> res_test = test();
        
        auto stop_epoch = chrono::high_resolution_clock::now();
        auto elapsed_epoch = chrono::duration_cast<chrono::milliseconds>(stop_epoch - start_epoch).count();     

        /*
        cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<this->Error_P<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<res_test.first<< " | val_AT "<<res_test.second
                                             <<" | nb "<<nb_modif<<" | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
        */
        //cout<<"real len "<<this->real_len_UD<<endl;
        update_prevParams(false);
        res.acc = glob_precision;
        res.at = glob_averageTurn;
        res.err = Error_P;
        res.acc_epoch.push_back(glob_precision);
        res.at_epoch.push_back(glob_averageTurn);
        res.time_elapsed += elapsed_epoch;
        res.time_epoch.push_back(elapsed_epoch);
        res.epoch_convg = iEpoch+1;

        if(res.acc >= this->convg ){
            bool_convg = true;
            res.time_convg = res.time_elapsed;
                
        }
        reset_metrics(this->env_nbThreads);
    }//endEpoch
    auto stop_train = chrono::high_resolution_clock::now();
    auto elapse = chrono::duration_cast<chrono::milliseconds>(stop_train - start_train).count();
    res.time_elapsed = elapse;
                
    cout<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"ms"<<endl;


}


void agent::par_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread,  result &res, bool implicite ){
    auto start_train = std::chrono::high_resolution_clock::now();
    std::fixed;
    cout<<"=== Training started : Synchonous Paralelization ... "<<endl;
    time_t maintenant = time(NULL);
    tm *ltm = localtime(&maintenant); 
    ofstream fic;
    ofstream fic2;
    string fileName = "resultats_TANEKEU_synchronousPar";
    string Time_fileName =  "Time_TANEKEU_synchronousPar";
    if(implicite){

        fileName += "_implicite";
        Time_fileName += "_implicite";
    }
    fileName = fileName + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    Time_fileName = Time_fileName + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    fic.open(this->path_res+fileName, ofstream::app);
    fic2.open(this->path_res+Time_fileName, ofstream::app);

    cout<<"SaveFile created : "<<this->path_res+fileName<<endl;

    
    cout << std::setprecision(4);
    int limit_UD = 100000;

    init_env(nb_thread, filedata, limit_UD);
    //init_networks(this->glob_env.dims.second, this->glob_env.nb_diseases, false, nb_thread);
    //this->eps = 0;
    float epsilon_min = 0.0;
    //float end_epsilon = Epoch/4;
    float end_epsilon = 4;
    
    float epsilon_decay_value = this->eps / end_epsilon;
    int max_turn = this->glob_env.MAX_TURN;
    int C = 400;
    int train_size = this->glob_env.end_train;
    int nBatch = train_size / batch_size;
    int nBatch_rem = train_size % batch_size;
    int barWidth = 23;
    vector<thread> threads;
    threads.resize(this->nb_threads);
    this->env_nbThreads = this->nb_threads;
    
    params pars;
    pars.ag = this;

    
    string eq;
    for (int i = 0; i < barWidth; ++i)
        eq += "=";

    
    for (int iEpoch = 0; iEpoch < Epoch; iEpoch++) {
        auto start_epoch = std::chrono::high_resolution_clock::now();
        float batch_error =0.0;
        double epochError = 0.0;
        int time_for_update=0;
        for (int iBatch = 0; iBatch < nBatch; iBatch++) {
            if (iBatch + 1 == nBatch)
                this->limit = train_size;
            else
                this->limit = iBatch * batch_size + batch_size;
            
            //if(this->limit % C ==0)  update_prevParams(false);
            int nb_env = min(this->nb_threads, batch_size); 
            this->env_nbThreads = nb_env;

            for(int t=0; t < nb_env; t++) {
                pars.num_thread = t;                
        	    //threads[t] = thread(dialog, pars);
                threads[t] = thread(dialog, pars);
	        }

	        for(int t=0; t<nb_env; t++) {
		        threads[t].join();
	        }


            /*
            float progress = 100 * float(position[0]) / float(train_size);
                cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ";
                cout << position[0] << "/" << train_size <<" [ ";
             int pos = barWidth * progress / 100;
                for (int i = 0; i < barWidth; ++i) {
                    if (i < pos)
                        cout << "=";
                    else if (i == pos)
                        cout << ">";
                    else
                        cout << "_";
                }

            cout << " ] " << int(progress) << " %"<<"real_len UD "<<this->real_len_UD;
            cout<<"\r";
            */
            auto start_update = std::chrono::high_resolution_clock::now();

            this->batch_states.clear();
            this->batch_nextstates.clear();
            this->batch_rewards.clear();
            this->batch_actions.clear();
            this->batch_dones.clear();
            this->batch_targets.clear();
            
            //cout<<"av ";
            
            
            create_batch(this->batch_trainning);
            
            epochError += update_params(this->batch_states, this->batch_nextstates, this->batch_actions, 
                    this->batch_dones, this->batch_rewards, this->batch_targets, true, nb_thread, implicite );
            


            //epochError += this->synch_loss; 
            auto end_update = std::chrono::high_resolution_clock::now();

            auto temp = chrono::duration_cast<chrono::milliseconds>(end_update - start_update).count();
            
            time_for_update += temp;

            //cout <<endl<<"epochError "<<epochError <<endl;
            
            
        }
        auto stop_epoch = chrono::high_resolution_clock::now();
        auto elapsed_epoch = chrono::duration_cast<chrono::milliseconds>(stop_epoch - start_epoch).count();     
        
        float glob_averageTurn = 0;
        float glob_precision =0;
        for(int i=0; i<this->nb_threads; i++){
            glob_averageTurn += this->average_turn[i];
            glob_precision += this->precision[i];
        }

        glob_averageTurn = glob_averageTurn /train_size;
        glob_precision = glob_precision/ train_size;

        if( this->eps > epsilon_min)  this->eps -= epsilon_decay_value;

        pair<float, float> res_test = test();
        
        /*
        cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<epochError<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<res_test.first<< " | val_AT "<<res_test.second<<
                                             " | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
        */
       
        fic<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<epochError<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<res_test.first<< " | val_AT "<<res_test.second<<
                                             " | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
        
        fic2<<iEpoch +1<<" "<<elapsed_epoch<<endl;
        /*for (auto s:seuils) cout<<" "<<s;
        cout<<endl;*/
        res.acc = glob_precision;
        res.at = glob_averageTurn;
        res.acc_epoch.push_back(glob_precision);
        res.at_epoch.push_back(glob_averageTurn);
        res.err = Error_P;
        res.time_elapsed += elapsed_epoch;
        res.time_epoch.push_back(elapsed_epoch);
        res.epoch_convg = iEpoch+1;

        if(res.acc >= this->convg ){
            auto elapse_conv = chrono::high_resolution_clock::now();
            auto elapsed_c = chrono::duration_cast<chrono::milliseconds>(elapse_conv - start_train).count();
                
            bool_convg = true;
            res.time_convg = elapsed_c;
            break;
        }
        reset_metrics(this->nb_threads);
        update_prevParams(true);
    }//endEpoch
    
    auto stop_train = chrono::high_resolution_clock::now();
    auto elapse = chrono::duration_cast<chrono::milliseconds>(stop_train - start_train).count();
    res.time_elapsed = elapse;
    
    cout<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"ms"<<endl;
    fic<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"ms"<<" nb threads "<<nb_thread<<endl;
    fic.close();
    fic2.close();


}


void agent::seq_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, result &res){
    auto start_train = std::chrono::high_resolution_clock::now();
    std::fixed;
    cout << std::setprecision(4);
    
    cout<<"=== Training started : Sequential version ... "<<endl;
    time_t maintenant = time(NULL);
    tm *ltm = localtime(&maintenant); 
    ofstream fic;
    ofstream fic2;
    string fileName = "resultats_TANEKEU_Sequential";
    string Time_fileName =  "Time_TANEKEU_Sequential";

    fileName = fileName + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    Time_fileName = Time_fileName + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    fic.open(this->path_res+fileName, ofstream::app);
    fic2.open(this->path_res+Time_fileName, ofstream::app);

    cout<<"SaveFile created : "<<this->path_res+fileName<<endl;


    int limit_UD = 100000;
    init_env(1, filedata, limit_UD);
    //init_networks(this->glob_env.dims.second, this->glob_env.nb_diseases, true, 1);
    
    float epsilon = this->eps;
    //float epsilon = 0;
    float epsilon_min = 0.0;
    //float end_epsilon = Epoch/4;
    float end_epsilon = 4;
    
    float epsilon_decay_value = epsilon / end_epsilon;
    int max_turn = this->glob_env.MAX_TURN;
    int C = 400;
    int train_size = this->glob_env.end_train;
    int nBatch = train_size / batch_size;
    int nBatch_rem = train_size % batch_size;
    int limit;
    int barWidth = 23;
    
   

    //vector<float> seuils(this->glob_env.nb_diseases, 0.5);
    
    string eq;
    for (int i = 0; i < barWidth; ++i)
        eq += "=";

    
    for (int iEpoch = 0; iEpoch < Epoch; iEpoch++) {
        auto start_epoch = std::chrono::high_resolution_clock::now();
        float batch_error =0.0;
        long time_for_update = 0;

        //int limit;
        double epochError = 0.0;
        float prec = 0.0;
        float cap = 0.0;
        float avg_turn =0.0;
        for (int iBatch = 0; iBatch < nBatch; iBatch++) {
            if (iBatch + 1 == nBatch)
                limit = train_size;
            else
                limit = iBatch * batch_size + batch_size;
            
            //cout<<endl<<"AV init state"<<endl;
            for (int id_obs = iBatch * batch_size; id_obs < limit; id_obs++) {//begin batch
                //if(id_obs % C ==0)  update_prevParams(true);
                //if(iEpoch % 2 ==0)  update_prevParams();
                /*
                float progress = 100 * float(id_obs) / float(train_size);
                cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ";
                cout << id_obs << "/" << train_size << " [ ";
                */
                vector<float> current_state(this->glob_env.dims.second,0);
                vector<float> t_symp(this->glob_env.dims.second,0);
                vector<int> all_symp;
                int action;
                int t_disease, dt;
                pair<int, float> diag;
                envs[0].initial_state(id_obs, current_state, t_symp, t_disease, all_symp);
                diag = choose_diagnostic(current_state, 0, true);
                envs[0].H_s0 = diag.second;
                //cout<<endl<<"AP init state"<<endl;
                for(int turn = 0; turn < max_turn; turn++){
                  
                    
                    float reward;
                    
                    float H_st = diag.second;
                    
                    dt = diag.first;
                    this->states[to_move[0]].assign(current_state.begin(), current_state.end());
                    action = choose_action(current_state, epsilon, true, 0, true, id_obs);
                    envs[0].step(current_state, t_symp, all_symp, reward, action, turn, t_disease, diag);
                    diag = choose_diagnostic(current_state, 0 , true);
            
                    float rh = (H_st -  diag.second )/envs[0].H_s0;
                   
                    if( rh < 0) rh = 0;
                    reward = reward + (2.5*rh);
                    
                    this->rewards[to_move[0]] = reward;
                    this->next_states[to_move[0]].assign(current_state.begin(), current_state.end());
                    this->actions[to_move[0]] = action;
                    this->targets[to_move[0]] = t_disease;
                    int t_mv = to_move[0];


                     if((this->to_move[0] + 1) >= this->limit_UD){
                        this->to_move[0] = 0;

                     }else{
                        this->to_move[0] += 1;
                     }
                    real_len_UD  = min(limit_UD, real_len_UD +1);

                    if( (H_st < seuils[dt]) or (turn == max_turn -1) ){
                        this->dones[t_mv] = true;

                        if(dt == t_disease and turn != (max_turn-1)){
                            prec += 1;
                            if( (seuils[t_disease] - H_st) < 0.2){
                                float l = 0.3;
                                seuils[t_disease] = l*seuils[t_disease] + (1-l)*H_st;
                            }
                        } 
                        avg_turn += turn+1; 
                        break;
                    }
                    else{
                        this->dones[t_mv] = false;
                    }
                } //end Dialog
                
                cap += this->envs[0].count_sympt/all_symp.size();
                
                /*
                int pos = barWidth * progress / 100;
                for (int i = 0; i < barWidth; ++i) {
                    if (i < pos)
                        cout << "=";
                    else if (i == pos)
                        cout << ">";
                    else
                        cout << "_";
                }

                cout << " ] " << int(progress) << " %";
                
                */
                if(id_obs % batch_size == 0){
                    auto stop_epoch = chrono::high_resolution_clock::now();
                    auto elapsed_epoch = chrono::duration_cast<chrono::milliseconds>(stop_epoch - start_epoch).count();     
                    //cout <<"| Error: "<<epochError<<"| Acc: "<<precision/(limit)<< " Cap "<<capture/(limit)<< " | elaps:"<< elapsed_epoch<<"s" ;
                    //cout <<"| Error: "<<epochError<<"| Acc: "<<prec/(limit)<< " | elaps:"<< elapsed_epoch/1000<<"s" ;

                }
                
                //cout<<"\r";

                
            }//end Batch
            
            this->batch_states.clear();
            this->batch_nextstates.clear();
            this->batch_rewards.clear();
            this->batch_actions.clear();
            this->batch_dones.clear();
            this->batch_targets.clear();
            //cout<<"av ";
            create_batch( this->batch_trainning);
            epochError += update_params(this->batch_states, this->batch_nextstates, this->batch_actions, this->batch_dones, this->batch_rewards,
                             this->batch_targets, false , 1);
            //cout <<"| ok: pre " <<endl;
            
            
        }
        auto stop_epoch = chrono::high_resolution_clock::now();
        auto elapsed_epoch = chrono::duration_cast<chrono::seconds>(stop_epoch - start_epoch).count();     
        
        if( epsilon > epsilon_min)  epsilon -= epsilon_decay_value; 
        pair<float, float> res_test = test();
        
        /*
        cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<epochError<<" | Acc: "<<prec/train_size<<" AT "
                                <<avg_turn/train_size<<" | val_Acc "<<res_test.first<< " | val_AT "<<res_test.second<<
                                             " | elaps: "<< elapsed_epoch<<"s"<<endl;
        */
        fic<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<epochError<<" | Acc: "<<prec/train_size<<" AT "
                                <<avg_turn/train_size<<" | val_Acc "<<res_test.first<< " | val_AT "<<res_test.second<<
                                             " | elaps: "<< elapsed_epoch<<"s"<<endl;
        
        fic2<<iEpoch +1<<" "<< elapsed_epoch<<endl;

        /*for (auto s:seuils) cout<<" "<<s;
        cout<<endl;*/
        res.acc = prec/train_size;
        res.at = avg_turn/train_size;
        res.err = epochError;
        res.time_elapsed += elapsed_epoch;
        res.time_epoch.push_back(elapsed_epoch);
        res.epoch_convg = iEpoch+1;

        res.acc_epoch.push_back(prec/train_size);
        res.at_epoch.push_back(avg_turn/train_size);
        if(res.acc >= this->convg ){
            auto elapse_conv = chrono::high_resolution_clock::now();
            auto elapsed_c = chrono::duration_cast<chrono::milliseconds>(elapse_conv - start_train).count();

            bool_convg = true;
            res.time_convg = elapsed_c;
            break;
        }
        update_prevParams(true);
        reset_metrics(this->nb_threads);
    }//endEpoch

    auto stop_train = chrono::high_resolution_clock::now();
    auto elapse = chrono::duration_cast<chrono::milliseconds>(stop_train - start_train).count();
    res.time_elapsed = elapse;
    
    cout<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"s"<<endl;
    fic<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"s"<<endl;
    fic.close();
    fic2.close();
    
}


void agent::parV1_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread,  result &res, bool implicite){
    auto start_train = std::chrono::high_resolution_clock::now();
    std::fixed;
    cout << std::setprecision(4);
    
    
    cout<<"=== Training started : multi-env version ... "<<endl;
    time_t maintenant = time(NULL);
    tm *ltm = localtime(&maintenant); 
    ofstream fic;
    ofstream fic2;
    string fileName = "resultats_TANEKEU_multiEnvPar";
    string Time_fileName =  "Time_TANEKEU_multiEnvPar";
    if(implicite){

        fileName += "_implicite";
        Time_fileName += "_implicite";
    }
    fileName = fileName + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    Time_fileName = Time_fileName + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    
    fic.open(this->path_res+fileName, ofstream::app);
    fic2.open(this->path_res+Time_fileName, ofstream::app);

    cout<<"SaveFile created : "<<this->path_res+fileName<<endl;

    
    int limit_UD = 100000;

    init_env(nb_thread, filedata, limit_UD);
    //init_networks(this->glob_env.dims.second, this->glob_env.nb_diseases, false, nb_thread);
    //this->eps = 0;
    float epsilon_min = 0.0;
    //float end_epsilon = Epoch/4;
    float end_epsilon =4;
    float epsilon_decay_value = this->eps / end_epsilon;
    int max_turn = this->glob_env.MAX_TURN;
    int C = 400;
    int train_size = this->glob_env.end_train;
    int nBatch = train_size / batch_size;
    int nBatch_rem = train_size % batch_size;
    int barWidth = 15;
    vector<thread> threads;
    threads.resize(this->nb_threads);
    this->env_nbThreads = this->nb_threads;
    
    params pars;
    pars.ag = this;

    
    string eq;
    for (int i = 0; i < barWidth; ++i)
        eq += "=";

    for (int iEpoch = 0; iEpoch < Epoch; iEpoch++) {
        auto start_epoch = std::chrono::high_resolution_clock::now();
        float batch_error =0.0;
        double epochError = 0.0;
        int time_for_update=0;
        for (int iBatch = 0; iBatch < nBatch; iBatch++) {
            if (iBatch + 1 == nBatch)
                this->limit = train_size;
            else
                this->limit = iBatch * batch_size + batch_size;
            
            //if(this->limit % C ==0)  update_prevParams(false);
            int nb_env = min(this->nb_threads, batch_size); 
            for(int t=0; t<nb_env; t++) {
                pars.num_thread = t;                
        	    threads[t] = thread(dialog, pars);
	        }

	        for(int t=0; t<nb_env; t++) {
		        threads[t].join();
	        }


            /*
            float progress = 100 * float(position[0]) / float(train_size);
                cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ";
                cout << position[0] << "/" << train_size <<" [ ";
             int pos = barWidth * progress / 100;
                for (int i = 0; i < barWidth; ++i) {
                    if (i < pos)
                        cout << "=";
                    else if (i == pos)
                        cout << ">";
                    else
                        cout << "_";
                }

            cout << " ] " << int(progress) << " %"<<"real_len UD "<<this->real_len_UD;
            cout<<"\r";
            */
            auto start_update = std::chrono::high_resolution_clock::now();

            this->batch_states.clear();
            this->batch_nextstates.clear();
            this->batch_rewards.clear();
            this->batch_actions.clear();
            this->batch_dones.clear();
            this->batch_targets.clear();
            
            //cout<<"av ";
            
            create_batch(this->batch_trainning);
            epochError += update_params(this->batch_states, this->batch_nextstates, this->batch_actions, 
                    this->batch_dones, this->batch_rewards, this->batch_targets, false, nb_thread, implicite );

            auto end_update = std::chrono::high_resolution_clock::now();

            auto temp = chrono::duration_cast<chrono::milliseconds>(end_update - start_update).count();
            
            time_for_update += temp;

            //cout <<endl<<"epochError "<<epochError <<endl;
            
            
        }
        auto stop_epoch = chrono::high_resolution_clock::now();
        auto elapsed_epoch = chrono::duration_cast<chrono::milliseconds>(stop_epoch - start_epoch).count();     
        
        float glob_averageTurn = 0;
        float glob_precision =0;
        for(int i=0; i<this->nb_threads; i++){
            glob_averageTurn += this->average_turn[i];
            glob_precision += this->precision[i];
        }

        glob_precision = glob_precision/train_size;
        glob_averageTurn = glob_averageTurn/train_size;

        if( this->eps > epsilon_min)  this->eps -= epsilon_decay_value;

        pair<float, float> res_test = test();
        
        /*
        cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<epochError<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<res_test.first<< " | val_AT "<<res_test.second<<
                                             " | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
        */
        fic<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<epochError<<" | Acc: "<<glob_precision<<" AT "
                                <<glob_averageTurn<<" | val_Acc "<<res_test.first<< " | val_AT "<<res_test.second<<
                                             " | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
        fic2<<iEpoch +1 <<" "<<elapsed_epoch;
        /*for (auto s:seuils) cout<<" "<<s;
        cout<<endl;*/
        res.acc = glob_precision;
        res.at = glob_averageTurn;
        res.epoch_convg = iEpoch+1;

        res.acc_epoch.push_back(glob_precision);
        res.at_epoch.push_back(glob_averageTurn);
        res.err = Error_P;
        res.time_elapsed += elapsed_epoch;
        res.time_epoch.push_back(elapsed_epoch);

        if(res.acc >= this->convg ){
           auto elapse_conv = chrono::high_resolution_clock::now();
            auto elapsed_c = chrono::duration_cast<chrono::milliseconds>(elapse_conv - start_train).count();

            bool_convg = true;
            res.time_convg = elapsed_c;
            break;
        }
        reset_metrics(this->nb_threads);
        update_prevParams(true);
    }//endEpoch
    
    auto stop_train = chrono::high_resolution_clock::now();
    auto elapse = chrono::duration_cast<chrono::milliseconds>(stop_train - start_train).count();
    res.time_elapsed = elapse;
    
    cout<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"s"<<endl;
    fic<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"s"<<endl;
    fic.close();
    fic2.close();

}

void agent::parSGD_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread, result &res, bool implicite){
    auto start_train = std::chrono::high_resolution_clock::now();
    std::fixed;
    cout << std::setprecision(4);
    
    int limit_UD = 100000;
    init_env(1, filedata, limit_UD);
    //init_networks(this->glob_env.dims.second, this->glob_env.nb_diseases, true, 1);
    
    float epsilon = this->eps;
    //float epsilon = 0;
    float epsilon_min = 0.0;
    float end_epsilon = Epoch/4;
    float epsilon_decay_value = epsilon / end_epsilon;
    int max_turn = this->glob_env.MAX_TURN;
    int C = 400;
    int train_size = this->glob_env.end_train;
    int nBatch = train_size / batch_size;
    int nBatch_rem = train_size % batch_size;
    int limit;
    int barWidth = 23;
   

    vector<float> seuils(this->glob_env.nb_diseases, 0.5);
    
    string eq;
    for (int i = 0; i < barWidth; ++i)
        eq += "=";

    cout<<"=== Training started pSGD ..."<<endl;
    
    for (int iEpoch = 0; iEpoch < Epoch; iEpoch++) {
        auto start_epoch = std::chrono::high_resolution_clock::now();
        float batch_error =0.0;
        long time_for_update = 0;

        int limit;
        double epochError = 0.0;
        float prec = 0.0;
        float cap = 0.0;
        float avg_turn =0.0;
        for (int iBatch = 0; iBatch < nBatch; iBatch++) {
            if (iBatch + 1 == nBatch)
                limit = train_size;
            else
                limit = iBatch * batch_size + batch_size;
            
            //cout<<endl<<"AV init state"<<endl;
            for (int id_obs = iBatch * batch_size; id_obs < limit; id_obs++) {//begin batch
                //if(id_obs % C ==0)  update_prevParams(true);
                //if(iEpoch % 2 ==0)  update_prevParams();
                /*
                float progress = 100 * float(id_obs) / float(train_size);
                cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ";
                cout << id_obs << "/" << train_size << " [ ";
                */
                vector<float> current_state(this->glob_env.dims.second,0);
                vector<float> t_symp(this->glob_env.dims.second,0);
                vector<int> all_symp;
                int action;
                int t_disease, dt;
                pair<int, float> diag;
                envs[0].initial_state(id_obs, current_state, t_symp, t_disease, all_symp);
                diag = choose_diagnostic(current_state, 0, true);
                envs[0].H_s0 = diag.second;
                //cout<<endl<<"AP init state"<<endl;
                for(int turn = 0; turn < max_turn; turn++){
                  
                    
                    float reward;
                    
                    float H_st = diag.second;
                    
                    dt = diag.first;
                    this->states[to_move[0]].assign(current_state.begin(), current_state.end());
                    action = choose_action(current_state, epsilon, true, 0, true, id_obs);
                    envs[0].step(current_state, t_symp, all_symp, reward, action, turn, t_disease, diag);
                    diag = choose_diagnostic(current_state, 0 , true);
            
                    float rh = (H_st -  diag.second )/envs[0].H_s0;
                   
                    if( rh < 0) rh = 0;
                    reward = reward + (2.5*rh);
                    
                    this->rewards[to_move[0]] = reward;
                    this->next_states[to_move[0]].assign(current_state.begin(), current_state.end());
                    this->actions[to_move[0]] = action;
                    this->targets[to_move[0]] = t_disease;
                    
                    if( (H_st < seuils[dt]) or (turn == max_turn -1) ){
                        this->dones[to_move[0]] = true;

                        if(dt == t_disease and turn != (max_turn-1)){
                            prec += 1;
                            if( (seuils[t_disease] - H_st) < 0.2){
                                float l = 0.3;
                                seuils[t_disease] = l*seuils[t_disease] + (1-l)*H_st;
                            }
                        } 
                        avg_turn += turn+1; 
                        break;
                    }
                    else{
                        this->dones[to_move[0]] = false;
                    }
                     if((this->to_move[0] + 1) >= this->limit_UD){
                        this->to_move[0] = 0;

                     }else{
                        this->to_move[0] += 1;
                     }
                        real_len_UD  = min(limit_UD, real_len_UD +1);
                } //end Dialog
                
                cap += this->envs[0].count_sympt/all_symp.size();
                
                /*
                int pos = barWidth * progress / 100;
                for (int i = 0; i < barWidth; ++i) {
                    if (i < pos)
                        cout << "=";
                    else if (i == pos)
                        cout << ">";
                    else
                        cout << "_";
                }

                cout << " ] " << int(progress) << " %";                
                */
                if(id_obs % batch_size == 0){
                    auto stop_epoch = chrono::high_resolution_clock::now();
                    auto elapsed_epoch = chrono::duration_cast<chrono::seconds>(stop_epoch - start_epoch).count();     
                    //cout <<"| Error: "<<epochError<<"| Acc: "<<precision/(limit)<< " Cap "<<capture/(limit)<< " | elaps:"<< elapsed_epoch<<"s" ;
                    //cout <<"| Error: "<<epochError<<"| Acc: "<<prec/(limit)<< " | elaps:"<< elapsed_epoch<<"s" ;

                }
                
                //cout<<"\r";

                
            }//end Batch
            
            //cout<<"av ";
            create_batch(this->batch_trainning);
            epochError += update_params(this->batch_states, this->batch_nextstates, this->batch_actions, this->batch_dones, this->batch_rewards,
                             this->batch_targets, true , nb_thread, implicite);
            //cout <<"| ok: pre " <<endl;
            
            
        }
        auto stop_epoch = chrono::high_resolution_clock::now();
        auto elapsed_epoch = chrono::duration_cast<chrono::milliseconds>(stop_epoch - start_epoch).count();     
        
        if( epsilon > epsilon_min)  epsilon -= epsilon_decay_value; 
        pair<float, float> res_test = test();
        /*
        cout<<"Epoch "<<iEpoch+1<<"/"<<Epoch<<": ["<<eq<<"] 100% "<<"| Error "<<epochError<<" | Acc: "<<prec/train_size<<" AT "
                                <<avg_turn/train_size<<" | val_Acc "<<res_test.first<< " | val_AT "<<res_test.second<<
                                             " | elaps: "<< elapsed_epoch/1000<<"s"<<endl;
        */
        /*for (auto s:seuils) cout<<" "<<s;
        cout<<endl;*/
        res.acc = prec/train_size;
        res.at = avg_turn/train_size;
        res.err = epochError;

        res.acc_epoch.push_back(prec/train_size);
        res.at_epoch.push_back(avg_turn/train_size);
        res.time_elapsed += elapsed_epoch;
        res.time_epoch.push_back(elapsed_epoch);
        res.epoch_convg = iEpoch+1;

        if(res.acc >= this->convg ){
            bool_convg = true;
            res.time_convg = res.time_elapsed;
            break;
        }
        update_prevParams(true);
        reset_metrics(this->nb_threads);
    }//endEpoch

    auto stop_train = chrono::high_resolution_clock::now();
    auto elapse = chrono::duration_cast<chrono::milliseconds>(stop_train - start_train).count();
    res.time_elapsed = elapse;
    
    cout<<"=== END TRAINNING << Time elapsed "<<res.time_elapsed<<"s"<<endl;

    
}



void agent::create_batch( int batch_size ){
                        
    this->mtx.lock();
    iter_batch +=1;
    this->start_create_batch = false;
    this->mtx.unlock();

    /*
    cout<<endl;
    for(int i =0; i< this->real_len_UD; i++){
        cout<<" "<<rewards[i];
    }
    
    cout<<endl;
    exit(0);
    */
        
    random_device rd;
    int d = this->real_len_UD;
    uniform_int_distribution<int> dist(0, d -1);

    if(iter_batch <= 1){

        for(int i=0; i<batch_size; i++){
         
        int index = dist(this->gen);
        this->batch_states.push_back(this->states[index]);
        this->batch_actions.push_back(this->actions[index]);
        this->batch_nextstates.push_back(this->next_states[index]);
        this->batch_dones.push_back(this->dones[index]);
        this->batch_rewards.push_back(this->rewards[index]);
        this->batch_targets.push_back(this->targets[index]);

        }
        //cout<<"thread batch off"<<endl;
        this->mtx.lock();
        this->end_create_batch = true;
        iter_batch = 0;
        this->mtx.unlock();


    }else{
        while(true){
            cout<<"\r";
            if(iter_batch < 1){
                break;
            }
        }
        cout<<"le Kon "<<endl;
    }

    
    
    
}

void agent::create_batch_V2(batch &b){

    random_device rd;
    int d = this->real_len_UD;
    uniform_int_distribution<int> dist(0, d -1);


    for(int i=0; i<b.size; i++){
        
        int index = dist(this->gen);
        b.batch_states.push_back(this->states[index]);
        b.batch_actions.push_back(this->actions[index]);
        b.batch_nextstates.push_back(this->next_states[index]);
        b.batch_dones.push_back(this->dones[index]);
        b.batch_rewards.push_back(this->rewards[index]);
        b.batch_targets.push_back(this->targets[index]);

    }
    
}


