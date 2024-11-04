#include "env.hpp"
#include <random>

/**/ 

void environment::simulation_patient(){
    int nb = 100;   
    //int nb = 50;
    for( int id= 0; id< data.diseases.size(); id++){
        //if(id == 1) exit(0);   
        for(int i =0; i<nb; i++){
            random_device rd;
            mt19937 gen(1);
            int len_symp = data.glob_symptoms[id].size();
            shuffle(data.glob_symptoms[id].begin(),data.glob_symptoms[id].begin(), gen );
            
            vector<int> vec_exp;
            vector<int> vec_imp;
            poisson_distribution<int> dist8(8);
            poisson_distribution<int> dist2(2);
            int n = dist8(gen);

            if(n > len_symp)    n = len_symp;
            if( n < len_symp/2) n = len_symp/2;

            uniform_int_distribution<int> dist(0, n-1);
            
            int nb_exp = dist2(gen) +1;
            if(nb_exp > len_symp-1) nb_exp = len_symp/2;
            
            for(int k = 0; k< nb_exp; k++){
                if(abs(data.glob_symptoms[id][k]) >= 883){
                    cout<<"except symp "<<data.glob_symptoms[id][k]<<endl;
                    exit(0);
                }
                vec_exp.push_back(data.glob_symptoms[id][k]);

            }  
            for(int k = nb_exp; k< n; k++){
                if(abs(data.glob_symptoms[id][k]) >= 883){
                    cout<<"except symp "<<data.glob_symptoms[id][k]<<endl;
                    exit(0);
                }
               vec_imp.push_back(data.glob_symptoms[id][k]);
 
            }  
            data.targets.push_back(data.diseases[id]);
            data.symp_exp.push_back(vec_exp);
            data.symp_imp.push_back(vec_imp);
          /*cout<<"\nglob"<<endl;
            for (auto s:data.glob_symptoms[id]) cout<<" "<<s;
            cout<<"\nexp"<<endl;
            for (auto s:vec_exp) cout<<" "<<s;
            cout<<"\nimp"<<endl;
            for (auto s:vec_imp) cout<<" "<<s;
            */
        }
    }

    random_device rd;
    mt19937 gen;
    auto gen2 = gen;
    auto gen3 = gen;
    
    shuffle(data.targets.begin(), data.targets.end(), gen);
    shuffle(data.symp_exp.begin(), data.symp_exp.end(), gen2);    
    shuffle(data.symp_imp.begin(), data.symp_imp.end(), gen3);    

}




environment::environment(){}

void environment::init_environment(string csv_file){
    pReader reader = make_unique<Reader>();
    reader->MedlinePlus_loadCSV(csv_file, this->data);
    //this->dims.first = this->data.features.size();
    //this->dims.second = this->data.features[0].size();
    
    this->dims.first = 2000;
    this->dims.second = 110;
    this->nb_diseases = 20;
    this->end_train = 1800;
    /*
    this->dims.first = 1000;
    this->dims.second = 61;
    this->nb_diseases = 10;
    this->end_train = 800;
    */
    /*
    this->dims.first = 10000;
    this->dims.second = 546;
    this->nb_diseases = 200;
    this->end_train = 8000;
    */
    simulation_patient();
}
/*
void environment::init_environment(string csv_file){
    string csv = csv_file;
    environment(csv);
}
*/

/* 
environment::environment(string csv_file){
   // cout<<"initial Environment"<<endl;
    this->idx = 0;
    pReader reader = make_unique<Reader>();
    reader->load_csv(csv_file, this->data);
    this->dims.first = this->data.features.size();
    this->dims.second = this->data.unique_symptoms.size();
    
    for(int i=0; i<this->data.unique_pathology.size(); i++){
        this->diseases_idx[this->data.unique_pathology[i]] = i;
    }

    for(int i=0; i<this->data.unique_symptoms.size(); i++){
        this->symptoms_idx[this->data.unique_symptoms[i]] = i;
    }

}
*/

env::env(environment &envi){
    this->Envi = envi;
}

void env::initial_state(int id, vector<float> &init_state, vector<float> &t_symp, int &t_disease, vector<int> &all_symp){

    //if(this->idx >= Envi.dims.first)  this->idx = 0;
    count_sympt = 0;
    this->done = false;
    t_disease = Envi.data.targets[id];
    
    for(auto symp:Envi.data.symp_exp[id]){
        all_symp.push_back(symp);
        //all_symp.push_back(this->symptoms_idx[symp]);
        
            
        init_state[symp] = 1;
        
         
        //t_symp[i] = 1;
        //init_state[j] = 1;
        //t_symp[j] = 1;

    }
    
    for(auto symp:Envi.data.symp_imp[id]){
        all_symp.push_back(symp);
    }
    //cout<<"initial state"<<endl;

}


float env::reward_func(vector<float> state, int action, pair<int, float> diag, int t_disease, int turn, vector<int> all_symp){
    float rp = -1;
    
    if( state[action] == Envi.ABS_VAL){
        auto iter = find(all_symp.begin(), all_symp.end(), action);
        
        if(iter != all_symp.end()){
            rp += 1.7;
        }
        else{
            rp += 0.7;
        }
    }
    
    if(diag.first == t_disease){
        //this->done = true;
        rp += 1;
    }
    
    if(turn == Envi.MAX_TURN -1 and diag.first != t_disease){
        this->done = true;
        rp -= 1;   
    }   

    if(isnan(rp)){
        cout<<"NAN Error reward_func rp : "<<rp<<endl;
        exit(0);
    }
    return rp;
}


void env::step(vector<float> &state, vector<float> &t_symp, vector<int> all_symp, float &reward,  int action,
                                     int turn, int t_disease, pair<int, float> diag ){
    //cout<<"step state"<<endl;

    reward = this->reward_func(state, action, diag, t_disease, turn, all_symp);
    if(turn == Envi.MAX_TURN-1){
        done = true;
    }else if(t_disease == diag.first){
        done = true;
    }

    auto iter = find(all_symp.begin(), all_symp.end(), action);
    if(iter != all_symp.end()){
         state[action] = Envi.PRES_VAL;
         t_symp[action] = Envi.PRES_VAL;
    }   
    else    state[action] = Envi.ABS_VAL;


}

