#include "MLP.hpp"
#include "env.hpp"
#include<random>
 
typedef unique_ptr<MLP> pMLP;

 
struct result{
int time_elapsed = 0;
float acc = 0;
float at = 0;
float err = 0;
vector<float> time_epoch;
int time_convg=0;
vector<float> at_epoch;
vector<float> acc_epoch;
vector<int> nb_modif;
int epoch_convg = 0;

}typedef result;



struct batch{
    int size;
    vector<vector<float>> batch_states;
    vector<vector<float>> batch_nextstates;
    vector<float> batch_rewards;
    vector<float> batch_actions;
    vector<bool> batch_dones;
    vector<float> batch_targets;
        
}typedef batch;


class agent{

    public:
        float eps=0.99;
        double l_rate = 0.001;
        double momemtum = 0.4;
        mt19937 gen;
        vector<mt19937> vec_gen;
        int batch_trainning = 640;
        float Lamda = 0.99;
        int nb_threads = 1;
        int iter_create_batch = 0;
        int N_epoch;
        bool stop_env_threads = false;
        vector<int> end_iEpoch;
        string path_res = "../Res/";
        int limit_UD;
        int real_len_UD;
        bool begin_create_batch = false;
        bool end_create_batch = false;

        bool start_create_batch = true;
        vector<float> seuils;
        vector<float> precision;
        vector<float> average_turn;
        vector<vector<float>> prec_per_Epoch;
        vector<vector<float>> avg_per_Epoch;
        vector<vector<float>> test_prec_per_Epoch;
        vector<vector<float>> test_avg_per_Epoch;
        vector<float> capture;
        vector<int> position;
        vector<int> to_move;
        int limit;
        int nb_modif = 0;
        int end;
        int begin;
        float Error_P = 0.0;
        bool UpdateP = false;
        int env_nbThreads;
        int endEnv_threads=0;
        int iter_batch = 0;
        float convg = 0.95;
        bool bool_convg = false;
        float synch_loss = 0.0;
        bool mutex_Mat = false;
        float policy_error_test = 0.0;
        mutex mtx;
        environment glob_env;
        vector<env> envs;
        vector<pMLP> mlp_models;
        vector<pMLP> classifier_models;  
        pMLP mlp = make_unique<MLP>(1);
        MLP test_test;
        pMLP classifier = make_unique<MLP>(1);
        
        vector<vector<float>> states;
        vector<vector<float>> next_states;
        vector<float> rewards;
        vector<float> avantages;
        vector<float> actions;
        vector<bool> dones;
        vector<float> targets;

        vector<vector<float>> batch_states;
        vector<vector<float>> batch_nextstates;
        vector<float> batch_rewards;
        vector<float> batch_actions;
        vector<bool> batch_dones;
        vector<float> batch_targets;
        
                
    public:
        
        void init_networks(int nb_action, int nb_diseases, bool seq, int nb_thread);
        void init_env(int nb_thread, string filedata, int limit_UD);
        float update_params(vector<vector<float>> &states,vector<vector<float>> &nextstates,
                         vector<float> &actions, vector<bool> &dones, 
                                vector<float> &rewards, vector<float> &targets,bool parallel, int nb_thread, bool implicite = false); 

        float Exp_update_params(vector<vector<float>> &states,vector<vector<float>> &nextstates,
                         vector<float> &actions, vector<bool> &dones, 
                                vector<float> &rewards, vector<float> &targets,bool parallel, int nb_thread); 

        
        void save_agent(string filename_classifier, string filename_policy);
        void load_agent(string filename_classifier, string filename_policy);
        pair<int, float> choose_diagnostic(vector<float> state, int id_thread, bool origin);
        int choose_action(vector<float> &state, float eps, bool eGreedy, int id_thread, bool origin,  int id_obs);
        void create_batch( int batch_size );
        void create_batch_V2(batch &b);
        void update_prevParams(bool seq);
        void par_trainParams(int iEpoch);
        void seq_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, result &res);
        void AdaptativePar_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread,  result &res, int envTH, bool implicite, float threshold);
        void AdaptativeImpPar_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread,  result &res, int envTH, bool implicite, float threshold, int max_thread);
        void parV1_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread,  result &res, bool implicite);
        void parSGD_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread, result &res, bool implicite);
        void par_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread, result &res,bool implicite);
        void Exp_synchronous_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread, result &res,bool implicite);
        void Imp_synchronous_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread, result &res,bool implicite, bool mutexMat);
        void Asynch_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread,  result &res, int envTH, bool implicite);
        void parV2_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread,  result &res, int envTH,bool implicite);
        void parV3_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread,  result &res, bool implicite);
        void Asynch_Adap_trainDQL(string filedata, string fileClassifier, int Epoch, int batch_size, int nb_thread,  result &res, int envTH, bool implicite);
        void Exp_synch_update(int nb_th, int th_id );
        void Imp_update( int nb_th, int th_id );
        pair<float, float> test();
        void test_par(int id_th);
        void reset_metrics(int nb_thread);
        void reset(int nb_threads);

};


struct params{
    int num_thread;
    agent *ag;
}typedef params;

