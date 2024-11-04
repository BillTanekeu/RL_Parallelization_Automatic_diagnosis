#include "Reader.hpp"
#include <map>

  
 
typedef unique_ptr<Reader> pReader;

class environment{

    public:
        environment();
        int NONE_VAL = 0;
        int PRES_VAL = 1;
        int ABS_VAL = -1;
        int MAX_TURN = 20;
        int nb_diseases;
        vector<int> cost;
        int end_train, start_test;
        ReturnMedlinePlus data;
        map<string, int> diseases_idx;
        map<string, int> symptoms_idx;
        pair<int, int> dims;
        void init_environment(string csv_file);
        //environment(string csv_file);
        void simulation_patient();
        
};


class env : public environment{
    public :
        environment Envi;
        env(environment &envi);
        int idx;
        float H_s0, H_nextState;
        bool done;
        int count_sympt;    

        void initial_state(int id, vector<float> &init_state, vector<float> &t_symp, int &t_disease, vector<int> &all_symp);
        float reward_func(vector<float> state, int action, pair<int, float> diag, int t_disease, int turn, vector<int> all_symp); 
        void step(vector<float> &state, vector<float> &t_symp, vector<int> all_symp, float &reward,
                                int action, int turn, int t_disease, pair<int, float> diag );

};