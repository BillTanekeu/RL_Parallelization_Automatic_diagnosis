#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <sstream>
#include "csv.hpp"
#include <map>
#include<ctime>
#include<string.h>


using namespace std;
using namespace csv;

struct MzReturnLoadCSV{
  vector<vector<float>> features;
  vector<float> symp_exp;
  vector<float> symp_imp;
  vector<float> targets;
}typedef MzReturnLoadCSV;

struct ReturnMedlinePlus{
  vector<vector<float>> glob_symptoms;
  vector<float> diseases;
}typedef ReturnMedlinePlus;

vector<string> split(string ch, string sep){
    
    vector<string> res;
    size_t pos = 0;
    string token;
    
    while ((pos = ch.find(sep)) != std::string::npos) {
        token = (ch.substr(0, pos));
        if(token != "") res.push_back(token);
        ch.erase(0, pos + sep.length());
    }

    if(ch != "") res.push_back(ch);
    return res;
}



bool Mz_load_csv(string csv_file, MzReturnLoadCSV &res){
     
    string columns[3] = {"disease_tag","exp_symp","imp_symp"};
    
    
    vector<string> exp_symp;
    vector<string> imp_symp;    
    vector<vector<int>> Encode_exp_symp;
    vector<vector<int>> Encode_imp_symp;    
    
    vector<float> pathology;


    CSVReader reader(csv_file);
    
    int i;
    cout<<"------ reading data MZ..."<<endl;
    for (CSVRow row : reader ){
        i = 0;
        for(CSVField field: row){
            
            if(i==0){
                pathology.push_back(stof(field.get()));
            }else if(i==1){
                exp_symp.push_back(field.get());
            }
            else if(i==2){
                imp_symp.push_back(field.get());
            }
            i++;
        }

    }

    cout<<"len symp "<< exp_symp.size()<<endl; 
    vector<vector<float>> features(exp_symp.size(), vector<float>(66, 0));

    for (int i = 0; i < exp_symp.size(); i++){
        exp_symp[i].erase(0,1);
        exp_symp[i].erase(exp_symp[i].length()-1,1);
        imp_symp[i].erase(0,1);
        imp_symp[i].erase(imp_symp[i].length()-1,1);
        
    }

    for ( int i=0; i< exp_symp.size(); i++){
        vector<string> vec1 = split(exp_symp[i], ",");
        vector<string> vec2 = split(imp_symp[i], ",");
        vector<int> temp1;
        vector<int> temp2;
    
        for(int j=0; j<vec1.size(); j++){
            if(vec1[j] != ""){
                temp1.push_back(stoi(vec1[j]));
                features[i][stoi(vec1[j])] = 1;
            }
            
        }

        for(int j=0; j<vec2.size(); j++){
            if(vec2[j] != ""){
                temp2.push_back(stoi(vec2[j]));
                features[i][stoi(vec2[j])] = 1;
            }
            
        }

        Encode_exp_symp.push_back(temp1);
        Encode_imp_symp.push_back(temp2);

    }
    cout<<" len "<<pathology.size()<<endl;
    for (auto c : pathology)    cout<<" "<<c<<endl;
    //for( auto c:Encode_exp_symp[2]) cout<<" "<<c<<endl;
    //for( auto c:Encode_imp_symp[2]) cout<<" "<<c<<endl;
    //for( auto c:features[2]) cout<<" "<<c<<endl;
    return true;
}



bool MedlinePlus_loadCSV(string csv_file, ReturnMedlinePlus &res){
    vector<float> diseases;
    vector<vector<float>> glob_symp;
    vector<string> str_glob_symp;
    int i;
    CSVReader reader(csv_file);

    cout<<"------ reading data MedlinePlus..."<<endl;
    for (CSVRow row : reader ){
        i = 0;
        for(CSVField field: row){
            
            if(i==0){
                diseases.push_back(stof(field.get()));
            }else if(i==1){
                string temp = field.get();
                //suppression des crochets 
                temp.erase(temp.size()-1, 1);
                temp.erase(0,1);
                str_glob_symp.push_back(temp);
            }
            i++;
        }

    }

    cout<<str_glob_symp[0]<<endl;
    
    for (string s : str_glob_symp){
        vector<string> vec = split(s, ",");
        vector<float> vec2;
        for( string val : vec){
            vec2.push_back(stof(val));
        }

        glob_symp.push_back(vec2);

    }

    res.diseases = diseases;
    res.glob_symptoms = glob_symp;
    for (float s : glob_symp[0]) cout<<" "<<s; 

    return true;
}


mt19937 gen;


void run_thread(int t)  {
    uniform_int_distribution<int> dist2(1, 20);
      for(int j = 0; j < 5; j++){
            cout<<" "<<dist2(gen);
    }
    cout<<endl<<"End thread"<<endl;
    
}

int main() {
    /*
    time_t maintenant = time(NULL);
    tm *ltm = localtime(&maintenant);

    ofstream fic;
    string fileName = "resultats_TANEKEU_"; 
    fileName = fileName + to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h.txt" ;
    cout<<fileName<<" "<< ltm->tm_mday<<endl;

    fileName = "result.csv";
    fic.open(fileName, ofstream::app);
    fic<<"bonjour,"<<0<<","<<1<<endl;
    fic<<"bonjour,"<<2<<","<<4<<endl;
    fic<<"bonjour,"<<1<<","<<2<<endl;
    
    fic.close();
    */

   /* 
    random_device rd;
    //mt19937 gen(0);
    mt19937 gen2;
    uniform_int_distribution<int> dist(1, 20);
    */
   
    /*
    vector<mt19937> G;

    for(int id_g = 0; id_g < 10; id_g++){
        G.push_back(mt19937(id_g));
    }

    
    for( int i=0; i<10; i++){
        for(int j = 0; j < 5; j++){
            cout<<" "<<dist(G[i]);
        }
        cout<<"\n";
        
    }
    */

   int t = 32;
    /*
    if(p.ag->env_nbThreads == 1 ){
        id_th = p.ag->nb_threads -p.ag->env_nbThreads;

    }else{
        id_th = p.num_thread + p.ag->env_nbThreads;
    }*/

    int env , env4, nb_thread, nb_thread2;
    int env_nb;
   for (int i = 2; i<33; i++){
        env = i /2 ;
        env4 = max(i/4, 1);
        
        if(i % 2 != 0){
            nb_thread = i - env4;
            nb_thread2 = i - env;

            cout<<" threads "<<i<<" env "<<env<<" threadUp2 "<<nb_thread2<<" total "<<env+nb_thread2 << " | env4 "<<env4 <<" threadUp4 "<<nb_thread<<" total "<<env4+nb_thread<<endl;   
        }
   }

    return 0;
}
