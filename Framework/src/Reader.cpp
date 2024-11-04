#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include "Reader.hpp"
#include "csv.hpp"
#include <map>

void Reader::setVerbosity(int v) { verbosity = v; }

int Reader::ReverseInt(int i) {

  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

pair<int, int> Reader::read_data(string filename, vector<vector<float>> &vec) {

  
  ifstream fichier(filename);
  if (!fichier)
    {
        cout << "Erreur d'ouverture du fichier" << endl;
        return pair<int, int>(-1, -1);
    }
  
   string ligne;
    while (getline(fichier, ligne))
    {
        // Créer un vector<double> vide pour stocker les valeurs de la ligne
        vector<float> valeurs;

        // Utiliser un istringstream pour séparer les valeurs de la ligne par le caractère ','
        istringstream iss(ligne);
        string valeur;
        while (getline(iss, valeur, ','))
        {
            // Convertir la valeur en double et l'ajouter au vector<double> de la ligne
            valeurs.push_back(stod(valeur));
        }

        // Ajouter le vector<double> de la ligne au vector<vector<double>> des données
        vec.push_back(valeurs);
    }

    // Fermer le fichier
    fichier.close();
    pair<int, int> dims(vec.size(), vec[1].size());

    return dims;
  
}

bool Reader::read_data_Label(string filename, vector<float> &vec) {

   ifstream fichier(filename);
  if (!fichier)
    {
        cout << "Erreur d'ouverture du fichier" << endl;
        return false;
    }
  
     vector<float> donnees;

    // Créer un vector<vector<double>> vide pour stocker les données
    string ligne;
    while (getline(fichier, ligne))
    {
        // Créer un vector<double> vide pour stocker les valeurs de la ligne
        float valeurs;

        // Utiliser un istringstream pour séparer les valeurs de la ligne par le caractère ','
        istringstream iss(ligne);
        string valeur;
        while (getline(iss, valeur, ','))
        {
            // Convertir la valeur en double et l'ajouter au vector<double> de la ligne
            valeurs = stod(valeur);
        }

        // Ajouter le vector<double> de la ligne au vector<vector<double>> des données
        donnees.push_back(valeurs);
    }

    // Fermer le fichier
    fichier.close();

  return true;
}




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


bool Reader::clear_Symptoms(vector<string> List_symptoms, EncodeSymptoms &res){
    
    vector<vector<string>> tab_symptoms;
    vector<string> symptoms;
    vector<string> total_unique_Symptoms;
    map<string, bool> glob_unique_symptoms;
    float nb_symp = 0;
    for (string str_symptom : List_symptoms){
        
        symptoms = split(str_symptom, ",");
        // move [] to the string
        string ch = symptoms.front();
        ch = ch.substr(1,ch.size()-1);
        symptoms.erase(symptoms.begin());
        symptoms.insert(symptoms.begin(), " "+ch);
        ch = symptoms.back();
        ch = ch.substr(0,ch.size() -1);
        symptoms.pop_back();
        symptoms.push_back(ch);


        map<string,bool> dico_symptoms;
        string s_temp;
        for (auto& s : symptoms){
            s = s.substr(2,s.size()-3);        //move '' to the string
            
            s_temp = split(s, "_@_").front();
            
            if(s_temp != s){
                s= s_temp;
            }
            glob_unique_symptoms[s] = true;
            dico_symptoms[s] = true; 
        }

        vector<string> unique_Symptoms;

        for(map<string, bool>::iterator it = dico_symptoms.begin(); it != dico_symptoms.end(); it++){
            unique_Symptoms.push_back(it->first);
        }
        nb_symp +=unique_Symptoms.size();
        /*
        auto it= unique_Symptoms.begin();
        random_device rd;
        uniform_int_distribution<int> dist(0, unique_Symptoms.size()-1);
        unique_Symptoms.erase(it + dist(rd));
        it= unique_Symptoms.begin();
        uniform_int_distribution<int> dist1(0, unique_Symptoms.size()-1);
        unique_Symptoms.erase(it + dist1(rd));
     
        it= unique_Symptoms.begin();
        uniform_int_distribution<int> dist2(0, unique_Symptoms.size()-1);
        unique_Symptoms.erase(it + dist2(rd));
        */
        //unique_Symptoms.erase(it + dist(rd));

        tab_symptoms.push_back(unique_Symptoms);
    }

     for(map<string, bool>::iterator it = glob_unique_symptoms.begin(); it != glob_unique_symptoms.end(); it++){
            total_unique_Symptoms.push_back(it->first);
        }
    
    sort(total_unique_Symptoms.begin(), total_unique_Symptoms.end(), compare);
    res.tab_symptoms = tab_symptoms;
    res.unique_symptoms = total_unique_Symptoms;
    cout<<"nb symptoms "<<nb_symp/List_symptoms.size()<<endl;
    return true;
}

bool compare(const string& a, const string& b) {
    return a < b;
}

void encode_age(vector<float> &ages){
    
    for(int i=0; i<ages.size(); i++){
        if(ages[i] < 1){
            ages[i] = 0.0;
        }else if(ages[i]>=1 and ages[i]<=4){
            ages[i] = 1.0;
        }else if(ages[i]>=5 and ages[i]<=14){
            ages[i] = 2.0;
        }else if(ages[i]>= 15 and ages[i]<=29){
            ages[i] = 3.0;
        }else if(ages[i]>= 30 and ages[i]<=44){
            ages[i] = 4.0;
        }else if(ages[i]>=45 and ages[i]<=59){
            ages[i] = 5.0;
        }else if(ages[i]>=60 and ages[i]<=74){
            ages[i] = 6.0;
        }else{
            ages[i] = 7.0;
        }

    }
}

vector<float> encode_sex(vector<string> &sex){
    vector<float> code_sex;
    for(string s:sex){
        if(s=="M"){
            code_sex.push_back(1.0);
        }else{
            code_sex.push_back(0.0);
        }
    }

    return code_sex;
}

bool Reader::load_csv(string csv_file, ReturnLoadCSV &res){
    
    string columns[6] = {"Age","Differential","Sex","Pathology","Evidences","Initial"};
    
    vector<string> Symptoms;
    vector<string> pathology;
    vector<float> age;
    vector<string> sex;



    CSVReader reader(csv_file);
    
    int i;
    cout<<"------ reading data ..."<<endl;
    for (CSVRow row : reader ){
        i = 0;
        for(CSVField field: row){

            
            if(i==0){
                age.push_back(stof(field.get()));
            }else if(i==2){
                sex.push_back(field.get());
            }
            else if(i==3){
                pathology.push_back(field.get());
                
            }else if(i==4){
                Symptoms.push_back(field.get());
                
            }
            i++;
        }
    
    }

    /*
    random_device rd;
    mt19937 gen(rd());
    auto gen2 = gen;
    
    shuffle(Symptoms.begin(), Symptoms.end(), gen);
    shuffle(pathology.begin(), pathology.end(), gen2);
    */
    //cout<<"len of datatse "<<Symptoms.size()<<endl;
    vector<string> path;
    vector<string> symp;
    copy(Symptoms.begin(), Symptoms.begin() + int(Symptoms.size()*0.5),back_inserter(symp)) ;
    copy(pathology.begin(), pathology.begin() + int(pathology.size()*0.5),back_inserter(path)) ;

    set<string> unique_path(path.begin(), path.end());
    vector<string> vec_unique_pathology(unique_path.begin(), unique_path.end());
    
    //sort(vec_unique_pathology.begin(), vec_unique_pathology.end(), compare);

    encode_age(age);
    vector<float> code_sex = encode_sex(sex);
    EncodeSymptoms res_p;
    
    clear_Symptoms(symp, res_p);


    res.features = res_p.tab_symptoms;
    res.unique_symptoms= res_p.unique_symptoms;    
    res.targets = path;    
    res.unique_pathology = vec_unique_pathology;
    res.age = age;
    res.sex = code_sex;
    cout<<"len unique symptoms "<<res.unique_symptoms.size()<<endl;
    cout<<"len unique pathology "<<res.unique_pathology.size()<<endl;
     
    return true;

}


pair<int, int> Reader::One_hot_encoding(ReturnLoadCSV data, ReturnOneHotEncoding &res){
    

    vector<vector<float>> features_after(data.features.size(), vector<float>(data.unique_symptoms.size(), 0));
    vector<float> targets_after(data.targets.size(),0);
    // the key is a symptom (or pathogie) and the value is an index
    map<string, int> dico_symptoms;
    map<string, int> dico_pathology;

    //initialize map of symptoms
    int index = 0;
    for(string s:data.unique_symptoms ){
        dico_symptoms[s] = index;
        index++;
    }
    //initialize map of pathology
    index = 0;
    for(string s:data.unique_pathology){
        dico_pathology[s] = index;
        index++;
    }
    cout<<"------ one-hot-encoding of features ..."<<endl;
    //one-hot-encoding of features
    for(int i=0; i<features_after.size(); i++){
        //features_after[i][0] = data.age[i];
        //features_after[i][1] = data.sex[i];
        for(auto symp:data.features[i]){
            features_after[i][dico_symptoms[symp]] = 1.0;
        }
    }

    cout<<"------ encoding targets ..."<<endl;
    //encoding targets
    for(int i=0; i<targets_after.size(); i++){
        targets_after[i] = dico_pathology[data.targets[i]]+0.0;
    }

    res.features = features_after;
    res.targets = targets_after;
    res.unique_pathology = data.unique_pathology;
    res.unique_symptoms = data.unique_symptoms;
    //cout<<"target 0 : "<<res.targets[0]<<endl;
    pair<int, int> dims(res.features.size(), res.features[1].size());
    return dims;
}


bool Reader::Mz_load_csv(string csv_file, MzReturnLoadCSV &res){
     
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

    res.features = features;
    res.symp_exp = Encode_exp_symp;
    res.symp_imp = Encode_imp_symp;
    res.targets = pathology;
    return true;
}

bool Reader::MedlinePlus_loadCSV(string csv_file, ReturnMedlinePlus &res){
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

    //cout<<str_glob_symp[0]<<endl;
    
    for (string s : str_glob_symp){
        vector<string> vec = split(s, ",");
        vector<float> vec2;
        for( string val : vec){
            if(stof(val) > 10000){
                cout<<"vrai read sup"<<endl;
                exit(0);
            }
            vec2.push_back(stof(val));
        }

        glob_symp.push_back(vec2);

    }

    res.diseases = diseases;
    res.glob_symptoms = glob_symp;
   
    return true;
}