#ifndef reader_h
#define reader_h

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include "csv.hpp"

using namespace std;
using namespace csv;



struct EncodeSymptoms{
    vector<vector<string>> tab_symptoms;
    vector<string> unique_symptoms;

}typedef EncodeSymptoms;


struct ReturnLoadCSV{
    vector<vector<string>> features;
    vector<string> targets;
    vector<string> unique_symptoms;
    vector<string> unique_pathology;
    vector<float> age;
    vector<float> sex;
}typedef ReturnLoadCSV;

struct MzReturnLoadCSV{
  vector<vector<float>> features;
  vector<vector<int>> symp_exp;
  vector<vector<int>> symp_imp;
  vector<float> targets;
}typedef MzReturnLoadCSV;


struct ReturnOneHotEncoding{
    vector<vector<float>> features;
    vector<float> targets;
    vector<string> unique_symptoms;
    vector<string> unique_pathology;
}typedef ReturnOneHotEncoding;

struct ReturnMedlinePlus{
  vector<vector<float>> glob_symptoms;
  vector<float> targets;
  vector<vector<int>> symp_exp;
  vector<vector<int>> symp_imp;
  vector<float> diseases;
}typedef ReturnMedlinePlus;

vector<string> split(string ch, string sep);
EncodeSymptoms clear_Symptoms(vector<string> List_symptoms);
bool compare(const string& a, const string& b);

class Reader {

private:
  int verbosity;

public:
  void setVerbosity(int v);
  pair<int, int> read_data(string filename, vector<vector<float>> &vec);
  bool read_data_Label(string filename, vector<float> &vec);
  int ReverseInt(int i);
  bool load_csv(string csv_file, ReturnLoadCSV &res);
  bool Mz_load_csv(string csv_file, MzReturnLoadCSV &res);
  bool MedlinePlus_loadCSV(string csv_file, ReturnMedlinePlus &res);
  bool clear_Symptoms(vector<string> List_symptoms, EncodeSymptoms &res);
  pair<int, int> One_hot_encoding(ReturnLoadCSV data, ReturnOneHotEncoding &res);
  
};

#endif