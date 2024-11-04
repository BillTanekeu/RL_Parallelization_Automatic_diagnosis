#include "main.hpp"

int main(int argc, char *argv[]) {

  if (argc < 3) {
    cout << "Usage: " << argv[0] << " <data Folder> <logFile> <loadFile>";
    return 0;
  }

  string dataFolder = argv[1];
  //string csvFile = "/release_train_patients.csv";
  string csvFile = "/release_validate_patients.csv";


  vector<vector<float>> train_features;
  vector<vector<float>> test_features;
  vector<float> train_targets;
  vector<float> test_targets;


  pMLP mlp = make_unique<MLP>(1);

  mlp->createLog(argv[2]);
  
  mlp->loadNetwork(argv[3]);
//  mlp->loadNetwork(dataFolder+"/"+argv[3]);

  if(!mlp->read_data(dataFolder+csvFile,train_features,
    test_features, train_targets, test_targets ))   return 0;

  //mlp->validateNetwork(test_features, test_targets, 0);

  cout << "All done.\n";
}