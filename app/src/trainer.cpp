#include "main.hpp"
 
int main(int argc, char *argv[]) {

  if (argc < 3) {
    cout << "Usage: " << argv[0] << " <data Folder> <logFile> <saveFile>";
    return 0;
  }

  string dataFolder = argv[1];
  //string csvFile = "/release_train_patients.csv";
  string csvFile = "/mz.csv";


  vector<vector<float>> train_features;
  vector<vector<float>> test_features;


  vector<float> train_targets;
  vector<float> test_targets;

  
  pMLP mlp = make_unique<MLP>(1);

  mlp->createLog(argv[2]);
  

  if(!mlp->read_data(dataFolder+csvFile,train_features,
    test_features, train_targets, test_targets ))   return 0;

  //train_features.resize(5);
  //train_targets.resize(5);
  cout<<"Train observations "<<train_features.size()<<" dim "<<train_features[1].size()<<endl;
  cout<<"Test observations "<<test_features.size()<<" dim "<<test_features[1].size()<<endl;
  
  //mlp->addLayer<Dense>(10, Activation::relu,1);
  //mlp->addLayer<Dense>(5, Activation::relu,1);

  //mlp->addLayer<Dense>(train_features[1].size(), Activation::relu);
  //mlp->addLayer<Dense>(505, Activation::relu);
  //mlp->addLayer<Dense>(20, Activation::relu);

  //mlp->addLayer<Dense>(4, Activation::softmax,1);
  
  mlp->compile(0.01, 0.4,1);
  
  //mlp->trainNetwork(train_features, train_targets, 65, 8);
  //mlp->trainClassifier(train_features,train_targets, 20, 0 );
 // mlp->validateNetwork(test_features, test_targets, 0);

  mlp->saveNetwork(argv[3]);

  cout << "All done.\n";


}