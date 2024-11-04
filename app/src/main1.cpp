#include "main.hpp"
#include "time.h"

void writeGnuFile( string fileName, string output, string data_file){
    ofstream file_gnu;
    file_gnu.open(fileName);
    file_gnu<<"Number of epochs"<<endl;
    file_gnu<<"set ylabel Accuracy\n"<<endl;
    file_gnu<<"Variation of Accuracy learning\n"<<endl;
    file_gnu<<"set terminal png\n"<<endl;
    file_gnu<<"set output "<<output<<"\n"<<endl;
    file_gnu<<"plot" <<data_file<< "using 1:2 title ' seq' with lines linecolor 'black' "<<endl;
    
}

void writeNbModif(result res, string fileName){
    ofstream fic;
    fic.open(fileName);
    for(int i = 0; i<res.nb_modif.size(); i++){
        fic<<i+1<<" "<<res.nb_modif[i]<<endl;
    }
    fic.close();
}





int main(int argc, char *argv[]){

    if(argc !=  5){
        cout<<"Erreur Nombre d'argument incorrect -> Template :"<<
        "chemin parent du dataset>> <<nb-threads>> <<accuracy convergence>> <<method>>"<<endl;
        exit(-1);
    }

    
    string dataFolder = argv[1];
    int nb_threads = atoi(argv[2]);
    float convg = atof(argv[3]);
    int method = atoi(argv[4]);

    int nb_env = nb_threads / 2;
    int nb_env1 = max(nb_threads/ 4, 1);

    //string csvFile = "/release_train_patients.csv";
    //string csvFile = dataFolder + "/release_validate_patients.csv";
    //string classifierModel = dataFolder + "/network_C2";
    ofstream fic;
    
    time_t maintenant = time(NULL);
    tm *ltm = localtime(&maintenant); 
    
    string fileName = "../SP/speedup.csv";
    string fModif_v2 = "../SP/Mo/V2_nb_modif_";
    string fModif_v2_i = "../SP/Mo/V2_i_nb_modif_";
    string fModif_v2_i1 = "../SP/Mo/V2_i1_nb_modif_";
    string fModif_v21 = "../SP/Mo/V21_nb_modif_";
    string fModif_A6 = "../SP/Mo/A6_nb_modif_";
    string fModif_Ai6 = "../SP/Mo/Ai6_nb_modif_";
    string fModif_A7 = "../SP/Mo/A7_nb_modif_";
    string fModif_Ai7 = "../SP/Mo/Ai7_nb_modif_";
    string fModif_A8 = "../SP/Mo/A8_nb_modif_";
    string fModif_Ai8 = "../SP/Mo/Ai8_nb_modif_";
    string fModif_AsynAdap = "../SP/Mo/AsynAdap_nb_modif_";



    string  day = to_string(ltm->tm_year + 1900) + "_" + to_string(ltm->tm_mon +1)
                         + "_" + to_string(ltm->tm_mday) + "_"+to_string(ltm->tm_hour)+"h" + to_string(ltm->tm_min) +"min.txt" ;
    
    //fileName = fileName + day; 
    fic.open(fileName, ofstream::app);
    

    string csvFile = dataFolder + "/MedlinePlus20.csv";
    string classifierModel = dataFolder + "/network_1";


    pAGENT ag = make_unique<agent>();
    ag->convg = convg;
    //pGAGENT ag2 = make_unique<grid_agent>();
    result res_seq, res_par, res_par_v2, res_par_v3,
             res_par_v1,res_par_v1_i, res_par_i, res_par_v2_i, res_par_v2_i1, res_par_v21,
              res_adap7, res_adap6, res_adap8, res_adapi7, res_adapi6, res_adapi8, res_Exp_sych,
               res_Imp_synch, res_Imp_synch_muMat, res_Async_seq, res_async_adap;

    float SpeedUp_v1, SpeedUp_v1_i;
    float SpeedUp_sgd;
    float SpeedUp_v3;
    float SpeedUp_par, SpeedUp_par_i;
    float SpeedUp_v2,SpeedUp_v21,SpeedUp_v2_i,SpeedUp_v2_i1 ;
    

    if(method == -2){
        ag->init_networks(110, 20, false, nb_threads);
        ag->save_agent("init_params_classifier", "init_params_policy");
        return 0;
    }else{
        ag->load_agent("init_params_classifier", "init_params_policy");

    }

    
    
    if(method == 0){
        ag->reset(nb_threads);
        ag->seq_trainDQL(csvFile, classifierModel, 40, 18, res_seq);
        fic<<"\n"<<method<<","<<res_seq.acc<<","<<res_seq.at<<","<<res_seq.time_elapsed<<","<<nb_threads<<","<<res_seq.epoch_convg<<flush;

    }
    else if(method == 1){
        ag->reset(nb_threads);
        ag->parV1_trainDQL(csvFile, classifierModel, 40, 18, nb_threads, res_par_v1, false);
        SpeedUp_v1 = float(res_seq.time_elapsed)/res_par_v1.time_elapsed;
        fic<<"\n"<<method<<","<<res_par_v1.acc<<","<<res_par_v1.at<<","<<res_par_v1.time_elapsed<<","<<nb_threads<<","<<res_par_v1.epoch_convg<<flush;
    }
    else if(method == 2){
        ag->reset(nb_threads);
        ag->par_trainDQL(csvFile, classifierModel, 40, 18, nb_threads, res_par, false);
        SpeedUp_par = float(res_seq.time_elapsed)/res_par.time_elapsed;
        fic<<"\n"<<method<<","<<res_par.acc<<","<<res_par.at<<","<<res_par.time_elapsed<<","<<nb_threads<<","<<res_par.epoch_convg<<flush;
    }
    else if(method == 3){
        ag->reset(nb_threads);
        ag->par_trainDQL(csvFile, classifierModel, 40, 18, nb_threads, res_par_i, true);
        SpeedUp_par_i = float(res_seq.time_elapsed)/res_par_i.time_elapsed;
        fic<<"\n"<<method<<","<<res_par_i.acc<<","<<res_par_i.at<<","<<res_par_i.time_elapsed<<","<<nb_threads<<","<<res_par_i.epoch_convg<<flush;
    
    }
  
    else if(method == 4){
        ag->reset(nb_threads);
        ag->parV2_trainDQL(csvFile, classifierModel, 400, 18, nb_threads, res_par_v2, nb_env, false);
        SpeedUp_v2 = float(res_seq.time_elapsed)/res_par_v2.time_elapsed;
        fic<<"\n"<<method<<","<<res_par_v2.acc <<","<<res_par_v2.at<<","<<res_par_v2.time_elapsed<<","<<nb_threads<<","<<res_par_v2.epoch_convg<<flush;

    }
    else if(method == 5){
        ag->reset(nb_threads);
        ag->parV2_trainDQL(csvFile, classifierModel, 400, 18, nb_threads, res_par_v21, nb_env1, false);
        SpeedUp_v21 = float(res_seq.time_elapsed)/res_par_v21.time_elapsed;
        fic<<"\n"<<method<<","<<res_par_v21.acc <<","<<res_par_v21.at<<","<<res_par_v21.time_elapsed<<","<<nb_threads<<","<<res_par_v21.epoch_convg<<flush;

    }
    else if(method == 6){
        ag->reset(nb_threads);
        ag->parV2_trainDQL(csvFile, classifierModel, 400, 18, nb_threads, res_par_v2_i, nb_env, true);
        SpeedUp_v2_i = float(res_seq.time_elapsed)/res_par_v2_i.time_elapsed;
        fic<<"\n"<<method<<","<<res_par_v2_i.acc <<","<<res_par_v2_i.at<<","<<res_par_v2_i.time_elapsed<<","<<nb_threads<<","<<res_par_v2_i.epoch_convg<<flush;

    }
    else if(method == 7){
        ag->reset(nb_threads);
        ag->parV2_trainDQL(csvFile, classifierModel, 400, 18, nb_threads, res_par_v2_i1, nb_env1, true);
        SpeedUp_v2_i1 = float(res_seq.time_elapsed)/res_par_v2_i1.time_elapsed;
        fic<<"\n"<<method<<","<<res_par_v2_i1.acc <<","<<res_par_v2_i1.at<<","<<res_par_v2_i1.time_elapsed<<","<<nb_threads<<","<<res_par_v2_i1.epoch_convg<<flush;
    
    }
    else if(method == 8){
        ag->reset(nb_threads);
        ag->AdaptativePar_trainDQL(csvFile, classifierModel, 400, 18, nb_threads, res_adap6, nb_env, false, 0.6 );
        float SpeedUp_A6 = float(res_seq.time_elapsed)/res_adap6.time_elapsed;
        fic<<"\n"<<method<<","<<res_adap6.acc <<","<<res_adap6.at<<","<<res_adap6.time_elapsed<<","<<nb_threads<<","<<res_adap6.epoch_convg<<flush;

    }
    
    else if(method == 9){
        ag->reset(nb_threads);
        ag->AdaptativePar_trainDQL(csvFile, classifierModel, 400, 18,nb_threads, res_adapi6, nb_env, true, 0.6 );
        float SpeedUp_Ai6 = float(res_seq.time_elapsed)/res_adapi6.time_elapsed;
        fic<<"\n"<<method<<","<<res_adapi6.acc <<","<<res_adapi6.at<<","<<res_adapi6.time_elapsed<<","<<nb_threads<<","<<res_adapi6.epoch_convg<<flush;

    }

    else if(method == 10){
        ag->reset(nb_threads);
        ag->AdaptativePar_trainDQL(csvFile, classifierModel, 400, 18,nb_threads, res_adap7, nb_env, false, 0.7 );
        float SpeedUp_A7 = float(res_seq.time_elapsed)/res_adap7.time_elapsed;
        fic<<"\n"<<method<<","<<res_adap7.acc <<","<<res_adap7.at<<","<<res_adap7.time_elapsed<<","<<nb_threads<<","<<res_adap7.epoch_convg<<flush;

    }
    else if(method == 11){
        ag->reset(nb_threads);
        ag->AdaptativePar_trainDQL(csvFile, classifierModel, 400, 18,nb_threads, res_adapi7, nb_env, true, 0.7 );
        float SpeedUp_Ai7 = float(res_seq.time_elapsed)/res_adapi7.time_elapsed;
        fic<<"\n"<<method<<","<<res_adapi7.acc <<","<<res_adapi7.at<<","<<res_adapi7.time_elapsed<<","<<nb_threads<<","<<res_adapi7.epoch_convg<<flush;

    }
    else if(method == 12){
        ag->reset(nb_threads);
        ag->AdaptativePar_trainDQL(csvFile, classifierModel, 400, 18,nb_threads, res_adap8, nb_env, false, 0.8 );
        float SpeedUp_A8 = float(res_seq.time_elapsed)/res_adap8.time_elapsed;
        fic<<"\n"<<method<<","<<res_adap8.acc <<","<<res_adap8.at<<","<<res_adap8.time_elapsed<<","<<nb_threads<<","<<res_adap8.epoch_convg<<flush; 
    
    }
    else if(method == 13){
        ag->reset(nb_threads);
        ag->AdaptativePar_trainDQL(csvFile, classifierModel, 400, 18,nb_threads, res_adapi8, nb_env, true, 0.8 );
        fic<<"\n"<<method<<","<<res_adapi8.acc <<","<<res_adapi8.at<<","<<res_adapi8.time_elapsed<<","<<nb_threads<<","<<res_adapi8.epoch_convg<<flush;
        writeNbModif(res_adapi8, fModif_Ai8 + day);
    
    }


    else if(method == 14){
        ag->reset(nb_threads);
        ag->Imp_synchronous_trainDQL(csvFile, classifierModel, 400, 18, nb_threads, res_Imp_synch, true, false);
        //float SpeedUp_Imp_synch = float(res_seq.time_elapsed)/res_Imp_synch.time_elapsed;
        //cout<<"ap train"<<endl;
        fic<<"\n"<<method<<","<<res_Imp_synch.acc <<","<<res_Imp_synch.at<<","<<res_Imp_synch.time_elapsed<<","<<nb_threads<<","<<res_Imp_synch.epoch_convg<<flush;

    }

    else if(method == 15){
        ag->reset(nb_threads);
        ag->Imp_synchronous_trainDQL(csvFile, classifierModel, 400, 18, nb_threads, res_Imp_synch_muMat, true, true);
        float SpeedUp_Imp_synch_muMat = float(res_seq.time_elapsed)/res_Imp_synch_muMat.time_elapsed;
        fic<<"\n"<<method<<","<<res_Imp_synch_muMat.acc <<","<<res_Imp_synch_muMat.at<<","<<res_Imp_synch_muMat.time_elapsed<<","<<nb_threads<<","<<res_Imp_synch_muMat.epoch_convg<<flush;

    }

    else if(method == 16){
        ag->reset(nb_threads);
        ag->Asynch_trainDQL(csvFile, classifierModel, 400, 18, nb_threads, res_Async_seq, nb_threads - 1, false);
        //float SpeedUp_Imp_synch_muMat = float(res_seq.time_elapsed)/res_Imp_synch_muMat.time_elapsed;
        fic<<"\n"<<method<<","<<res_Async_seq.acc <<","<<res_Async_seq.at<<","<<res_Async_seq.time_elapsed<<","<<nb_threads<<","<<res_Async_seq.epoch_convg<<flush;

    }


    else if(method == 17){
        ag->reset(nb_threads);
        ag->AdaptativeImpPar_trainDQL(csvFile, classifierModel, 400, 18, nb_threads, res_adap8, nb_env, false, 0.6 , nb_threads);
        float SpeedUp_A8 = float(res_seq.time_elapsed)/res_adap8.time_elapsed;
        fic<<"\n"<<method<<","<<res_adap8.acc <<","<<res_adap8.at<<","<<res_adap8.time_elapsed<<","<<nb_threads<<","<<res_adap8.epoch_convg<<flush; 
        writeNbModif(res_adap8, fModif_A8 + day);

    }
    else if(method == 18){
        ag->reset(nb_threads);
        ag->AdaptativeImpPar_trainDQL(csvFile, classifierModel, 400, 18,nb_threads, res_adapi8, nb_env, true, 0.6, nb_threads );
        float SpeedUp_Ai8 = float(res_seq.time_elapsed)/res_adapi8.time_elapsed;
        fic<<"\n"<<method<<","<<res_adapi8.acc <<","<<res_adapi8.at<<","<<res_adapi8.time_elapsed<<","<<nb_threads<<","<<res_adapi8.epoch_convg<<flush;
        writeNbModif(res_adapi8, fModif_Ai8 + day);
    
    }

     else if(method == 19){
        ag->reset(nb_threads);
        ag->AdaptativeImpPar_trainDQL(csvFile, classifierModel, 400, 18,nb_threads, res_adap8, nb_env, false, 0.7, nb_threads );
        fic<<"\n"<<method<<","<<res_adap8.acc <<","<<res_adap8.at<<","<<res_adap8.time_elapsed<<","<<nb_threads<<","<<res_adap8.epoch_convg<<flush; 
        writeNbModif(res_adap8, fModif_A8 + day);

    }
    else if(method == 20){
        ag->reset(nb_threads);
        ag->AdaptativeImpPar_trainDQL(csvFile, classifierModel, 400, 18,nb_threads, res_adapi8, nb_env, true, 0.7, nb_threads );
        float SpeedUp_Ai8 = float(res_seq.time_elapsed)/res_adapi8.time_elapsed;
        fic<<"\n"<<method<<","<<res_adapi8.acc <<","<<res_adapi8.at<<","<<res_adapi8.time_elapsed<<","<<nb_threads<<","<<res_adapi8.epoch_convg<<flush;
        writeNbModif(res_adapi8, fModif_Ai8 + day);
    
    }

    else if(method == 21){
        ag->reset(nb_threads);
        ag->AdaptativeImpPar_trainDQL(csvFile, classifierModel, 400, 18, nb_threads, res_adap8, nb_env, false, 0.6 , 4);
        float SpeedUp_A8 = float(res_seq.time_elapsed)/res_adap8.time_elapsed;
        fic<<"\n"<<method<<","<<res_adap8.acc <<","<<res_adap8.at<<","<<res_adap8.time_elapsed<<","<<nb_threads<<","<<res_adap8.epoch_convg<<flush; 
    
    }
    else if(method == 22){
        ag->reset(nb_threads);
        ag->AdaptativeImpPar_trainDQL(csvFile, classifierModel, 400, 18,nb_threads, res_adapi8, nb_env, true, 0.6, 4 );
        float SpeedUp_Ai8 = float(res_seq.time_elapsed)/res_adapi8.time_elapsed;
        fic<<"\n"<<method<<","<<res_adapi8.acc <<","<<res_adapi8.at<<","<<res_adapi8.time_elapsed<<","<<nb_threads<<","<<res_adapi8.epoch_convg<<flush;
        writeNbModif(res_adapi8, fModif_Ai8 + day);
    
    }

     else if(method == 23){
        ag->reset(nb_threads);
        ag->AdaptativeImpPar_trainDQL(csvFile, classifierModel, 400, 18,nb_threads, res_adap8, nb_env, false, 0.7, 4 );
        fic<<"\n"<<method<<","<<res_adap8.acc <<","<<res_adap8.at<<","<<res_adap8.time_elapsed<<","<<nb_threads<<","<<res_adap8.epoch_convg<<flush; 
        writeNbModif(res_adap8, fModif_A8 + day);

    }
    else if(method == 24){
        ag->reset(nb_threads);
        ag->AdaptativeImpPar_trainDQL(csvFile, classifierModel, 400, 18,nb_threads, res_adapi8, nb_env, true, 0.7, 4 );
        float SpeedUp_Ai8 = float(res_seq.time_elapsed)/res_adapi8.time_elapsed;
        fic<<"\n"<<method<<","<<res_adapi8.acc <<","<<res_adapi8.at<<","<<res_adapi8.time_elapsed<<","<<nb_threads<<","<<res_adapi8.epoch_convg<<flush;
        writeNbModif(res_adapi8, fModif_Ai8 + day);
    
    }

    else if(method == 25){
        ag->reset(nb_threads);
        ag->Asynch_Adap_trainDQL(csvFile, classifierModel, 400, 18, nb_threads, res_async_adap, nb_env, false);
        fic<<"\n"<<method<<","<<res_async_adap.acc <<","<<res_async_adap.at<<","<<res_async_adap.time_elapsed<<","<<nb_threads<<","<<res_async_adap.epoch_convg<<flush;
        writeNbModif(res_async_adap, fModif_AsynAdap + day);

    }


    else if(method == 26){
        ag->reset(nb_threads);
        ag->Asynch_Adap_trainDQL(csvFile, classifierModel, 400, 18, nb_threads, res_async_adap, nb_env1, false);
        fic<<"\n"<<method<<","<<res_async_adap.acc <<","<<res_async_adap.at<<","<<res_async_adap.time_elapsed<<","<<nb_threads<<","<<res_async_adap.epoch_convg<<flush;
        writeNbModif(res_async_adap, fModif_AsynAdap + day);

    }

    else if(method == 27){
        ag->reset(nb_threads);
        ag->Asynch_Adap_trainDQL(csvFile, classifierModel, 400, 18, nb_threads, res_async_adap, nb_env, true);
        fic<<"\n"<<method<<","<<res_async_adap.acc <<","<<res_async_adap.at<<","<<res_async_adap.time_elapsed<<","<<nb_threads<<","<<res_async_adap.epoch_convg<<flush;
        writeNbModif(res_async_adap, fModif_AsynAdap + day);

    }

    else if(method == 28){
        ag->reset(nb_threads);
        ag->Asynch_Adap_trainDQL(csvFile, classifierModel, 400, 18, nb_threads, res_async_adap, nb_env1, true);
        fic<<"\n"<<method<<","<<res_async_adap.acc <<","<<res_async_adap.at<<","<<res_async_adap.time_elapsed<<","<<nb_threads<<","<<res_async_adap.epoch_convg<<flush;
        writeNbModif(res_async_adap, fModif_AsynAdap + day);

    }

    
    else if(method == 57){
        ag->reset(nb_threads);
        ag->Exp_synchronous_trainDQL(csvFile, classifierModel, 40, 18, nb_threads, res_Exp_sych, false);
        float SpeedUp_Exp_sych = float(res_seq.time_elapsed)/res_Exp_sych.time_elapsed;
        fic<<"\n"<<method<<","<<res_Exp_sych.acc <<","<<res_Exp_sych.at<<","<<res_Exp_sych.time_elapsed<<","<<nb_threads<<","<<res_Exp_sych.epoch_convg<<flush;

    }


    
    else{
        cout<<"Erreur methode inconnue :"<<endl;
    }
   
    
    
    fic.close();


    writeNbModif(res_par_v2, fModif_v2 + day);
    writeNbModif(res_par_v2_i, fModif_v2_i + day);
    writeNbModif(res_par_v2_i1, fModif_v2_i1 + day);
    writeNbModif(res_par_v21, fModif_v21 + day);
    writeNbModif(res_adap6, fModif_A6 + day);
    writeNbModif(res_adapi6, fModif_Ai6 + day);
    writeNbModif(res_adap7, fModif_A7 + day);
    writeNbModif(res_adapi7, fModif_Ai7 + day);
    
   
   
    //train()
    return 0;
}
