#ifndef mymain_h
#define mymain_h

#include "MLP.hpp"
#include "Preprocessor.hpp"
#include "agent.hpp"

using namespace std;

typedef unique_ptr<MLP> pMLP;
typedef unique_ptr<agent> pAGENT;
typedef unique_ptr<Preprocessor> pPreprocessor;

#endif