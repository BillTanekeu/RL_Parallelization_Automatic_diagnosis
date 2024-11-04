#ifndef activationfunctions_h
#define activationfunctions_h

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
 
using namespace std;

enum class Activation {
  relu,
  sigmoid,
  tanh,
  gauss,
  bent,
  softplus,
  sinusoid,
  isrlu,
  identity,
  softmax
};


/*
e(x)/sum = e(x)sum - e(x)e(x) = e(x)(sum - e(x))
*/
class IActivationFunction {
public:
  virtual float activation(const float &x) = 0;
  virtual float derivativeactivation(const float &x) = 0;
  virtual float s_activation(const float &x, const float &sum_exp, float &max_output) = 0;
  virtual float s_derivativeactivation(const float &x, const float &sum_exp, float &max_output) = 0;
};

class Identity : public IActivationFunction {
public:
  float activation(const float &x);
  float derivativeactivation(const float &x);
  float s_activation(const float &x, const float &sum_exp, float &max_output);
  float s_derivativeactivation(const float &x, const float &sum_exp, float &max_output);
};

class Relu : public IActivationFunction {
public:
  float activation(const float &x);
  float derivativeactivation(const float &x);
  float s_activation(const float &x, const float &sum_exp, float &max_output);
  float s_derivativeactivation(const float &x, const float &sum_exp, float &max_output);
};

class Sigmoid : public IActivationFunction {
public:
  float activation(const float &x);
  float derivativeactivation(const float &x);
  float s_activation(const float &x, const float &sum_exp, float &max_output);
  float s_derivativeactivation(const float &x, const float &sum_exp, float &max_output);
};

class Tanh : public IActivationFunction {
public:
  float activation(const float &x);
  float derivativeactivation(const float &x);
  float s_activation(const float &x, const float &sum_exp, float &max_output);
  float s_derivativeactivation(const float &x, const float &sum_exp, float &max_output);
};

class Gauss : public IActivationFunction {
public:
  float activation(const float &x);
  float derivativeactivation(const float &x);
  float s_activation(const float &x, const float &sum_exp, float &max_output);
  float s_derivativeactivation(const float &x, const float &sum_exp, float &max_output);
};

class Bent : public IActivationFunction {
public:
  float activation(const float &x);
  float derivativeactivation(const float &x);
  float s_activation(const float &x, const float &sum_exp, float &max_output);
  float s_derivativeactivation(const float &x, const float &sum_exp, float &max_output);
};

class SoftPlus : public IActivationFunction {
public:
  float activation(const float &x);
  float derivativeactivation(const float &x);
  float s_activation(const float &x, const float &sum_exp, float &max_output);
  float s_derivativeactivation(const float &x, const float &sum_exp, float &max_output);
};

class Sinusoid : public IActivationFunction {
public:
  float activation(const float &x);
  float derivativeactivation(const float &x);
  float s_activation(const float &x, const float &sum_exp, float &max_output);
  float s_derivativeactivation(const float &x, const float &sum_exp, float &max_output);
};

class Softmax : public IActivationFunction {
public:
  float activation(const float &x);
  float derivativeactivation(const float &x);
  float s_activation(const float &x, const float &sum_exp, float &max_output);
  float s_derivativeactivation(const float &x, const float &sum_exp, float &max_output);
};

class ISRLU : public IActivationFunction {
public:
  float alpha = 0.1;
  float activation(const float &x);
  float derivativeactivation(const float &x);
  float s_activation(const float &x, const float &sum_exp, float &max_output);
  float s_derivativeactivation(const float &x, const float &sum_exp, float &max_output);

};

#endif