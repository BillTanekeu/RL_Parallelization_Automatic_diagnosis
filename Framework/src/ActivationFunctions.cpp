#include "ActivationFunctions.hpp"

float Identity::activation(const float &x) { return x; }
float Identity::derivativeactivation(const float &x) { return 1.0; }
float Identity::s_activation(const float &x, const float &sum_exp, float &max_output){return 0;}
float Identity::s_derivativeactivation(const float &x, const float &sum_exp, float &max_output)
                {
                  float softmax_x = s_activation(x, sum_exp, max_output);

                  return softmax_x*(1 - softmax_x);
                  }

float Relu::activation(const float &x) { return max(float(0.0), x); }
float Relu::derivativeactivation(const float &x) {
  if(x>0) return 1;
  return 0;
 }
float Relu::s_activation(const float &x, const float &sum_exp, float &max_output){return 0;}
float Relu::s_derivativeactivation(const float &x, const float &sum_exp, float &max_output)
                {
                  float softmax_x = s_activation(x, sum_exp, max_output);

                  return softmax_x*(1 - softmax_x);
                  }



float Sigmoid::activation(const float &x) { return 1.0 / (1.0 + exp(-x)); }
float Sigmoid::derivativeactivation(const float &x) { return x * (1.0 - x); }
float Sigmoid::s_activation(const float &x, const float &sum_exp, float &max_output){return 0;}
float Sigmoid::s_derivativeactivation(const float &x, const float &sum_exp, float &max_output)
                {
                  float softmax_x = s_activation(x, sum_exp, max_output);

                  return softmax_x*(1 - softmax_x);
                  }



float Tanh::activation(const float &x) {
  return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}
float Tanh::derivativeactivation(const float &x) {
  return 1.0 - pow(activation(x), 2.0);
}
float Tanh::s_activation(const float &x, const float &sum_exp, float &max_output){return 0;}
float Tanh::s_derivativeactivation(const float &x, const float &sum_exp, float &max_output)
                {
                  float softmax_x = s_activation(x, sum_exp, max_output);

                  return softmax_x*(1 - softmax_x);
                  }


float Gauss::activation(const float &x) { return exp(-pow(x, 2.0)); }
float Gauss::derivativeactivation(const float &x) {
  return -2.0 * x * exp(-pow(x, 2.0));
}
float Gauss::s_activation(const float &x, const float &sum_exp, float &max_output){return 0;}
float Gauss::s_derivativeactivation(const float &x, const float &sum_exp, float &max_output)
                {
                  float softmax_x = s_activation(x, sum_exp, max_output);

                  return softmax_x*(1 - softmax_x);
                  }


float Bent::activation(const float &x) {
  return (sqrt(x * x + 1.0) - 1.0) / 2.0 + x;
}
float Bent::derivativeactivation(const float &x) {
  return x / (2.0 * sqrt(x * x + 1.0)) + 1.0;
}
float Bent::s_activation(const float &x, const float &sum_exp, float &max_output){return 0;}
float Bent::s_derivativeactivation(const float &x, const float &sum_exp, float &max_output)
                {
                  float softmax_x = s_activation(x, sum_exp, max_output);

                  return softmax_x*(1 - softmax_x);
                  }


float SoftPlus::activation(const float &x) { return log(1.0 + exp(x)); }
float SoftPlus::derivativeactivation(const float &x) {
  return 1.0 / (1.0 + exp(-x));
}
float SoftPlus::s_activation(const float &x, const float &sum_exp, float &max_output){return 0;}
float SoftPlus::s_derivativeactivation(const float &x, const float &sum_exp, float &max_output)
                {
                  float softmax_x = s_activation(x, sum_exp, max_output);

                  return softmax_x*(1 - softmax_x);
                  }


float Sinusoid::activation(const float &x) { return sin(x); }
float Sinusoid::derivativeactivation(const float &x) { return cos(x); }
float Sinusoid::s_activation(const float &x, const float &sum_exp, float &max_output){return 0;}
float Sinusoid::s_derivativeactivation(const float &x, const float &sum_exp, float &max_output)
                {
                  float softmax_x = s_activation(x, sum_exp, max_output);

                  return softmax_x*(1 - softmax_x);
                  }


float Softmax::s_activation(const float &x, const float &sum_exp, float &max_output){

  return exp(x - max_output)/sum_exp;
}
float Softmax::s_derivativeactivation(const float &x, const float &sum_exp, float &max_output){
  // pas complet
  return 0;
}
float Softmax::activation(const float &x) { return 0; }
float Softmax::derivativeactivation(const float &x) { return 0; }

float ISRLU::activation(const float &x) {
  if (x >= 0)
    return x;
  else
    return x / sqrt(1.0 + alpha * x * x);
}
float ISRLU::derivativeactivation(const float &x) {
  if (x >= 0)
    return 1.0;
  else
    return pow(1.0 / sqrt(1.0 + alpha * x * x), 3.0);
}
float ISRLU::s_activation(const float &x, const float &sum_exp, float &max_output){return 0;}
float ISRLU::s_derivativeactivation(const float &x, const float &sum_exp, float &max_output)
                {
                  float softmax_x = s_activation(x, sum_exp, max_output);

                  return softmax_x*(1 - softmax_x);
                  }

