#include<iostream>
#include "cus_min.h"

using namespace std;


torch::Tensor Cus_Min::cus_min (torch::Tensor a,torch::Tensor b){   
            a=torch::multiply(a,-1);
            b=torch::multiply(b,-1);
            torch::Tensor res=torch::maximum(a,b);
            res=torch::multiply(res,-1);
            return res;
        }