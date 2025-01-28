#ifndef CUS_MIN_H
#define CUS_MIN_H

#include<torch/torch.h>
using namespace std;


class Cus_Min{
    public:
        torch::Tensor cus_min (torch::Tensor a,torch::Tensor b){   
            a=torch::multiply(a,-1);
            b=torch::multiply(b,-1);
            torch::Tensor res=torch::maximum(a,b);
            res=torch::multiply(res,-1);
            return res;
        }
};
#endif


