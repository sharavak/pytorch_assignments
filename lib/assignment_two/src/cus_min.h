#ifndef CUS_MIN_H
#define CUS_MIN_H

#include<torch/torch.h>
using namespace std;


class Cus_Min{
    public:
        torch::Tensor cus_min (torch::Tensor ,torch::Tensor);
};
#endif


