#ifndef CUS_OPS_H
#define CUS_OPS_H


#include<torch/torch.h>
using namespace at::indexing;
#include<iostream>

class Custom_Ops{
    public:
        torch::Tensor cus_logaddexp(torch::Tensor,torch::Tensor);
        torch::Tensor cus_addbmm(torch::Tensor,torch::Tensor,torch::Tensor);
};

#endif
