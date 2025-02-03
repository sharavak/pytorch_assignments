#include<iostream>
#include "cus_ops.h"

using namespace std;


torch::Tensor Custom_Ops::cus_logaddexp(torch::Tensor a,torch::Tensor b){
    a=torch::exp(a);
    b=torch::exp(b);
    return torch::log(torch::add(a,b));
}

torch::Tensor Custom_Ops::cus_addbmm(torch::Tensor input,torch::Tensor mat_a,torch::Tensor mat_b){
    int r1=mat_a.sizes()[1];
    int c1=mat_a.sizes()[2];
    int c2=mat_b.sizes()[2];
    int b_size=mat_a.sizes()[0];
    torch::Tensor res=torch::zeros({b_size,r1,c2});
    
    for (int b=0;b<b_size;b++){
         for(int i=0;i<r1;i++){
            for(int j=0;j<c2;j++){
              for(int k=0;k<c1;k++)
                res[b][i][j]+=mat_a[b][i][k]*mat_b[b][k][j];
             }
         }
     }
    res=torch::sum(res,0);
    
    return torch::add(res,input);
}