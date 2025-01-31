#include <iostream>
#include <gtest/gtest.h>
#include "../src/cus_ops.h"
#include<torch/torch.h>
#include<tuple>

using namespace std;

class LogaddexpParamTest :public ::testing::TestWithParam<tuple<torch::Tensor,torch::Tensor>> {
    public:
     Custom_Ops obj;
};


class AddBmmParamTest :public ::testing::TestWithParam<tuple<torch::Tensor,torch::Tensor,torch::Tensor>> {
    public:
    Custom_Ops obj;
};

TEST_P(LogaddexpParamTest, Logaddexp) {
    torch::Tensor a = get<0>(GetParam());
    torch::Tensor b = get<1>(GetParam());
    ASSERT_TRUE(torch::allclose(obj.cus_logaddexp(a,b), torch::logaddexp(a,b)));
}

TEST_P(AddBmmParamTest, AddBmmexp) {
    torch::Tensor inp = get<0>(GetParam());
    torch::Tensor batch_a= get<1>(GetParam());
    torch::Tensor batch_b= get<2>(GetParam());
    ASSERT_TRUE(torch::allclose(obj.cus_addbmm(inp,batch_a,batch_b), torch::addbmm(inp,batch_a,batch_b)));
}

INSTANTIATE_TEST_SUITE_P(
        MinimumOpsTest,
        LogaddexpParamTest,
        ::testing::Values(
                make_tuple(torch::randn({2,2}), torch::randn({2,2})),
                make_tuple(torch::randn({5,1,8}), torch::randn({5,1,8})),
                make_tuple(torch::randn({1,2,15}), torch::randn({1,2,15}))
            )
);

INSTANTIATE_TEST_SUITE_P(
        MinimumOpsTest,
        AddBmmParamTest,
        ::testing::Values(
                make_tuple(torch::randn({7,2}) ,torch::randn({1,7,2}), torch::randn({1,2,2})),
                make_tuple(torch::randn({3,5}),torch::randn({5,3,2}), torch::randn({5,2,5})),
                make_tuple(torch::randn({4,12}),torch::randn({1,4,15}), torch::randn({1,15,12}))
            )
);


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}