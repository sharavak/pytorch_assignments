#include <iostream>
#include <gtest/gtest.h>

#include "../src/cus_min.h"
#include<torch/torch.h>
#include<tuple>

using namespace std;


class MinimumParamTest :public ::testing::TestWithParam<tuple<torch::Tensor,torch::Tensor>> {
    public:
     Cus_Min obj;
};

TEST_P(MinimumParamTest, minimum) {
    torch::Tensor a = get<0>(GetParam());
    torch::Tensor b = get<1>(GetParam());
    ASSERT_TRUE(torch::equal(obj.cus_min(a,b), torch::minimum(a,b)));
}


INSTANTIATE_TEST_SUITE_P(
        MinimumOpsTest,
        MinimumParamTest,
        ::testing::Values(
                make_tuple(torch::randn({2,2}), torch::randn({2,2})),
                make_tuple(torch::randn({5,1,8}), torch::randn({5,1,8})),
                make_tuple(torch::randn({1,2,2,5}), torch::randn({1,2,2,5}))
            )
);


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}