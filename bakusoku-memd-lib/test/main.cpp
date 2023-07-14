//
// Created by YukiShimada on 2023/07/04.
//

#include <gtest/gtest.h>

#include "../include/memd.hpp"


double calc_maximum_relative_diff_of_memd();

TEST(MainTest, accuracy_of_memd) {
    auto diff = calc_maximum_relative_diff_of_memd();
    EXPECT_LE(diff, 1.e-10) << "IMFの max(|相対誤差|) が大きすぎるぞ";
    EXPECT_GE(diff, 1.e-11) << "IMFの max(|相対誤差|) が小さすぎてなんか怪しいぞ";
}

double calc_maximum_relative_diff_of_memd() {
    static const auto PREFIX_EXPECTED = "./res/ndarray_173_5_seed0_11dim_imf";
    static const auto PREFIX_OUT = "./out_173_5_11dim_imf";
    std::ifstream ifs("./res/ndarray_173_5_seed0.csv");
    assert(ifs.good());
    xt::xarray<double> inp = xt::load_csv<double>(ifs);
    ifs.close();
    yukilib::memd(PREFIX_OUT, inp, 11);
    xt::xarray<double> imf_out = yukilib::get_imf(PREFIX_OUT, 9);
    xt::xarray<double> imf_expected = yukilib::get_imf(PREFIX_EXPECTED, 9);
    xt::xarray<double> relative_diff = imf_out;
    std::transform(imf_out.begin(), imf_out.end(), imf_expected.begin(), relative_diff.begin(),
                   [](double lhs, double rhs) {
                       return (std::abs(lhs - rhs) / std::abs(rhs));
                   });
    return *std::max_element(relative_diff.begin(), relative_diff.end());
}

