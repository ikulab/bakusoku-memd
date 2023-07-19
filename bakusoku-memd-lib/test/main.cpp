//
// Created by YukiShimada on 2023/07/04.
//

#include <gtest/gtest.h>

#include "../include/memd.hpp"

/// 警告出力用
class TestCout : public std::stringstream {
public:
    ~TestCout() override {
        std::cerr << "\033[0;32m[          ]\033[0;33m " << str().c_str() << "\033[m";
    }
};

/// MPIを使うコードをテストできるようにする
class MPIEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        int mpi_err = MPI_Init(nullptr, nullptr);
        ASSERT_FALSE(mpi_err);
    }

    void TearDown() override {
        int mpi_err = MPI_Finalize();
        ASSERT_FALSE(mpi_err);
    }

    ~MPIEnvironment() override = default;
};

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
    return RUN_ALL_TESTS();
}

int get_mpi_rank();

int get_mpi_size();

double calc_maximum_relative_diff_of_memd();

TEST(MainTest, accuracy_of_memd) {
    auto diff = calc_maximum_relative_diff_of_memd();
    if (get_mpi_size() <= 1) {
        TestCout() << "警告: MPIプロセス並列が無効! mpiexec 経由でテストを実行しようね!\n";
    }
#pragma omp parallel default(none)
    if (omp_get_num_threads() <= 1 && omp_get_thread_num() == 0) {
        TestCout() << "警告: OpenMPスレッド並列が無効! 環境変数 OMP_NUM_THREADS=n を設定して実行しようね!\n";
    }
    if (get_mpi_rank() == 0) {
        TestCout() << "\033[mIMFの max(|相対誤差|): " << diff << "\n";
        EXPECT_LE(diff, 1.e-10) << "IMFの max(|相対誤差|) が大きすぎるぞ";
        EXPECT_GE(diff, 1.e-12) << "IMFの max(|相対誤差|) が小さすぎてなんか怪しいぞ";
    }
}

int get_mpi_rank() {
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    assert(0 <= mpi_rank);
    return mpi_rank;
}

int get_mpi_size() {
    int mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    assert(0 < mpi_size);
    return mpi_size;
}

double calc_maximum_relative_diff_of_memd() {
    static const auto PREFIX_EXPECTED = "./res/ndarray_173_5_seed0_11dim_imf";
    static const auto PREFIX_OUT = "./out_173_5_11dim_imf";
    std::ifstream ifs("./res/ndarray_173_5_seed0.csv");
    assert(ifs.good());
    xt::xarray<double> inp = xt::load_csv<double>(ifs);
    ifs.close();
    yukilib::memd(PREFIX_OUT, inp, 11);
    MPI_Barrier(MPI_COMM_WORLD);
    xt::xarray<double> imf_out = yukilib::get_imf(PREFIX_OUT, 9);
    xt::xarray<double> imf_expected = yukilib::get_imf(PREFIX_EXPECTED, 9);
    xt::xarray<double> relative_diff = imf_out;
    std::transform(imf_out.begin(), imf_out.end(), imf_expected.begin(), relative_diff.begin(),
                   [](double lhs, double rhs) {
                       return (std::abs(lhs - rhs) / std::abs(rhs));
                   });
    return *std::max_element(relative_diff.begin(), relative_diff.end());
}

