#include <iostream>
#include "memd.hpp"
#include <xtensor/xnpy.hpp>

void print_parallel_info_rank0() {
    if (yukilib::internal::get_mpi_rank() == 0) {
#pragma omp parallel default(none) shared(std::cout)
        if (omp_get_thread_num() == 0) {
            std::cout << "Parallel execution info:\n\tMPI_Comm_size = " << yukilib::internal::get_mpi_size()
                      << "\n\tomp_get_num_threads = "
                      << omp_get_num_threads() << std::endl;
        }
    }
}

/// std::filesystem::path の代用
/// \param path 拡張子を含むファイルパス
/// \return ドットを含む拡張子
std::string get_extension(std::string_view path) {
    std::string ret{};
    size_t pos_dot = path.rfind('.');
    if (pos_dot != std::string::npos) {
        ret = path.substr(pos_dot, path.size() - pos_dot);
        for (auto &&v: ret) {
            v = static_cast<char>(std::tolower(v));
        }
    }
    return ret;
}

/// memdを実行して、imfをファイル出力する
/// \param dat_path 入力データファイルのパス 拡張子は .csv か .npy
/// \param out_path_prefix 出力ファイル名のプレフィックス  例 prefix="./out_imf_" でimfが2つの場合 -> ./out_imf_000.csv, ./out_imf_001.csv
/// \param num_directions 単位ベクトルの次元数
void calc_memd(std::string_view dat_path, std::string_view out_path_prefix, unsigned num_directions) {
    const int mpi_rank = yukilib::internal::get_mpi_rank();

    xt::xarray<double> inp;
    size_t row, col;
    if (mpi_rank == 0) {
        if (get_extension(dat_path) == ".csv") {
            std::ifstream ifs(dat_path.data());
            inp = xt::load_csv<double>(ifs);
        } else if (get_extension(dat_path) == ".npy") {
            inp = xt::load_npy<double>(dat_path.data());
        } else {
            throw std::runtime_error(std::string{
                    "入力ファイルの拡張子は .csv または .npy である必要があります。\n\tファイル名="}
                                             .append(dat_path.data())
                                             .append(", 拡張子=")
                                             .append(get_extension(dat_path)));
        }
        row = inp.shape(0);
        col = inp.shape(1);
    }
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now(); // 計測開始時間
    MPI_Bcast(&row, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&col, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0) {
        inp = xt::zeros<double>({row, col});
    }
    MPI_Bcast(inp.data(), static_cast<int>(inp.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0 || mpi_rank == 1) {
        std::cout << "input shape:(" << inp.shape(0) << ", " << inp.shape(1) << ")\n";
        if (mpi_rank == 1) {
            auto now = std::chrono::system_clock::now();
            auto elapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - start).count());
            std::cout << "MPI_Bcast にかかった時間" << elapsed << " [msec]\n";
        }
        std::cout << std::flush;
    }

    const auto imf_num = yukilib::memd(out_path_prefix, inp, num_directions);

    if (mpi_rank == 0) {
        auto now = std::chrono::system_clock::now();
        auto elapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                now - start).count());
        std::cout << "全てのIMFを取り出すまでかかった時間" << elapsed << " [msec]\n";
        const auto imf = yukilib::get_imf(out_path_prefix, imf_num);
        std::cout << "imf shape:(" << imf.shape(0) << ", " << imf.shape(1) << ", " << imf.shape(2) << ")" << std::endl;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    omp_set_max_active_levels(1);  // 多重の omp parallel を禁止する
    print_parallel_info_rank0();

    if (argc != 4) {
        std::cerr << "引数の数が期待通りじゃない!(" << argc << " 個)\n"
                  << "実行方法: $> 実行ファイル名 入力データ.[csv|npy] 出力ファイル名のプレフィックス 単位ベクトルの次元数" << std::endl;
        MPI_Finalize();
        return -1;
    } else {
        calc_memd(argv[1], argv[2], static_cast<unsigned>(std::stoi(argv[3])));
    }

    MPI_Finalize();
    return 0;
}

