#include <iostream>
#include "memd.hpp"
#include <xtensor/xnpy.hpp>


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
    xt::xarray<double> inp;
    size_t row, col;

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

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now(); // 計測開始時間


    std::cout << "input shape:(" << inp.shape(0) << ", " << inp.shape(1) << ")\n";
    std::cout << std::flush;


    const auto imf_num = yukilib::memd(out_path_prefix, inp, num_directions);


    auto now = std::chrono::system_clock::now();
    auto elapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start).count());
    std::cout << "全てのIMFを取り出すまでかかった時間" << elapsed << " [msec]\n";
    const auto imf = yukilib::get_imf(out_path_prefix, imf_num);
    std::cout << "imf shape:(" << imf.shape(0) << ", " << imf.shape(1) << ", " << imf.shape(2) << ")" << std::endl;
//        yukilib::write_imf_for_video(imf, out_path_prefix);

}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "引数の数が期待通りじゃない!(" << argc << " 個)\n"
                  << "実行方法: $> 実行ファイル名 入力データ.[csv|npy] 出力ファイル名のプレフィックス 単位ベクトルの次元数" << std::endl;
        return -1;
    } else {
        calc_memd(argv[1], argv[2], static_cast<unsigned>(std::stoi(argv[3])));
        return 0;
    }

}

