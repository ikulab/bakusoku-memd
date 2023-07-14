#pragma once

#include <algorithm>
#include <numbers>
#include <fstream>
#include <optional>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include "./define.hpp"
#include "./Spline.hpp"

//namespace {

namespace yukilib {

    enum class EStopCriteria {
        FixH, Stop
    };

    namespace internal {

        namespace np = xt;
//using namespace ::yukilib;
        using namespace xt::placeholders;  // to enable _ syntax

        xt::xarray<fp_t>
        CubicSpline(const xt::xarray<ptrdiff_t> &x, const xt::xarray<fp_t> &y, const xt::xarray<size_t> &t) {
            const std::vector<fp_t> x_vec(x.begin(), x.end());
            xt::xarray<fp_t> ret(std::vector<size_t>{t.shape(0), y.shape(1)});
            for (size_t i = 0; i < ret.shape(1); ++i) {
                assert(3 <= x.shape(0));
                auto y_vec = xt::view(y, xt::all(), i);
//        tk::spline spline(x_vec, std::vector(y_vec.begin(), y_vec.end()));
                tk::spline spline(x_vec, std::vector(y_vec.begin(), y_vec.end()), tk::spline::cspline, false,
                                  tk::spline::not_a_knot,
                                  -99.0, tk::spline::not_a_knot, -99.0);

//        auto spline = x.shape(0) < 4 ?
//                      //                      tk::spline(x_vec, std::vector(y_vec.begin(), y_vec.end())) :
//                      tk::spline(x_vec, std::vector(y_vec.begin(), y_vec.end()), tk::spline::cspline, false,
//                                 tk::spline::first_deriv,
//                                 0.0, tk::spline::first_deriv, 0.0) :
//                      tk::spline(x_vec, std::vector(y_vec.begin(), y_vec.end()), tk::spline::cspline, false,
//                                 tk::spline::not_a_knot,
//                                 -99.0, tk::spline::not_a_knot, -99.0);

//        for (const auto &v: t) {
//            ret(static_cast<size_t>(v), i) = spline(static_cast<fp_t>(v));
//        }
                for (size_t j = 0; j < t.shape(0); ++j) {
                    ret(j, i) = spline(static_cast<fp_t>(t(j)));
                }
            }
            return ret;
        }

        template<typename T>
        xt::xarray<T> make_empty() {
//    return xt::empty<T>(typename xt::xarray<T>::shape_type(0));
            return xt::xarray<T>(std::vector<size_t>{0});
        }

        /// ひとつのimfをファイル出力する　入力、出力共に 行=step, 列=channel を想定
        void write_imf(const xt::xarray<fp_t> &imf, unsigned imf_index, std::string_view prefix) {
//    const auto dat_channels = imf.shape(1);
//    const auto dat_steps = imf.shape(2);
            if (imf.shape(0) < imf.shape(1)) {
                std::cerr << "yukilib::internal::write_imf の入力imfが step < channel になってるぞ！ \n\t"
                          << "shape:(" << imf.shape(0) << ',' << imf.shape(1) << ')' << std::endl;
            }
            std::ostringstream sout{};

            sout << prefix << std::setfill('0') << std::setw(3) << imf_index << ".csv";
            std::ofstream ofs(sout.str());
//    for (size_t epoch = 0; epoch < dat_steps; ++epoch) {
//        for (size_t ch = 0; ch < dat_channels; ++ch) {
//            ofs << std::scientific << std::setprecision(18) << imf(i, ch, epoch);
//            if (ch != dat_channels - 1) { ofs << ','; }
//        }
//        if (epoch != dat_steps - 1) { ofs << '\n'; }
//    }
            ofs << std::scientific << std::setprecision(18);
            xt::dump_csv(ofs, imf);
        }

        std::pair<np::xarray<size_t>, np::xarray<size_t>> local_peaks(const np::xarray<fp_t> &x_arg);

        np::xarray<fp_t> hamm(unsigned n, int base) {
            np::xarray<fp_t> seq = np::zeros<fp_t>({1u, n});

            if (1 < base) {
                xt::xarray<unsigned> seed = xt::arange<unsigned>(1u, n + 1);
                auto base_inv = 1.0 / base;
                while (std::any_of(seed.begin(), seed.end(), [](const auto &x) { return x != 0; })) {
//            auto digit = np::remainder(xt::view(seed, xt::range(0, n)), static_cast<unsigned>(base));
                    // 変更点
//            xt::xarray<unsigned> digit = xt::fmod(xt::view(seed, xt::range(0, n)), static_cast<unsigned>(base));
                    xt::xarray<unsigned> digit = xt::view(seed, xt::range(0, n)) % static_cast<unsigned>(base);
                    seq = seq + digit * base_inv;
                    base_inv = base_inv / base;
//            seed = np::floor(seed / base);
                    // 変更点
                    seed /= static_cast<unsigned>(base);
                }
            } else {
                auto temp = np::arange(1u, n + 1);
//        seq = (np::remainder(temp, (-base + 1)) + 0.5) / (-base);
                // 変更点
                seq = (temp % (-base + 1) + 0.5) / (-base);
            }
            return seq;
        }


        np::xarray<size_t> zero_crossings(const np::xarray<fp_t> &x) {
            np::xarray<size_t> indzer = xt::adapt(
                    np::where(xt::view(x, xt::range(0, -1)) * xt::view(x, xt::range(1, _)) < 0)[0]);
            if (xt::any(xt::equal(x, 0.0))) {
                np::xarray<fp_t> iz = xt::adapt(np::where(xt::equal(x, 0.0))[0]);
                np::xarray<size_t> indz;
                if (xt::any(xt::equal(np::diff(iz), 1))) {
                    np::xarray<size_t> zer = xt::equal(x, 0.0);
                    np::xarray<size_t> diff_arg = np::concatenate(
                            xt::xtuple(np::xarray<size_t>({0}), zer, np::xarray<size_t>{0}));
                    auto dz = np::diff(diff_arg);
                    auto debz = xt::adapt(np::where(xt::equal(dz, 0))[0]);
                    auto finz = xt::adapt(np::where(xt::equal(dz, -1))[0]) - 1;
                    indz = np::round((debz + finz) / 2.0);
                } else {
                    indz = iz;
                }
                np::xarray<size_t> conc = np::concatenate(xt::xtuple(indzer, indz));
                indzer = np::sort(conc);
            }
            return indzer;
        }

        xt::xarray<fp_t>
        boundary_conditions_helper0(const xt::xarray<fp_t> &elements, const xt::xarray<size_t> &index) {
            xt::xarray<fp_t> ret(std::vector<size_t>{index.shape(0), elements.shape(1)});
            for (size_t i = 0; i < index.shape(0); ++i) {
                xt::view(ret, i, xt::all()) = xt::view(elements, index(i), xt::all());
            }
            return ret;
        }

        std::pair<std::optional<std::tuple<np::xarray<ptrdiff_t>, np::xarray<ptrdiff_t>, np::xarray<fp_t>, np::xarray<fp_t>>>, bool>
        boundary_conditions(
                const np::xarray<size_t> &indmin, const np::xarray<size_t> &indmax, const np::xarray<size_t> &t,
                const np::xarray<fp_t> &x,
                const np::xarray<fp_t> &z, size_t nbsym) {

            size_t lx = x.shape(0) - 1;
            size_t end_max = indmax.shape(0) - 1; // マイナスの値になるときは早期returnするから問題ないはず()
            size_t end_min = indmin.shape(0) - 1;
//    indmin = indmin.astype(int)
//    indmax = indmax.astype(int)

            //bool mode;
            if (indmin.shape(0) + indmax.shape(0) < 3) {
                return {std::nullopt, false};
            } else {
                //mode = 1; //#the projected signal has inadequate extrema
                assert(0 <= indmax.shape(0) && 0 <= indmin.shape(0) && 1 < nbsym && "size_tのend_maxがマイナスかもしれない");
            }

            np::xarray<size_t> lmax, lmin, rmax, rmin;
            size_t lsym, rsym;
            //#boundary conditions for interpolations :
            if (indmax[0] < indmin[0]) {
                if (x(0, 0) > x(indmin[0], 0)) {
                    auto n = std::min(1u, 2u);
                    lmax = np::flip(xt::view(indmax, xt::range(1, std::min(end_max + 1, nbsym + 1))), 0);
                    lmin = np::flip(xt::view(indmin, xt::range(_, std::min(end_min + 1, nbsym))), 0);
                    lsym = indmax[0];

                } else {
                    lmax = np::flip(xt::view(indmax, xt::range(_, std::min(end_max + 1, nbsym))), 0);
                    lmin = np::concatenate(
                            xt::xtuple(np::flip(xt::view(indmin, xt::range(_, std::min(end_min + 1, nbsym - 1))), 0),
                                       np::xarray<size_t>({0})));
                    lsym = 0;
                }

            } else {
                if (x(0, 0) < x(indmax[0], 0)) {
                    lmax = np::flip(xt::view(indmax, xt::range(_, std::min(end_max + 1, nbsym))), 0);
                    lmin = np::flip(xt::view(indmin, xt::range(1, std::min(end_min + 1, nbsym + 1))), 0);
                    lsym = indmin[0];

                } else {
                    lmax = np::concatenate(
                            xt::xtuple(np::flip(xt::view(indmax, xt::range(_, std::min(end_max + 1, nbsym - 1))), 0),
                                       np::xarray<size_t>({0})));
                    lmin = np::flip(xt::view(indmin, xt::range(_, std::min(end_min + 1, nbsym))), 0);
                    lsym = 0;
                }
            }

            if (indmax.back() < indmin.back()) {
                if (x.back() < x(indmax.back(), 0)) {
                    rmax = np::flip(
                            xt::view(indmax,
                                     xt::range(std::max<ptrdiff_t>(static_cast<ptrdiff_t>(end_max - nbsym + 1), 0), _)),
                            0);
                    auto _dbg = rmax.shape(0);
                    rmin = np::flip(
                            xt::view(indmin,
                                     xt::range(std::max<ptrdiff_t>(static_cast<ptrdiff_t>(end_min - nbsym), 0), -1)),
                            0);
                    rsym = indmin.back();

                } else {
                    rmax = np::concatenate(xt::xtuple(np::xarray<size_t>({lx}), np::flip(
                            xt::view(indmax,
                                     xt::range(std::max<ptrdiff_t>(static_cast<ptrdiff_t>(2 + end_max - nbsym), 0), _)),
                            0)));
                    auto _dbg = rmax.shape(0);
                    rmin = np::flip(
                            xt::view(indmin,
                                     xt::range(std::max<ptrdiff_t>(static_cast<ptrdiff_t>(end_min - nbsym + 1), 0), _)),
                            0);
                    rsym = lx;
                }

            } else {
                if (x.back() > x(indmin.back(), 0)) {
                    rmax = np::flip(
                            xt::view(indmax,
                                     xt::range(std::max<ptrdiff_t>(static_cast<ptrdiff_t>(end_max - nbsym), 0), -1)),
                            0);
                    rmin = np::flip(
                            xt::view(indmin,
                                     xt::range(std::max<ptrdiff_t>(static_cast<ptrdiff_t>(end_min - nbsym + 1), 0), _)),
                            0);
                    rsym = indmax.back();

                } else {
                    rmax = np::flip(
                            xt::view(indmax,
                                     xt::range(std::max<ptrdiff_t>(static_cast<ptrdiff_t>(end_max - nbsym + 1), 0), _)),
                            0);
                    rmin = np::concatenate(xt::xtuple(np::xarray<size_t>({lx}), np::flip(
                            xt::view(indmin,
                                     xt::range(std::max<ptrdiff_t>(static_cast<ptrdiff_t>(2 + end_min - nbsym), 0), _)),
                            0)));
                    rsym = lx;
                }
            }

//    np::xarray<ptrdiff_t> tlmin = static_cast<ptrdiff_t>((2 * t[lsym])) - static_cast<ptrdiff_t>(t[lmin]);
//    np::xarray<ptrdiff_t> tlmax = static_cast<ptrdiff_t>(2 * t[lsym]) - static_cast<ptrdiff_t>(t[lmax]);
//    np::xarray<ptrdiff_t> trmin = static_cast<ptrdiff_t>(2 * t[rsym]) - static_cast<ptrdiff_t>(t[rmin]);
//    np::xarray<ptrdiff_t> trmax = static_cast<ptrdiff_t>(2 * t[rsym]) - static_cast<ptrdiff_t>(t[rmax]);
            // 変更点
            np::xarray<ptrdiff_t> t_signed = xt::cast<ptrdiff_t>(t);
            np::xarray<ptrdiff_t> tlmin = (2 * t_signed[lsym]) - xt::index_view(t_signed, lmin);
            np::xarray<ptrdiff_t> tlmax = (2 * t_signed[lsym]) - xt::index_view(t_signed, lmax);
            np::xarray<ptrdiff_t> trmin = (2 * t_signed[rsym]) - xt::index_view(t_signed, rmin);
            np::xarray<ptrdiff_t> trmax = (2 * t_signed[rsym]) - xt::index_view(t_signed, rmax);

            //in case symmetrized parts do not extend enough
            if (tlmin[0] > static_cast<ptrdiff_t>(t[0]) or tlmax[0] > static_cast<ptrdiff_t>(t[0])) {
                if (lsym == indmax[0]) {
                    lmax = np::flip(xt::view(indmax, xt::range(_, std::min(end_max + 1, nbsym))), 0);
                } else {
                    lmin = np::flip(xt::view(indmin, xt::range(_, std::min(end_min + 1, nbsym))), 0);
                }
                if (lsym == 1) {
                    throw (std::logic_error("bug"));
                }
                lsym = 0;
//        tlmin = static_cast<ptrdiff_t>(2 * t[lsym]) - static_cast<ptrdiff_t>(t[lmin]);
//        tlmax = static_cast<ptrdiff_t>(2 * t[lsym]) - static_cast<ptrdiff_t>(t[lmax]);
                // 変更点
                tlmin = (2 * t_signed[lsym]) - xt::index_view(t_signed, lmin);
                tlmax = (2 * t_signed[lsym]) - xt::index_view(t_signed, lmax);
            }

            if (trmin.back() < static_cast<ptrdiff_t>(t[lx]) or trmax.back() < static_cast<ptrdiff_t>(t[lx])) {
                if (rsym == indmax.back()) {
                    rmax = np::flip(xt::view(indmax, xt::range(std::max<size_t>(end_max + 1 - nbsym, 0), _)), 0);
                } else {
                    rmin = np::flip(xt::view(indmin, xt::range(std::max<size_t>(end_min + 1 - nbsym, 0), _)), 0);
                }
                if (rsym == lx) {
                    throw (std::logic_error("bug"));
                }
                rsym = lx;
//        trmin = static_cast<ptrdiff_t>(2 * t[rsym]) - static_cast<ptrdiff_t>(t[rmin]);
//        trmax = static_cast<ptrdiff_t>(2 * t[rsym]) - static_cast<ptrdiff_t>(t[rmax]);
                // 変更点
                trmin = (2 * t_signed[rsym]) - xt::index_view(t_signed, rmin);
                trmax = (2 * t_signed[rsym]) - xt::index_view(t_signed, rmax);
            }

//    np::xarray<fp_t> zlmax = xt::view(z, xt::range(lmax, _));
//    np::xarray<fp_t> zlmin = xt::view(z, xt::range(lmin, _));
//    np::xarray<fp_t> zrmax = xt::view(z, xt::range(rmax, _));
//    np::xarray<fp_t> zrmin = xt::view(z, xt::range(rmin, _));
            // 変更点
            np::xarray<fp_t> zlmax = boundary_conditions_helper0(z, lmax);
            np::xarray<fp_t> zlmin = boundary_conditions_helper0(z, lmin);
            np::xarray<fp_t> zrmax = boundary_conditions_helper0(z, rmax);
            np::xarray<fp_t> zrmin = boundary_conditions_helper0(z, rmin);

//    np::xarray<ptrdiff_t> tmin = np::hstack(xt::xtuple(tlmin, np::xarray<ptrdiff_t>({t_signed[indmin]}), trmin));
//    np::xarray<ptrdiff_t> tmax = np::hstack(xt::xtuple(tlmax, np::xarray<ptrdiff_t>({t_signed[indmax]}), trmax));
//    np::xarray<fp_t> zmin = np::vstack(xt::xtuple(zlmin, xt::view(z, xt::range(indmin, _)), zrmin));
//    np::xarray<fp_t> zmax = np::vstack(xt::xtuple(zlmax, xt::view(z, xt::range(indmax, _)), zrmax));

            // 変更点
            np::xarray<ptrdiff_t> tmin = np::hstack(
                    xt::xtuple(tlmin, np::xarray<ptrdiff_t>({xt::index_view(t_signed, indmin)}), trmin));
            np::xarray<ptrdiff_t> tmax = np::hstack(
                    xt::xtuple(tlmax, np::xarray<ptrdiff_t>({xt::index_view(t_signed, indmax)}), trmax));
            np::xarray<fp_t> zmin = np::vstack(xt::xtuple(zlmin, boundary_conditions_helper0(z, indmin), zrmin));
            np::xarray<fp_t> zmax = np::vstack(xt::xtuple(zlmax, boundary_conditions_helper0(z, indmax), zrmax));

            return {std::optional{std::tuple{tmin, tmax, zmin, zmax}}, true};
        }


// computes the mean of the envelopes and the mode amplitude estimate
        std::tuple<np::xarray<fp_t>, np::xarray<fp_t>, np::xarray<fp_t>, np::xarray<fp_t>>
        envelope_mean(const np::xarray<fp_t> &m, const np::xarray<size_t> &t, const np::xarray<fp_t> &seq, size_t ndir,
                      size_t N, size_t N_dim) {

            constexpr size_t NBSYM = 2;
            size_t count = 0;

            np::xarray<fp_t> env_mean = np::zeros<fp_t>({t.shape(0), N_dim});
            np::xarray<fp_t> amp = np::zeros<fp_t>({t.shape(0)});
            np::xarray<fp_t> nem = np::zeros<fp_t>({ndir});
            np::xarray<fp_t> nzm = np::zeros<fp_t>({ndir});

            np::xarray<fp_t> dir_vec = np::zeros<fp_t>({N_dim, static_cast<size_t>(1)});
            for (size_t it = 0; it < ndir; ++it) {
                if (N_dim != 3) {     // Multivariate signal (for N_dim ~=3) with hammersley sequence
                    // # Linear normalisation of hammersley sequence in the range of -1.00 - 1.00
                    np::xarray<fp_t> b = 2.0 * xt::view(seq, it, xt::all()) - 1.0;

                    // # Find angles corresponding to the normalised sequence
                    np::xarray<fp_t> tht = np::transpose(
                            np::atan2(np::sqrt(np::flip(np::cumsum(xt::square(xt::view(b, xt::range(_, 0, -1)))), 0)),
                                      xt::view(b, xt::range(_, N_dim - 1))));

                    // # Find coordinates of unit direction vectors on n-sphere
                    xt::view(dir_vec, xt::all(), 0) = np::cumprod(
                            np::concatenate(xt::xtuple(np::xarray<fp_t>({1.0}), np::sin(tht))));
                    xt::view(dir_vec, xt::range(_, N_dim - 1), 0) =
                            np::cos(tht) * xt::view(dir_vec, xt::range(_, N_dim - 1), 0);

                } else { // # Trivariate signal with hammersley sequence
                    // # Linear normalisation of hammersley sequence in the range of -1.0 - 1.0
                    fp_t tt = 2.0 * seq(it, 0) - 1.0;
                    if (tt > 1) {
                        tt = 1;
                    } else if (tt < -1) {
                        tt = -1;
                    }

                    // # Normalize angle from 0 - 2*pi
                    fp_t phirad = seq(it, 1) * 2.0 * numbers::pi_v<fp_t>;
                    fp_t st = std::sqrt(1.0 - tt * tt);

//            dir_vec[0] = st * std::cos(phirad);
//            dir_vec[1] = st * std::sin(phirad);
//            dir_vec[2] = tt;
                    //変更点
                    xt::view(dir_vec, 0, 0) = st * std::cos(phirad);
                    xt::view(dir_vec, 1, 0) = st * std::sin(phirad);
                    xt::view(dir_vec, 2, 0) = tt;
                }

                // # Projection of input signal on nth (out of total ndir) direction vectors
                np::xarray<fp_t> y = xt::linalg::dot(m, dir_vec);

                // # Calculates the extrema of the projected signal
                const auto [indmin, indmax] = local_peaks(y);

                nem[it] = static_cast<fp_t>(indmin.shape(0) + indmax.shape(0));
                np::xarray<size_t> indzer = zero_crossings(y);
                nzm[it] = static_cast<fp_t>(indzer.shape(0));

//        tmin, tmax, zmin, zmax, mode = boundary_conditions(indmin, indmax, t, y, m, NBSYM); // 原本はこっちだよ
                const auto [may_null_results, mode] = boundary_conditions(indmin, indmax, t, y, m, NBSYM);

                // # Calculate multidimensional envelopes using spline interpolation
                // # Only done if number of extrema of the projected signal exceed 3
                if (mode) {
                    assert(may_null_results.has_value() && "mode == true ");
                    const auto [tmin, tmax, zmin, zmax] = *may_null_results;
//            fmin = CubicSpline(tmin, zmin, bc_type = 'not-a-knot')
//            env_min = fmin(t)
                    const auto env_min = CubicSpline(tmin, zmin, t);
//            fmax = CubicSpline(tmax, zmax, bc_type = 'not-a-knot')
//            env_max = fmax(t)

                    const auto env_max = CubicSpline(tmax, zmax, t);
//            amp = amp + np.sqrt(np.sum(np.power(env_max - env_min, 2), axis = 1)) / 2
                    amp = amp + np::sqrt(np::sum(np::square(env_max - env_min), 1)) / 2;

                    env_mean = env_mean + (env_max + env_min) / 2;
                } else {   // # if the projected signal has inadequate extrema
                    count = count + 1;
                }
            }

            if (ndir > count) {
                env_mean = env_mean / (ndir - count);
                amp = amp / (ndir - count);
            } else {
                env_mean = np::zeros<fp_t>({N, N_dim});
                amp = np::zeros<fp_t>({N});
                nem = np::zeros<fp_t>({ndir});
            }

            return {env_mean, nem, nzm, amp};
        }

// #Stopping criterion
        std::pair<bool, np::xarray<fp_t>>
        stop(const np::xarray<fp_t> &m, const np::xarray<size_t> &t, fp_t sd, fp_t sd2, fp_t tol,
             const np::xarray<fp_t> &seq,
             size_t ndir, size_t N, size_t N_dim) {
            bool stp;
            np::xarray<fp_t> env_mean;
            try {
                const auto [env_mean_tmp, nem, nzm, amp] = envelope_mean(m, t, seq, ndir, N, N_dim);
                env_mean = env_mean_tmp;
                np::xarray<fp_t> sx = np::sqrt(np::sum(np::square(env_mean), 1));

                if (all(amp)) {    // # something is wrong here
                    sx = sx / amp;
                }

                if (!((np::mean(sx > sd)(0) > tol or any(sx > sd2)) and any(nem > 2))) {
                    stp = true;
                } else {
                    stp = false;
                }
            } catch (const std::exception &e) {
                std::cerr << "想定済みの例外だよん-> " << e.what() << '\n';
                env_mean = np::zeros<fp_t>({N, N_dim});
                stp = true;
            }

            return {stp, env_mean};
        }

        std::tuple<bool, np::xarray<fp_t>, size_t>
        fix(const np::xarray<fp_t> &m, const np::xarray<size_t> &t, const np::xarray<fp_t> &seq, size_t ndir,
            size_t stp_cnt,
            size_t counter, size_t N, size_t N_dim) {
            bool stp;
            np::xarray<fp_t> env_mean;
            try {
                const auto [env_mean_tmp, nem, nzm, amp] = envelope_mean(m, t, seq, ndir, N, N_dim);
                env_mean = env_mean_tmp;
                if (all(np::abs(nzm - nem) > 1)) {
                    stp = false;
                    counter = 0;
                } else {
                    counter = counter + 1;
                    stp = (counter >= stp_cnt);
                }
            } catch (const std::exception &e) {
                std::cerr << "想定済みの例外だよん-> " << e.what() << '\n';
                env_mean = np::zeros<fp_t>({N, N_dim});
                stp = true;
            }

            return {stp, env_mean, counter};
        }

        std::pair<np::xarray<fp_t>, np::xarray<size_t>> peaks(const np::xarray<fp_t> &X) {
            np::xarray<fp_t> dX = np::transpose(np::sign(np::diff(np::transpose(X))));
            np::xarray<size_t> locs_max =
                    xt::adapt(np::where(xt::view(dX, xt::range(_, -1)) > 0 && xt::view(dX, xt::range(1, _)) < 0)[0]) +
                    1;
//    np::xarray<fp_t> pks_max = xt::index_view(X, locs_max);
            // 変更点
            np::xarray<fp_t> X_copied = X;
            np::xarray<fp_t> pks_max = xt::index_view(X_copied.reshape({X_copied.shape(0)}), locs_max);
            pks_max = pks_max.reshape({pks_max.shape(0), 1});
            // 変更点ここまで
            return {pks_max, locs_max};
        }

        std::pair<np::xarray<size_t>, np::xarray<size_t>> local_peaks(const np::xarray<fp_t> &x_arg) {
            np::xarray<fp_t> x = x_arg;
            if (all(x < 1e-5)) {
                x = np::zeros<fp_t>({static_cast<size_t>(1), x.shape(0)});
            }

            size_t m = x.shape(0) - 1;

//# Calculates the extrema of the projected signal
//# Difference between subsequent elements:
            np::xarray<fp_t> dy = xt::transpose(np::diff(xt::transpose(x)));
            xt::xarray<size_t> a = xt::adapt(np::where(xt::not_equal(dy, 0.0))[0]);
            xt::xarray<size_t> lm = xt::adapt(np::where(xt::not_equal(np::diff(a), 1))[0]);
            lm += 1;
            np::xarray<size_t> d = xt::index_view(a, lm) - xt::index_view(a, lm - 1);
            xt::index_view(a, lm) = xt::index_view(a, lm) - np::floor(d / 2);
//    a = np.insert(a, a.shape(0), m)
            a = np::concatenate(xt::xtuple(a, np::xarray<size_t>({m})), 0);

//    xt::xarray<fp_t> ya = xt::index_view(x, a);
            // 変更点
            xt::xarray<fp_t> ya;
            if (x.shape(0) == 1) {
                ya = xt::index_view(x, a);
            } else {
                ya = xt::index_view(x.reshape({x.shape(0)}), a);
                ya = ya.reshape({ya.shape(0), 1});
            }
            // 変更点ここまで

            np::xarray<size_t> indmin, indmax;
            if (ya.shape(0) > 1) {
//# Maxima
                const auto [pks_max, loc_max] = peaks(ya);
//# Minima
                const auto [pks_min, loc_min] = peaks(-ya);

                if (pks_min.shape(0) > 0) {
//            indmin = a[loc_min];
                    // 変更点
                    indmin = xt::index_view(a, loc_min);
                } else {
//            indmin = np.asarray([]);
                    indmin = make_empty<size_t>();
                }
                if (pks_max.shape(0) > 0) {
//            indmax = a[loc_max];
                    // 変更点
                    indmax = xt::index_view(a, loc_max);
                } else {
//            indmax = np.asarray([])
                    indmax = make_empty<size_t>();
                }
            } else {
//        indmin = np.array([])
//        indmax = np.array([])
                indmin = make_empty<size_t>();
                indmax = make_empty<size_t>();
            }
            return {indmin, indmax};
        }

        bool stop_emd(const np::xarray<fp_t> &r, const np::xarray<fp_t> &seq, size_t ndir, size_t N_dim) {
            np::xarray<fp_t> ner = np::zeros<fp_t>({ndir, static_cast<size_t>(1)});
            np::xarray<fp_t> dir_vec = np::zeros<fp_t>({N_dim, static_cast<size_t>(1)});

            for (size_t it = 0; it < ndir; ++it) {
                if (N_dim != 3) {  //# Multivariate signal( for N_dim ~ = 3) with hammersley sequence
//# Linear normalisation of hammersley sequence in the range of -1.00 - 1.00
                    np::xarray<fp_t> b = 2 * xt::view(seq, it, xt::all()) - 1;

//# Find angles corresponding to the normalised sequence
                    xt::xarray<fp_t> tht = xt::transpose(
                            xt::atan2(np::sqrt(xt::flip(np::cumsum(xt::square(xt::view(b, xt::range(_, 0, -1)))), 0)),
                                      xt::view(b, xt::range(_, N_dim - 1))));

//# Find coordinates of unit direction vectors on n-sphere
                    xt::view(dir_vec, xt::all(), 0) = np::cumprod(
                            np::concatenate(xt::xtuple(np::xarray<fp_t>({1.0}), np::sin(tht))));
                    xt::view(dir_vec, xt::range(_, N_dim - 1), 0) =
                            np::cos(tht) * xt::view(dir_vec, xt::range(_, N_dim - 1), 0);

                } else {  //# Trivariate signal with hammersley sequence
//# Linear normalisation of hammersley sequence in the range of -1.0 - 1.0
                    fp_t tt = 2 * seq(it, 0) - 1;
                    if (tt > 1) {
                        tt = 1;
                    } else if (tt < -1) {
                        tt = -1;
                    }
//# Normalize angle from 0 - 2*pi
                    fp_t phirad = seq(it, 1) * 2 * numbers::pi_v<fp_t>;
                    fp_t st = std::sqrt(1.0 - tt * tt);

                    xt::view(dir_vec, 0, xt::all()) = st * cos(phirad);
                    xt::view(dir_vec, 1, xt::all()) = st * sin(phirad);
                    xt::view(dir_vec, 2, xt::all()) = tt;
                }
//# Projection of input signal on nth (out of total ndir) direction
//# vectors
                xt::xarray<fp_t> y = xt::linalg::dot(r, dir_vec);

//# Calculates the extrema of the projected signal
                const auto [indmin, indmax] = local_peaks(y);

                xt::view(ner, it) = indmin.shape(0) + indmax.shape(0);
            }

//# Stops if the all projected signals have less than 3 extrema
            bool stp = xt::all(ner < 3);

            return stp;
        }

        constexpr bool is_prime(unsigned x) {
            if (x == 2) {
                return true;
            } else {
                for (unsigned number = 3; number < x; ++number) {
                    if (x % number == 0 or x % 2 == 0) {
//# print number
                        return false;
                    }
                }
                return true;
            }
        }

        std::vector<unsigned> nth_prime(unsigned n) {
            std::vector<unsigned> lst({2});
            for (unsigned i = 3; i < 104745; ++i) {
                if (is_prime(i)) {
                    lst.push_back(i);
                    if (lst.size() == n) {
                        return lst;
                    }
                }
            }
            throw (std::logic_error("ここを通るはずがないよ"));
        }

        std::tuple<np::xarray<fp_t>, np::xarray<fp_t>, np::xarray<size_t>, size_t, size_t, size_t, fp_t, fp_t, fp_t, size_t, size_t, EStopCriteria, std::optional<size_t>>
        set_value(const np::xarray<fp_t> &arg, std::optional<unsigned> n1, std::optional<EStopCriteria> n2,
                  std::optional<std::tuple<fp_t, fp_t, fp_t>> n3) {

//    args = args[0]
//    narg = len(args)
            np::xarray<fp_t> q = arg;
            unsigned narg = 1;
            if (n1.has_value()) {
                ++narg;
            }
            if (n2.has_value()) {
                ++narg;
            }
            if (n3.has_value()) {
                ++narg;
            }
//    arg = args[0]

            unsigned ndir{}, MAXITERATIONS{};
            std::optional<size_t> stp_cnt{};
            fp_t sd{}, sd2{}, tol{};
            EStopCriteria stp_crit{};
            np::xarray<fp_t> stp_vec{};
            std::vector<int> base{};

            if (narg == 0) {
                throw std::runtime_error("Not enough input arguments.");
            } else if (narg > 4) {
                throw std::runtime_error("Too many input arguments.");
            } else if (narg == 1) {
                ndir = 64;  //# default
                stp_crit = EStopCriteria::Stop;  //# default
                stp_vec = np::xarray<fp_t>({0.075, 0.75, 0.075});  //# default
                sd = stp_vec[0];
                sd2 = stp_vec[1];
                tol = stp_vec[2];
            } else if (narg == 2) {
                ndir = n1.value();
                stp_crit = EStopCriteria::Stop; //# default
                stp_vec = np::xarray<fp_t>({0.075, 0.75, 0.075});  //# default
                sd = stp_vec[0];
                sd2 = stp_vec[1];
                tol = stp_vec[2];
            } else if (narg == 3) {
                if (n1.has_value()) {
                    ndir = n1.value();
                } else {
                    ndir = 64;  //# default
                }
                stp_crit = n2.value();
                if (stp_crit == EStopCriteria::Stop) {
                    stp_vec = np::xarray<fp_t>({0.075, 0.75, 0.075});  //# default
                    sd = stp_vec[0];
                    sd2 = stp_vec[1];
                    tol = stp_vec[2];
                } else if (stp_crit == EStopCriteria::FixH) {
                    stp_cnt = 2;  //# default
                }
            } else if (narg == 4) {
                throw std::logic_error("n3, stp_vec, stp_cnt らへんの型がおかしい");
//        if (n1.has_value()) {
//            ndir = n1.value();
//        } else {
//            ndir = 64;  //# default
//        }
//        stp_crit = n2.value();
//        if (n2==EStopCriteria::Stop) {
//            stp_vec = n3.value();
//            sd=stp_vec[0];
//            sd2=stp_vec[1];
//            tol=stp_vec[2];
//        } else if (n2 == EStopCriteria::FixH) {
//            stp_cnt = n3.value();
//        }
            }

//# Rescale input signal if required
            if (q.shape(0) == 0) {  //# Doesn't do the same as the Matlab script
                throw std::runtime_error("emptyDataSet. Data set cannot be empty.");
            }
            if (q.shape(0) < q.shape(1)) {
                q = np::xarray<fp_t>(xt::transpose(arg));
            }

//# Dimension of input signal
            auto N_dim = static_cast<unsigned>(q.shape(1));
            if (N_dim < 3) {
                throw std::runtime_error("Function only processes the signal having more than 3.");
            }

//# Length of input signal
            size_t N = q.shape(0);

//# Check validity of Input parameters                                       #  Doesn't do the same as the Matlab script
            if (ndir < 6) {
                throw std::runtime_error("invalid num_dir. num_dir should be an integer greater than or equal to 6.");
            }
            if (stp_crit != EStopCriteria::Stop and stp_crit != EStopCriteria::FixH) {
                throw std::runtime_error("invalid stop_criteria. stop_criteria should be either fix_h or stop");
            }
//    if (not isinstance(stp_vec, (list, tuple, np.ndarray)) or any(
//                x for x in stp_vec if not isinstance(x, (int, float, complex)))){
//    sys.exit(
//            'invalid stop_vector. stop_vector should be a list with three elements e.g. default is [0.75,0.75,0.75]')}
//    if (stp_cnt != None) {
//        if (not isinstance(stp_cnt, int) or stp_cnt < 0) {
//            sys.exit('invalid stop_count. stop_count should be a nonnegative integer.')
//        }
//    }

//# Initializations for Hammersley function
            base.push_back(static_cast<int>(-ndir));

            np::xarray<fp_t> seq;
//# Find the pointset for the given input signal
            if (N_dim == 3) {
                base.push_back(2);
                seq = np::zeros<fp_t>({ndir, N_dim - 1});
                for (size_t it = 0; it < N_dim - 1; ++it) {
                    // 変更点
                    auto tmp = hamm(ndir, base[it]);
                    if (tmp.shape(0) == 1 && tmp.shape(1) == seq.shape(0)) {
                        xt::view(seq, xt::all(), it) = tmp.reshape({seq.shape(0)});
                    } else {
                        xt::view(seq, xt::all(), it) = tmp;
                    }
                }
            } else {
//# Prime numbers for Hammersley sequence
                auto prm = nth_prime(N_dim - 1);
                for (unsigned itr = 1; itr < N_dim; ++itr) {
                    base.push_back(static_cast<int>(prm[itr - 1]));
                }
                seq = np::zeros<fp_t>({ndir, N_dim});
                for (unsigned it = 0; it < N_dim; ++it) {
                    // xt::view(seq, xt::all(), it) = hamm(ndir, base[it]);
                    // 変更点
                    auto tmp = hamm(ndir, base[it]);
                    if (tmp.shape(0) == 1 && tmp.shape(1) == seq.shape(0)) {
                        xt::view(seq, xt::all(), it) = tmp.reshape({seq.shape(0)});
                    } else {
                        xt::view(seq, xt::all(), it) = tmp;
                    }
                }
            }
//# Define t
            auto t = np::arange<size_t>(1, N + 1);
//# Counter
            size_t nbit = 0;
            MAXITERATIONS = 1000;  //# default

            return {q, seq, t, ndir, N_dim, N, sd, sd2, tol, nbit, MAXITERATIONS, stp_crit, stp_cnt};
        }

    }  // namespace internal


/**
 * 多変量経験的モード分解をするよ
 * @param arg 分解したいデータ列
 * @param n1 num_directions
 * @param n2 stopping criteria
 * @param n3 stop_vec
 * @return 3Dの行列　
 */
    unsigned memd(std::string_view prefix, const xt::xarray<fp_t> &arg, std::optional<unsigned> n1 = std::nullopt,
                  std::optional<EStopCriteria> n2 = std::nullopt,
                  std::optional<std::tuple<fp_t, fp_t, fp_t>> n3 = std::nullopt) {
        auto [x, seq, t, ndir, N_dim, N, sd, sd2, tol, _nbit, MAXITERATIONS, stop_crit, stp_cnt] = internal::set_value(
                arg,
                n1,
                n2,
                n3);
        size_t nbit = _nbit;
        auto r = x;
        unsigned n_imf = 1;
        std::vector<xt::xarray<fp_t>>
                q{};
        q.reserve(10);  //todo 適切な初期容量を決める

        while (!internal::stop_emd(r, seq, ndir, N_dim)) {
//# current mode
            auto m = r;

            bool stop_sift{};
            xt::xarray<fp_t> env_mean{};
            size_t counter{};

//# computation of mean and stopping criterion
            if (stop_crit == EStopCriteria::Stop) {
                const auto [a, b] = internal::stop(m, t, sd, sd2, tol, seq, ndir, N, N_dim);
                stop_sift = a;
                env_mean = b;
            } else {
                counter = 0;
                const auto [a, b, c] = internal::fix(m, t, seq, ndir, stp_cnt.value(), counter, N, N_dim);
                stop_sift = a;
                env_mean = b;
                counter = c;
            }

//# In case the current mode is so small that machine precision can cause
//# spurious extrema to appear
            if (xt::amax(xt::abs(m))(0) < (1e-10) * (xt::amax(xt::abs(x)))(0)) {
                if (!stop_sift) {
                    std::cout << "emd:warning: forced stop of EMD : too small amplitude\n";
                } else {
                    std::cout << "forced stop of EMD : too small amplitude\n    ";
                }
                break;
            }

//# sifting loop
            while (!stop_sift && nbit < MAXITERATIONS) {
//# sifting
                m = m - env_mean;

//# computation of mean and stopping criterion
                if (stop_crit == EStopCriteria::Stop) {
                    const auto [a, b] = internal::stop(m, t, sd, sd2, tol, seq, ndir, N, N_dim);
                    stop_sift = a;
                    env_mean = b;
                } else {
                    const auto [a, b, c] = internal::fix(m, t, seq, ndir, stp_cnt.value(), counter, N, N_dim);
                    stop_sift = a;
                    env_mean = b;
                    counter = c;
                }

                nbit = nbit + 1;

                if (nbit == (MAXITERATIONS - 1) and nbit > 100) {
                    std::cerr << "emd:warning!  forced stop of sifting : too many erations" << '\n';
                }
            }
            q.emplace_back(xt::transpose(m));

            internal::write_imf(m, n_imf - 1, prefix);

            n_imf = n_imf + 1;
            r = r - m;
            nbit = 0;
        }

//# Stores the residue
        q.emplace_back(xt::transpose(r));
//    q = xt.asarray(q)
        // この辺自分オリジナル処理
        xt::xarray<fp_t> q_ret{};
        q_ret = q[0].reshape({1, q[0].shape(0), q[0].shape(1)});
        for (size_t i = 1; i < q.size(); ++i) {
            q_ret = xt::vstack(xt::xtuple(q_ret, q[i].reshape({1, q[i].shape(0), q[i].shape(1)})));
        }
//#sprintf('Elapsed time: %f\n',toc);
//        return q_ret;
        return n_imf;
    }

    /// prefixから始まるimf_num個のファイルからimfを取り出し、[imf_num, channel, step]の形で返す
/// \param prefix imfを保存したファイルの接頭辞
/// \param imf_num imfを保存したファイルの数
/// \return [imf_num, channel, step] の3次元行列
    xt::xarray<fp_t> get_imf(std::string_view prefix, unsigned imf_num) {
        xt::xarray<fp_t> q_ret{};
        std::ostringstream sout{};
        for (size_t i = 0; i < imf_num; ++i) {
            sout.str("");
            sout.clear();
            sout << prefix << std::setfill('0') << std::setw(3) << i << ".csv";
            std::ifstream ifs(sout.str());
            if (ifs.fail()) {
                break;
            }
            xt::xarray<double> imf = xt::transpose(xt::load_csv<double>(ifs));
            if (i == 0) {
                q_ret = imf.reshape({1, imf.shape(0), imf.shape(1)});
            } else {
                q_ret = xt::vstack(xt::xtuple(q_ret, imf.reshape({1, imf.shape(0), imf.shape(1)})));
            }
        }
        return q_ret;
    }
}  // namespace yukilib

//}  // unnamed namespace
