// Known-solution SPD example and API regression gate. No timing collection.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>
#include "gpu_pcg.cuh"

#ifndef TEST_DOUBLE
#define TEST_DOUBLE 0
#endif
using T = std::conditional_t<TEST_DOUBLE, double, float>;

int main(int argc, char** argv) try {
    const std::string mode = argc > 1 ? argv[1] : "host";
    const int s = STATE_SIZE, N = KNOT_POINTS, n = s*N;
    std::vector<T> A(3*s*s*N, 0), P(A.size(), 0), b(n, 0), x(n, 0);
    for (int k=0; k<N; ++k) for (int j=0; j<s; ++j) {
        const int row = k*s+j;
        const T diagonal = T(2) + T(row)/T(n);
        A[(3*k+1)*s*s+j*s+j] = diagonal;
        P[(3*k+1)*s*s+j*s+j] = T(1)/diagonal;
        b[row] = diagonal;
        if (k) { A[3*k*s*s+j*s+j] = T(-0.125); b[row] -= T(0.125); }
        if (k+1<N) { A[(3*k+2)*s*s+j*s+j] = T(-0.125); b[row] -= T(0.125); }
    }
    pcg_config<T> config;
    config.pcg_exit_tol = T(1e-18);
    config.pcg_rel_tol = T(1e-14);
    config.pcg_max_iter = 200;
    config.pcg_block = dim3(64);
    if (mode == "invalid") {
        int rejected = 0;
        auto expect = [&](auto operation) {
            try { operation(); } catch (const std::invalid_argument&) { ++rejected; }
        };
        expect([&]{ solvePCG(A.data(), b.data(), x.data(), s+1, N, &config); });
        config.empty_pinv = 0;
        expect([&]{ solvePCG(A.data(), b.data(), x.data(), s, N, &config); });
        config.empty_pinv = 1; config.pcg_rel_tol = T(-1);
        expect([&]{ solvePCG(A.data(), b.data(), x.data(), s, N, &config); });
        config.pcg_rel_tol = T(1e-14); config.pcg_block = dim3(32,2);
        expect([&]{ solvePCG(A.data(), b.data(), x.data(), s, N, &config); });
        config.pcg_block = dim3(64); b[0] = std::numeric_limits<T>::quiet_NaN();
        expect([&]{ solvePCG(A.data(), b.data(), x.data(), s, N, &config); });
        if (rejected != 5) throw std::runtime_error("Invalid configuration accepted");
    } else if (mode == "host") {
        const auto original = A;
        const auto iterations = solvePCG(A.data(), b.data(), x.data(), s, N, &config);
        if (iterations == 0 || iterations >= config.pcg_max_iter || A != original)
            throw std::runtime_error("Host wrapper result/input contract failed");
    } else {
        if (mode == "warm") std::fill(x.begin(), x.end(), T(1));
        else if (mode == "zero" || mode == "zero-exact") {
            std::fill(b.begin(), b.end(), T(0));
            if (mode == "zero-exact") config.pcg_exit_tol = config.pcg_rel_tol = T(0);
        }
        else if (mode == "cap") config.pcg_max_iter = 0;
        else if (mode != "device") throw std::invalid_argument("Unknown test case");
        gbd_detail::Buffer<T> a(A.size()), p_inv(P.size()), rhs(n), solution(n), r(n), p(n), v(N), eta(N);
        auto upload = [](T* dst, const std::vector<T>& src) {
            gbd_detail::check(cudaMemcpy(dst, src.data(), src.size()*sizeof(T), cudaMemcpyHostToDevice));
        };
        upload(a.data,A); upload(p_inv.data,P); upload(rhs.data,b); upload(solution.data,x);
        auto result = solvePCGChecked(s,N,a.data,p_inv.data,rhs.data,solution.data,r.data,p.data,v.data,eta.data,&config);
        gbd_detail::check(cudaMemcpy(x.data(),solution.data,n*sizeof(T),cudaMemcpyDeviceToHost));
        if ((mode == "cap") != result.iteration_limit)
            throw std::runtime_error("Iteration-limit status failed");
        if ((mode == "zero" || mode == "zero-exact" || mode == "warm" || mode == "cap") && result.iterations != 0)
            throw std::runtime_error("Zero-iteration result failed");
    }
    if (mode != "invalid") {
        const T expected = mode == "zero" || mode == "zero-exact" || mode == "cap" ? T(0) : T(1);
        for (T value : x) if (!std::isfinite(value) || std::abs(value-expected) > T(2e-5))
            throw std::runtime_error("Known-solution mismatch");
    }
    std::cout << "PASS " << mode << " " << s << "x" << N << " " << (TEST_DOUBLE ? "double" : "float") << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
