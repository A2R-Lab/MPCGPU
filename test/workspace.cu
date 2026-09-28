#include "common/workspace.cuh"
#include <iostream>

int main() try {
    mpcgpu::SqpWorkspace first(14,7,64), second(14,7,64);
    first.begin(14,7,64);
    auto* device = first.device<float>(20);
    auto* host = first.host<float>(20);
    for (int i=0;i<20;++i) host[i]=0;
    host[0] = 17;
    mpcgpu::checkCuda(cudaMemcpy(device,host,20*sizeof(float),cudaMemcpyHostToDevice));
    first.begin(14,7,64);
    if (first.device<float>(20) != device || first.host<float>(20) != host ||
        first.deviceAllocations() != 1 || host[0] != 17)
        throw std::runtime_error("Workspace did not reuse storage");
    second.begin(14,7,64);
    if (second.device<float>(20) == device || second.host<float>(20) == host)
        throw std::runtime_error("Independent workspaces share storage");
    int rejected = 0;
    try { first.begin(14,7,32); } catch (const std::invalid_argument&) { ++rejected; }
    try { first.begin(14,7,64,1); } catch (const std::invalid_argument&) { ++rejected; }
    first.begin(14,7,64);
    try { first.device<float>(21); } catch (const std::invalid_argument&) { ++rejected; }
    if (rejected != 3) throw std::runtime_error("Workspace mismatch accepted");
    std::cout << "PASS workspace ownership/reuse\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
}
