#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>

#include "typedef.hpp"
#include "npu_utils.hpp"
#include "vm_args.hpp"
#include "utils.hpp"


namespace po = boost::program_options;

void MM(int K, int N, vdtype& q, vdtype& k, vdtype& o);

int main(int argc, const char *argv[]) {
	// Fix the seed to ensure reproducibility in CI.
    srand(1);
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    // Add custom options
    desc.add_options()("Trace_size,t", po::value<int>()->default_value(131072), "Trace size");

    arg_utils::parse_options(argc, argv, desc, vm);
    
    // User logic
    int K = 256;
    int N = 128;
    int TRACE_SIZE = vm["Trace_size"].as<int>();
    int O_VOLUME = 8 * K;
    int A_VOLUME = 8 * N;
    int B_VOLUME = N * K;

    accel_user_desc accel_desc;
    accel_desc.xclbin_name = "build/xclbins/aie2.xclbin";
    accel_desc.instr_name = "build/insts/aie2.txt";

    npu_app npu_instance(1, 1, 0);
    int app_id = npu_instance.register_accel_app(accel_desc);

	vector a = npu_instance.create_bo_vector<dtype>(A_VOLUME, 3, app_id);
	vector b = npu_instance.create_bo_vector<dtype>(B_VOLUME, 4, app_id);
	vector o = npu_instance.create_bo_vector<dtype>(O_VOLUME, 5, app_id);
	vector bo_trace = npu_instance.create_bo_vector<char>(TRACE_SIZE, 7, app_id);

    o.memset(0);

    for (int i = 0; i < A_VOLUME; i++){
        a[i] = utils::getRand(0.2, 1);
    }

    for (int i = 0; i < B_VOLUME; i++){
        b[i] = utils::getRand(0.2, 4);
    }

	header_print("info", "Calculate reference!");
    vdtype o_ref(O_VOLUME);    
    MM(K, N, a, b, o_ref);

	header_print("info", "Preparing kernel!");
	a.sync_to_device();
	b.sync_to_device();
    o.sync_to_device();

	header_print("info", "Running kernel!");
	time_utils::time_with_unit npu_time = {0.0, "us"};
    for (int i = 0; i < 1; i++) {
		time_utils::time_point start = time_utils::now();
	    npu_instance.run_trace(a.bo(), b.bo(), o.bo(), bo_trace.bo(), app_id);
		time_utils::time_point stop = time_utils::now();
	    npu_time.first += time_utils::duration_us(start, stop).first;
    }
	o.sync_from_device();
	bo_trace.sync_from_device();
	header_print("info", "Finished kernel!");

	header_print("info", "O!");
    utils::print_matrix(o, K);

	header_print("info", "O_ref!");
    utils::print_matrix(o_ref, K);

    npu_instance.write_out_trace(((char *)bo_trace.data()), TRACE_SIZE, "trace.txt");

    bool pass = true;
	if (utils::compare_vectors<dtype>(o, o_ref, 16, -1, 0.1) > 0){
        pass = false;
    }

    if (pass){
        header_print("info", "PASSED ");
    } else {
        header_print("info", "FAILED!");
    }

    utils::print_npu_profile(npu_time, 16.0 * float(K) * float(N));

    return 0;
}

void MM(int K, int N, vdtype& a, vdtype& b, vdtype& o){
	assert((K * N) == b.size());

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0;
            for (int d = 0; d < N; d++) {
                sum += a[i * N + d] * b[d * K + j];
            }
            o[i * K + j] = (dtype)sum;
        }
    }
}
