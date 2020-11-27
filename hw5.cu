#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <assert.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

namespace param {
const int n_steps = 200000;
//const double dt = 60;
//const double eps = 1e-3;
//const double G = 6.674e-11;
// double gravity_device_mass(double m0, double t) {
//     return m0 + 0.5 * m0 * fabs(sin(t / 6000));
// }

// double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

void read_input(const char* filename, int& n, int& planet, int& asteroid,
    double*& qx, double*& qy, double*& qz,
    double*& vx, double*& vy, double*& vz,
    double*& m, std::vector<std::string>& type) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx = new double[n];
    qy = new double[n];
    qz = new double[n];
    vx = new double[n];
    vy = new double[n];
    vz = new double[n];
    m = new double[n];
    type.resize(n);
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type[i];
    }
}

void type_redefine(int n, std::vector<std::string>& type, int*& t)
{
    t = new int[n];
    for(int i=0; i<n; ++i){
        if(type[i] == "device")t[i] = 1;
        else t[i] = 0;
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

__global__
void run_step(int step,const int n, double* qx, double* qy, double* qz,
    double* vx, double* vy, double* vz, double* m, int* type) {
    // compute accelerations
    __shared__ double ax, ay, az;
    int thr = threadIdx.x;
    int blo = blockIdx.x;
    if(thr == 0){
        ax = 0;
        ay = 0;
        az = 0;
    }
    if (thr != blo){
        double mj = m[thr];
        if (type[thr] == 1) {
            mj = mj * (1 + 0.5 * fabs(sin((double)step / (double)100)));
        }
        double dx = qx[thr] - qx[blo];
        double dy = qy[thr] - qy[blo];
        double dz = qz[thr] - qz[blo];
        double dist3 =
            pow(dx * dx + dy * dy + dz * dz + 1e-6, 1.5)/ mj / 6.674e-11;
        atomicAdd(&ax, dx / dist3);
        atomicAdd(&ay, dy / dist3);
        atomicAdd(&az, dz / dist3);
    }
    __syncthreads();

    // update velocities
    if(thr == 0){
        vx[blo] += ax * 60;
        vy[blo] += ay * 60;
        vz[blo] += az * 60;
    }

    // update positions
    if(thr == 0){
        qx[blo] += vx[blo] * 60;
        qy[blo] += vy[blo] * 60;
        qz[blo] += vz[blo] * 60;
    }
}

__global__
void set_zero(double* m, int* type){
    if(type[threadIdx.x]==1)m[threadIdx.x] = 0;
}

__global__
void define_min(double* qx, double* qy, double* qz, int planet, int asteroid, double* min_dist){
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    *min_dist = min(*min_dist, sqrt(dx * dx + dy * dy + dz * dz));
}

__global__
void hit_steps(int step, double* qx, double* qy, double* qz, int planet, int asteroid, int* hit_time_step){
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    if (dx * dx + dy * dy + dz * dz < 1e14) *hit_time_step = step;
}

__global__
void kill_device(int step, double* qx, double* qy, double* qz,
    int i, int planet, int asteroid, double* travel_dist,
    double* crash_device_step, bool* once, double* m){
    if(*once){
        *travel_dist += 6e7;
        double tx = qx[planet] - qx[i];
        double ty = qy[planet] - qy[i];
        double tz = qz[planet] - qz[i];
        double total_dist = sqrt(tx * tx + ty * ty + tz * tz);
        if(*travel_dist > total_dist){
            m[i] = 0;
            *crash_device_step = step;
            *once = false;
        }
    }
}

__global__
void init_param(const int n, double* o_qx, double* o_qy, double* o_qz,
    double* o_vx, double* o_vy, double* o_vz, double* o_m, int* o_type,
    double* o_min_dist, int* o_hit_time_step, int* o_hit_time_step_1,
    double* o_travel_dist, double* o_crash_device_step, bool* o_once,
    double* d_qx, double* d_qy, double* d_qz,
    double* d_vx, double* d_vy, double* d_vz, double* d_m, int* d_type,
    double* d_min_dist, int* d_hit_time_step, int* d_hit_time_step_1,
    double* d_travel_dist, double* d_crash_device_step, bool* d_once){
    d_qx[threadIdx.x] = o_qx[threadIdx.x];
    d_qy[threadIdx.x] = o_qy[threadIdx.x];
    d_qz[threadIdx.x] = o_qz[threadIdx.x];
    d_vx[threadIdx.x] = o_vx[threadIdx.x];
    d_vy[threadIdx.x] = o_vy[threadIdx.x];
    d_vz[threadIdx.x] = o_vz[threadIdx.x];
    d_m[threadIdx.x] = o_m[threadIdx.x];
    d_type[threadIdx.x] = o_type[threadIdx.x];
    if(threadIdx.x == 0){
        *d_min_dist = *o_min_dist;
        *d_hit_time_step = *o_hit_time_step;
        *d_hit_time_step_1 = *o_hit_time_step_1;
        *d_travel_dist = *o_travel_dist;
        *d_crash_device_step = *o_crash_device_step;
        *d_once = *o_once;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    int n, planet, asteroid;
    double *qx, *qy, *qz, *vx, *vy, *vz, *m;
    int *t;
    double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz, *d_m;
    int *d_type;
    double *o_qx, *o_qy, *o_qz, *o_vx, *o_vy, *o_vz, *o_m;
    int *o_type;
    std::vector<std::string> type;
    double min_dist, *d_min_dist, *o_min_dist;
    int hit_time_step, *d_hit_time_step, *o_hit_time_step;
    int hit_time_step_1, *d_hit_time_step_1, *o_hit_time_step_1;
    double crash_device_step, *d_crash_device_step, *o_crash_device_step;
    double travel_dist, *d_travel_dist, *o_travel_dist;
    bool once, *d_once, *o_once;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    size_t size_d = n * sizeof(double);
    size_t size_i = n * sizeof(int);

    cudaMalloc(&d_qx, size_d);
    cudaMalloc(&d_qy, size_d);
    cudaMalloc(&d_qz, size_d);
    cudaMalloc(&d_vx, size_d);
    cudaMalloc(&d_vy, size_d);
    cudaMalloc(&d_vz, size_d);
    cudaMalloc(&d_m, size_d);
    cudaMalloc(&d_type, size_i);
    cudaMalloc(&o_qx, size_d);
    cudaMalloc(&o_qy, size_d);
    cudaMalloc(&o_qz, size_d);
    cudaMalloc(&o_vx, size_d);
    cudaMalloc(&o_vy, size_d);
    cudaMalloc(&o_vz, size_d);
    cudaMalloc(&o_m, size_d);
    cudaMalloc(&o_type, size_i);
    cudaMalloc(&d_min_dist, sizeof(double));
    cudaMalloc(&d_hit_time_step, sizeof(int));
    cudaMalloc(&d_hit_time_step_1, sizeof(int));
    cudaMalloc(&d_crash_device_step, sizeof(double));
    cudaMalloc(&d_travel_dist, sizeof(double));
    cudaMalloc(&d_once, sizeof(bool));
    cudaMalloc(&o_min_dist, sizeof(double));
    cudaMalloc(&o_hit_time_step, sizeof(int));
    cudaMalloc(&o_hit_time_step_1, sizeof(int));
    cudaMalloc(&o_crash_device_step, sizeof(double));
    cudaMalloc(&o_travel_dist, sizeof(double));
    cudaMalloc(&o_once, sizeof(bool));
    // Problem 1
    type_redefine(n, type, t);
    min_dist = std::numeric_limits<double>::infinity();
    hit_time_step = -2;
    hit_time_step_1 = -2;
    crash_device_step = 0;
    travel_dist = 0;
    once = true;
    cudaMemcpy(o_qx, qx, size_d, cudaMemcpyHostToDevice);
    cudaMemcpy(o_qy, qy, size_d, cudaMemcpyHostToDevice);
    cudaMemcpy(o_qz, qz, size_d, cudaMemcpyHostToDevice);
    cudaMemcpy(o_vx, vx, size_d, cudaMemcpyHostToDevice);
    cudaMemcpy(o_vy, vy, size_d, cudaMemcpyHostToDevice);
    cudaMemcpy(o_vz, vz, size_d, cudaMemcpyHostToDevice);
    cudaMemcpy(o_m, m, size_d, cudaMemcpyHostToDevice);
    cudaMemcpy(o_type, t, size_i, cudaMemcpyHostToDevice);
    cudaMemcpy(o_min_dist, &min_dist, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(o_hit_time_step, &hit_time_step, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(o_hit_time_step_1, &hit_time_step_1, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(o_crash_device_step, &crash_device_step, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(o_travel_dist, &travel_dist, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(o_once, &once, sizeof(bool), cudaMemcpyHostToDevice);

    init_param<<<1, n>>>(n, o_qx, o_qy, o_qz,o_vx, o_vy, o_vz, o_m, o_type,
                            o_min_dist, o_hit_time_step, o_hit_time_step_1,
                            o_travel_dist, o_crash_device_step, o_once,
                            d_qx, d_qy, d_qz,d_vx, d_vy, d_vz, d_m, d_type,
                            d_min_dist, d_hit_time_step, d_hit_time_step_1,
                            d_travel_dist, d_crash_device_step, d_once);
    set_zero<<<1, n>>>(d_m, d_type);

    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step<<<n, n>>>(step, n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_m, d_type);
        }
        define_min<<<1, 1>>>(d_qx, d_qy, d_qz, planet, asteroid, d_min_dist);
    }
    cudaMemcpy(&min_dist, d_min_dist, sizeof(double), cudaMemcpyDeviceToHost);

    // Problem 2
    init_param<<<1, n>>>(n, o_qx, o_qy, o_qz,o_vx, o_vy, o_vz, o_m, o_type,
        o_min_dist, o_hit_time_step, o_hit_time_step_1,
        o_travel_dist, o_crash_device_step, o_once,
        d_qx, d_qy, d_qz,d_vx, d_vy, d_vz, d_m, d_type,
        d_min_dist, d_hit_time_step, d_hit_time_step_1,
        d_travel_dist, d_crash_device_step, d_once);

    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step<<<n, n>>>(step, n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_m, d_type);
        }
        hit_steps<<<1, 1>>>(step, d_qx, d_qy, d_qz, planet, asteroid, d_hit_time_step);
    }
    cudaMemcpy(&hit_time_step, d_hit_time_step, sizeof(int), cudaMemcpyDeviceToHost);
    // Problem 3
    // TODO
    int gravity_device_id = -1;
    double missile_cost = -1;
    if(hit_time_step != -2){
        for (int i=0; i<n; ++i) {
            if (t[i] != 1)continue;
            init_param<<<1, n>>>(n, o_qx, o_qy, o_qz,o_vx, o_vy, o_vz, o_m, o_type,
                o_min_dist, o_hit_time_step, o_hit_time_step_1,
                o_travel_dist, o_crash_device_step, o_once,
                d_qx, d_qy, d_qz,d_vx, d_vy, d_vz, d_m, d_type,
                d_min_dist, d_hit_time_step, d_hit_time_step_1,
                d_travel_dist, d_crash_device_step, d_once);
            for (int step = 0; step <= param::n_steps; step++) {
                if (step > 0) {
                    run_step<<<n, n>>>(step, n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_m, d_type);
                }
                kill_device<<<1, 1>>>(step, d_qx, d_qy, d_qz, i, planet, asteroid, d_travel_dist, d_crash_device_step, d_once, d_m);
                hit_steps<<<1, 1>>>(step, d_qx, d_qy, d_qz, planet, asteroid, d_hit_time_step_1);
            }
            cudaMemcpy(&hit_time_step_1, d_hit_time_step_1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&crash_device_step, d_crash_device_step, sizeof(double), cudaMemcpyDeviceToHost);
            if(hit_time_step_1 == -2){
                double temp = (100 + crash_device_step * 60) * 1e3;
                if(missile_cost < 0){
                    missile_cost = temp;
                    gravity_device_id = i;
                }else if(missile_cost > temp){
                    missile_cost = temp;
                    gravity_device_id = i;
                }
            }
        }
    }
    if(gravity_device_id == -1)missile_cost = 0;

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
    delete[] qx;
    delete[] qy;
    delete[] qz;
    delete[] vx;
    delete[] vy;
    delete[] vz;
    delete[] m;
    delete[] t;
    cudaDeviceReset();
}