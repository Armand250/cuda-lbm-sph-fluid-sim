#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

const int WIDTH = 400;
const int HEIGHT = 200;
const int STEPS = 1000;
const float OMEGA = 1.0f / 0.6f;

enum ObstacleType { CIRCLE = 0, PLATE = 1, WING = 2, VCUP = 3, COMBINED = 4, COUNT = 5 };
const std::string OBSTACLE_NAMES[COUNT] = { "Circle", "Vertical_Plate", "Airfoil_Wing", "V_Cup_Trap", "All_Shapes_Combined" };

__constant__ float d_w[9] = {
    4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

__constant__ int d_cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
__constant__ int d_cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };

struct DeviceGridSoA { float* f[9]; };

// Cuda kernels
__global__ void lbm_optimized_pull_kernel(DeviceGridSoA src, DeviceGridSoA dst, const bool* __restrict__ is_solid, int width, int height, float omega) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    if (is_solid[idx]) {
        dst.f[1][idx] = src.f[3][idx]; dst.f[2][idx] = src.f[4][idx]; dst.f[3][idx] = src.f[1][idx]; dst.f[4][idx] = src.f[2][idx];
        dst.f[5][idx] = src.f[7][idx]; dst.f[6][idx] = src.f[8][idx]; dst.f[7][idx] = src.f[5][idx]; dst.f[8][idx] = src.f[6][idx];
        dst.f[0][idx] = src.f[0][idx]; return;
    }
    if (x == 0 || x == width - 1) return;
    float f_local[9];
    for (int i = 0; i < 9; ++i) {
        int src_x = x - d_cx[i]; int src_y = y - d_cy[i];
        src_y = (src_y + height) % height;
        f_local[i] = src.f[i][src_y * width + src_x];
    }
    float rho = 0.0f; float ux = 0.0f; float uy = 0.0f;
    for (int i = 0; i < 9; i++) { rho += f_local[i]; ux += f_local[i] * d_cx[i]; uy += f_local[i] * d_cy[i]; }
    if (rho > 0.0f) { ux /= rho; uy /= rho; }
    for (int i = 0; i < 9; i++) {
        float dot = d_cx[i] * ux + d_cy[i] * uy; float u_sq = ux * ux + uy * uy;
        float feq = d_w[i] * rho * (1.0f + 3.0f * dot + 4.5f * dot * dot - 1.5f * u_sq);
        dst.f[i][idx] = f_local[i] + omega * (feq - f_local[i]);
    }
}

__global__ void lbm_boundaries_kernel(DeviceGridSoA src, DeviceGridSoA dst, int width, int height) {
    int y = blockIdx.x * blockDim.x + threadIdx.x; if (y >= height) return;
    float u_in = 0.12f; int left_idx = y * width + 0;
    float rho_wall = (dst.f[0][left_idx] + dst.f[2][left_idx] + dst.f[4][left_idx] + 2.0f * (dst.f[3][left_idx] + dst.f[6][left_idx] + dst.f[7][left_idx])) / (1.0f - u_in);
    dst.f[1][left_idx] = dst.f[3][left_idx] + (2.0f / 3.0f) * rho_wall * u_in;
    dst.f[5][left_idx] = dst.f[7][left_idx] - 0.5f * (dst.f[2][left_idx] - dst.f[4][left_idx]) + (1.0f / 6.0f) * rho_wall * u_in;
    dst.f[8][left_idx] = dst.f[6][left_idx] + 0.5f * (dst.f[2][left_idx] - dst.f[4][left_idx]) + (1.0f / 6.0f) * rho_wall * u_in;
    int right_idx = y * width + (width - 1); int inner_idx = y * width + (width - 2);
    for (int i = 0; i < 9; i++) { dst.f[i][right_idx] = dst.f[i][inner_idx]; }
}

// Generate the obstacles
void generate_obstacle_mask(bool* h_solid, ObstacleType type) {
    std::fill(h_solid, h_solid + (WIDTH * HEIGHT), false);
    
    if (type == CIRCLE || type == COMBINED) {
        int cx_pos = (type == COMBINED) ? WIDTH / 5 : WIDTH / 4;
        int rad = (type == COMBINED) ? HEIGHT / 12 : HEIGHT / 10;
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                if (std::sqrt((x - cx_pos)*(x - cx_pos) + (y - HEIGHT/2)*(y - HEIGHT/2)) < rad)
                    h_solid[y * WIDTH + x] = true;
            }
        }
    }
    if (type == PLATE || type == COMBINED) {
        int px_pos = (type == COMBINED) ? (2 * WIDTH) / 5 : WIDTH / 4;
        for (int y = 0; y < HEIGHT; y++) {
            if (y >= HEIGHT / 3 && y <= (2 * HEIGHT) / 3)
                h_solid[y * WIDTH + px_pos] = true;
        }
    }
    if (type == WING || type == COMBINED) {
        int wx_pos = (type == COMBINED) ? (3 * WIDTH) / 5 : WIDTH / 4;
        float chord = (type == COMBINED) ? HEIGHT * 0.5f : HEIGHT * 0.8f;
        float thickness = (type == COMBINED) ? HEIGHT * 0.10f : HEIGHT * 0.15f;
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                if (x >= wx_pos && x <= (wx_pos + chord)) {
                    float xc = (x - wx_pos) / chord;
                    float y_half = 5.0f * thickness * (0.2969f * std::sqrt(xc) - 0.1260f * xc - 0.3516f * xc * xc + 0.2843f * xc * xc * xc - 0.1015f * xc * xc * xc * xc);
                    if (std::abs(y - HEIGHT/2) <= y_half) h_solid[y * WIDTH + x] = true;
                }
            }
        }
    }
    if (type == VCUP || type == COMBINED) {
        int base_x = (type == COMBINED) ? (4 * WIDTH) / 5 : WIDTH / 4;
        int apex_x = base_x + ((type == COMBINED) ? HEIGHT / 5 : HEIGHT / 4);
        int apex_y = HEIGHT / 2;
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                if (x >= base_x && x <= apex_x) {
                    int dist_to_apex = apex_x - x;
                    if (std::abs((y - apex_y) - dist_to_apex) <= 1 || std::abs((y - apex_y) + dist_to_apex) <= 1)
                        h_solid[y * WIDTH + x] = true;
                }
            }
        }
    }
}

void save_gpu_density_csv(DeviceGridSoA d_grid, const bool* h_solid, int step, size_t plane_bytes, std::string folder, double pure_time, double pure_mlups) {
    char filename[256]; sprintf(filename, "output/gpu/%s/frame_%04d.csv", folder.c_str(), step);
    std::ofstream file(filename);
    file << "# " << pure_time << " s , " << pure_mlups << " MLUPS\n";
    float* h_temp_f[9]; for (int i = 0; i < 9; i++) { h_temp_f[i] = (float*)malloc(plane_bytes); cudaMemcpy(h_temp_f[i], d_grid.f[i], plane_bytes, cudaMemcpyDeviceToHost); }
    int cx_cpu[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 }; int cy_cpu[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int idx = y * WIDTH + x;
            if (h_solid[idx]) { file << "0.0"; } 
            else {
                float rho = 0.0f; float ux = 0.0f; float uy = 0.0f;
                for (int i = 0; i < 9; i++) { float f_val = h_temp_f[i][idx]; rho += f_val; ux += f_val * cx_cpu[i]; uy += f_val * cy_cpu[i]; }
                if (rho > 0.0f) { ux /= rho; uy /= rho; }
                file << std::sqrt(ux * ux + uy * uy);
            }
            if (x < WIDTH - 1) file << ",";
        }
        file << "\n";
    }
    file.close(); for (int i = 0; i < 9; i++) free(h_temp_f[i]);
}

int main() {
    int grid_size = WIDTH * HEIGHT; size_t plane_bytes = grid_size * sizeof(float);
    int ret = system("mkdir -p output");
    float* h_f[9]; for(int i=0; i<9; ++i) h_f[i] = (float*)malloc(plane_bytes);
    bool* h_solid = (bool*)malloc(grid_size * sizeof(bool));
    DeviceGridSoA d_src, d_dst; bool* d_solid; cudaMalloc(&d_solid, grid_size * sizeof(bool));
    for (int i = 0; i < 9; i++) { cudaMalloc(&d_src.f[i], plane_bytes); cudaMalloc(&d_dst.f[i], plane_bytes); }
    dim3 threadsPerBlock(16, 16); dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dim3 boundaryThreads(128); dim3 boundaryBlocks((HEIGHT + boundaryThreads.x - 1) / boundaryThreads.x);

    for (int scene = 0; scene < COUNT; scene++) {
        std::string name = OBSTACLE_NAMES[scene];
        std::cout << "\n========================================================" << std::endl;
        std::cout << "SCENE [" << scene + 1 << "/" << COUNT << "]: " << name << std::endl;
        std::cout << "========================================================" << std::endl;
        std::string mkdir_cmd = "mkdir -p output/gpu/" + name; int r = system(mkdir_cmd.c_str());
        generate_obstacle_mask(h_solid, (ObstacleType)scene);
        cudaMemcpy(d_solid, h_solid, grid_size * sizeof(bool), cudaMemcpyHostToDevice);

        float u_init = 0.1f; float w_cpu[9] = {4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f}; int cx_cpu[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                int idx = y * WIDTH + x;
                for (int i = 0; i < 9; i++) { float dot = cx_cpu[i] * u_init; h_f[i][idx] = w_cpu[i] * 1.0f * (1.0f + 3.0f * dot + 4.5f * dot * dot - 1.5f * (u_init * u_init)); }
            }
        }
        for (int i = 0; i < 9; i++) cudaMemcpy(d_src.f[i], h_f[i], plane_bytes, cudaMemcpyHostToDevice);

        auto start = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < STEPS; step++) {
            lbm_optimized_pull_kernel<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, d_solid, WIDTH, HEIGHT, OMEGA);
            lbm_boundaries_kernel<<<boundaryBlocks, boundaryThreads>>>(d_src, d_dst, WIDTH, HEIGHT);
            std::swap(d_src, d_dst);
        }
        cudaDeviceSynchronize(); auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start; double mlups = ((double)WIDTH * HEIGHT * STEPS / elapsed.count()) / 1e6;
        std::cout << "Kernel Time: " << elapsed.count() << " seconds (" << mlups << " MLUPS)" << std::endl;

        for (int i = 0; i < 9; i++) cudaMemcpy(d_src.f[i], h_f[i], plane_bytes, cudaMemcpyHostToDevice);
        for (int step = 0; step < STEPS; step++) {
            lbm_optimized_pull_kernel<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, d_solid, WIDTH, HEIGHT, OMEGA);
            lbm_boundaries_kernel<<<boundaryBlocks, boundaryThreads>>>(d_src, d_dst, WIDTH, HEIGHT);
            std::swap(d_src, d_dst);
            if (step % 20 == 0) { save_gpu_density_csv(d_src, h_solid, step, plane_bytes, name, elapsed.count(), mlups); }
        }
    }
    cudaFree(d_solid); free(h_solid); for(int i=0; i<9; ++i) { cudaFree(d_src.f[i]); cudaFree(d_dst.f[i]); free(h_f[i]); }
}