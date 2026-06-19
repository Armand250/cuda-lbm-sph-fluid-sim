#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <string>

// Simulation parameters
const int WIDTH = 400;
const int HEIGHT = 200;
const int STEPS = 1000;      
const float OMEGA = 1.0f / 0.6f; 

// Multi-Scene enumerations matching GPU
enum ObstacleType { CIRCLE = 0, PLATE = 1, WING = 2, VCUP = 3, COMBINED = 4, COUNT = 5 };
const std::string OBSTACLE_NAMES[COUNT] = { "Circle", "Vertical_Plate", "Airfoil_Wing", "V_Cup_Trap", "All_Shapes_Combined" };

// D2Q9 constants
const float w[9] = {
    4.0f/9.0f, 
    1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};
const int cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
const int cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };

// Global Structure of Arrays (SoA) Grids (9 flat vectors for 9 directions)
std::vector<std::vector<float>> grid_src(9, std::vector<float>(WIDTH * HEIGHT));
std::vector<std::vector<float>> grid_dst(9, std::vector<float>(WIDTH * HEIGHT));
std::vector<bool> is_solid(WIDTH * HEIGHT, false);

// Generate solid mask geometries
void generate_obstacle_mask(ObstacleType type) {
    std::fill(is_solid.begin(), is_solid.end(), false);
    
    if (type == CIRCLE || type == COMBINED) {
        int cx_pos = (type == COMBINED) ? WIDTH / 5 : WIDTH / 4;
        int rad = (type == COMBINED) ? HEIGHT / 12 : HEIGHT / 10;
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                if (std::sqrt((x - cx_pos)*(x - cx_pos) + (y - HEIGHT/2)*(y - HEIGHT/2)) < rad)
                    is_solid[y * WIDTH + x] = true;
            }
        }
    }
    if (type == PLATE || type == COMBINED) {
        int px_pos = (type == COMBINED) ? (2 * WIDTH) / 5 : WIDTH / 4;
        for (int y = 0; y < HEIGHT; y++) {
            if (y >= HEIGHT / 3 && y <= (2 * HEIGHT) / 3)
                is_solid[y * WIDTH + px_pos] = true;
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
                    if (std::abs(y - HEIGHT/2) <= y_half) is_solid[y * WIDTH + x] = true;
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
                        is_solid[y * WIDTH + x] = true;
                }
            }
        }
    }
}

// Initialize background
void reset_fluid_field() {
    float rho_baseline = 1.0f;
    float u_init = 0.1f; 
    float v_init = 0.0f;

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int idx = y * WIDTH + x;
            for (int i = 0; i < 9; i++) {
                float dot = cx[i] * u_init + cy[i] * v_init;
                float u_sq = u_init * u_init + v_init * v_init;
                grid_src[i][idx] = w[i] * rho_baseline * (1.0f + 3.0f * dot + 4.5f * dot * dot - 1.5f * u_sq);
            }
        }
    }
}

void cpu_lbm_step_soa() {
    // Phase 1: Collision (Local relaxation)
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int idx = y * WIDTH + x;

            // Solid obstacle handling
            if (is_solid[idx]) {
                grid_dst[1][idx] = grid_src[3][idx];
                grid_dst[2][idx] = grid_src[4][idx];
                grid_dst[3][idx] = grid_src[1][idx];
                grid_dst[4][idx] = grid_src[2][idx];
                grid_dst[5][idx] = grid_src[7][idx];
                grid_dst[6][idx] = grid_src[8][idx];
                grid_dst[7][idx] = grid_src[5][idx];
                grid_dst[8][idx] = grid_src[6][idx];
                grid_dst[0][idx] = grid_src[0][idx];
                continue; 
            }

            if (x == 0 || x == WIDTH - 1) continue;

            float f_local[9];
            for (int i = 0; i < 9; ++i) {
                int src_x = x - cx[i];
                int src_y = y - cy[i];
                
                src_y = (src_y + HEIGHT) % HEIGHT;
                
                f_local[i] = grid_src[i][src_y * WIDTH + src_x];
            }

            float rho = 0.0f; float ux = 0.0f; float uy = 0.0f;
            for (int i = 0; i < 9; i++) {
                rho += f_local[i]; ux += f_local[i] * cx[i]; uy += f_local[i] * cy[i];
            }
            if (rho > 0.0f) { ux /= rho; uy /= rho; }

            // Apply BGK relaxation
            for (int i = 0; i < 9; i++) {
                float dot = cx[i] * ux + cy[i] * uy;
                float u_sq = ux * ux + uy * uy;
                float feq = w[i] * rho * (1.0f + 3.0f * dot + 4.5f * dot * dot - 1.5f * u_sq);
                
                grid_dst[i][idx] = f_local[i] + OMEGA * (feq - f_local[i]);
            }
        }
    }

    // Phase 2: Streaming phase (Advection)
    float u_in = 0.12f;
    for (int y = 0; y < HEIGHT; y++) {

        // Left Wall: Zou-He velocity inflow
        int left_idx = y * WIDTH + 0;
        float rho_wall = (grid_dst[0][left_idx] + grid_dst[2][left_idx] + grid_dst[4][left_idx] + 
                          2.0f * (grid_dst[3][left_idx] + grid_dst[6][left_idx] + grid_dst[7][left_idx])) / (1.0f - u_in);
        
        grid_dst[1][left_idx] = grid_dst[3][left_idx] + (2.0f / 3.0f) * rho_wall * u_in;
        grid_dst[5][left_idx] = grid_dst[7][left_idx] - 0.5f * (grid_dst[2][left_idx] - grid_dst[4][left_idx]) + (1.0f / 6.0f) * rho_wall * u_in;
        grid_dst[8][left_idx] = grid_dst[6][left_idx] + 0.5f * (grid_dst[2][left_idx] - grid_dst[4][left_idx]) + (1.0f / 6.0f) * rho_wall * u_in;

        // Right Wall: Convective outflow
        int right_idx = y * WIDTH + (WIDTH - 1);
        int inner_idx = y * WIDTH + (WIDTH - 2);
        for (int i = 0; i < 9; i++) {
            grid_dst[i][right_idx] = grid_dst[i][inner_idx];
        }
    }
}

void save_density_csv(int step, std::string folder, double pure_time, double pure_mlups) {
    char filename[256]; sprintf(filename, "output/cpu/%s/frame_%04d.csv", folder.c_str(), step);
    std::ofstream file(filename);

    if (!file.is_open()) {
        return;
    }

    file << "# " << pure_time << " s , " << pure_mlups << " MLUPS\n";

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int idx = y * WIDTH + x;
            if (is_solid[idx]) {
                file << "0.0"; 
            } else {
                float rho = 0.0f; float ux = 0.0f; float uy = 0.0f;
                for (int i = 0; i < 9; i++) { float f_val = grid_src[i][idx]; rho += f_val; ux += f_val * cx[i]; uy += f_val * cy[i]; }
                if (rho > 0.0f) { ux /= rho; uy /= rho; }
                file << std::sqrt(ux * ux + uy * uy);
            }
            if (x < WIDTH - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

int main() {
    int ret = system("mkdir -p output"); 
    for (int scene = 0; scene < COUNT; scene++) {
        std::string name = OBSTACLE_NAMES[scene];
        std::cout << "\n========================================================" << std::endl;
        std::cout << "SCENE [" << scene + 1 << "/" << COUNT << "]: " << name << std::endl;
        std::cout << "========================================================" << std::endl;
        std::string mkdir_cmd = "mkdir -p output/cpu/" + name; int r = system(mkdir_cmd.c_str());
        generate_obstacle_mask((ObstacleType)scene); reset_fluid_field();
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < STEPS; step++) { cpu_lbm_step_soa(); std::swap(grid_src, grid_dst); }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        double total_lattice_updates = static_cast<double>(WIDTH) * HEIGHT * STEPS; double mlups = (total_lattice_updates / elapsed.count()) / 1e6;
        std::cout << "CPU Execution Time: " << elapsed.count() << " seconds (" << mlups << " MLUPS)" << std::endl;
        reset_fluid_field();
        for (int step = 0; step < STEPS; step++) {
            cpu_lbm_step_soa(); std::swap(grid_src, grid_dst);
            if (step % 20 == 0) { save_density_csv(step, name, elapsed.count(), mlups); }
        }
    }
}