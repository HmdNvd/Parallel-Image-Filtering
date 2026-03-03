#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
using namespace chrono;

// Clamp helper
int clamp(int v, int minV = 0, int maxV = 255) {
    if (v < minV) return minV;
    if (v > maxV) return maxV;
    return v;
}

int main() {

    // -------------------------------
    // OpenMP Information
    // -------------------------------
    cout << "=================================================\n";
    cout << "       OpenMP Image Processing Pipeline\n";
    cout << "=================================================\n";

    #ifdef _OPENMP
        int num_threads = omp_get_max_threads();
        cout << "OpenMP enabled\n";
        cout << "Number of threads available: " << num_threads << "\n";
    #else
        cout << "OpenMP not enabled\n";
    #endif

    cout << "-------------------------------------------------\n";

    int width, height, channels;

    auto total_start = high_resolution_clock::now();

    // -------------------------------
    // Step 1: Load Image
    // -------------------------------
    string filename;
    cout << "Enter image name: ";
    getline(cin >> ws, filename);

    auto load_start = high_resolution_clock::now();

    unsigned char* inputImage = stbi_load(
        filename.c_str(),
        &width, &height, &channels, 3
    );
    channels = 3;

    if (!inputImage) {
        cout << "Error: Could not load image!" << endl;
        cout << "Reason: " << stbi_failure_reason() << endl;
        return -1;
    }

    auto load_end = high_resolution_clock::now();
    double load_time = duration<double>(load_end - load_start).count();

    cout << "Image loaded: " << width << "x" << height
         << " | Channels: " << channels << endl;
    cout << "Image load time: " << load_time << " seconds\n";
    cout << "-------------------------------------------------\n";

    // -------------------------------
    // Step 2: Convert to Grayscale
    // -------------------------------
    auto convert_start = high_resolution_clock::now();

    vector<int> grayImage(width * height);

    #pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        int r = inputImage[i * channels + 0];
        int g = inputImage[i * channels + 1];
        int b = inputImage[i * channels + 2];

        grayImage[i] = static_cast<int>(0.299*r + 0.587*g + 0.114*b);
    }

    stbi_image_free(inputImage);

    auto convert_end = high_resolution_clock::now();
    double convert_time = duration<double>(convert_end - convert_start).count();

    cout << "Grayscale conversion time: " << convert_time << " seconds\n";
    cout << "-------------------------------------------------\n";

    vector<int> laplacianOut(width * height, 0);
    vector<int> medianOut(width * height, 0);

    // -------------------------------
    // Step 3: Apply Filters
    // -------------------------------
    cout << "Applying Laplacian and Median filters...\n";
    auto filter_start = high_resolution_clock::now();

    // -------------------------------
    // Step 4: Laplacian Filter (Parallelized)
    // -------------------------------
    #pragma omp parallel for
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;

            int value =
                -grayImage[(y - 1) * width + x] +
                -grayImage[(y + 1) * width + x] +
                -grayImage[y * width + (x - 1)] +
                -grayImage[y * width + (x + 1)] +
                 4 * grayImage[idx];

            laplacianOut[idx] = clamp(abs(value));
        }
    }

    // -------------------------------
    // Step 5: Median Filter (3×3) (Parallelized)
    // -------------------------------
    int window[9];

    #pragma omp parallel for private(window)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int k = 0;

            // Fill the window with surrounding pixel values
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    window[k++] =
                        grayImage[(y + dy) * width + (x + dx)];
                }
            }

            // Sort the window and take the median
            sort(window, window + 9);
            medianOut[y * width + x] = window[4];
        }
    }

    // -------------------------------
    // Step 6: Stop Filter Timing
    // -------------------------------
    auto filter_end = high_resolution_clock::now();
    double filter_time = duration<double>(filter_end - filter_start).count();

    cout << "Filter processing time: " << filter_time << " seconds\n";
    cout << "-------------------------------------------------\n";

    // -------------------------------
    // Step 7: Save Output Images
    // -------------------------------
    vector<unsigned char> laplacianImg(width * height);
    vector<unsigned char> medianImg(width * height);

    for (int i = 0; i < width * height; i++) {
        laplacianImg[i] = static_cast<unsigned char>(laplacianOut[i]);
        medianImg[i]    = static_cast<unsigned char>(medianOut[i]);
    }

    static int run_id = 0;

    string lapName = "openmp_laplacian_" + to_string(run_id) + ".png";
    string medName = "openmp_median_" + to_string(run_id) + ".png";

    run_id++;

    stbi_write_png(lapName.c_str(), width, height, 1,
               laplacianImg.data(), width);

    stbi_write_png(medName.c_str(), width, height, 1,
               medianImg.data(), width);

    auto total_end = high_resolution_clock::now();
    double total_time = duration<double>(total_end - total_start).count();

    cout << "Output images saved successfully:\n";
    cout << "  - " << lapName << "\n";
    cout << "  - " << medName << "\n";
    cout << "=================================================\n";
    cout << "               Timing Summary\n";
    cout << "=================================================\n";
    cout << "Image loading:          " << load_time << " seconds\n";
    cout << "Grayscale conversion:   " << convert_time << " seconds\n";
    cout << "Parallel filter exec:   " << filter_time << " seconds\n";
    cout << "-------------------------------------------------\n";
    cout << "Total execution time:   " << total_time << " seconds\n";
    cout << "=================================================\n";

    return 0;
}
