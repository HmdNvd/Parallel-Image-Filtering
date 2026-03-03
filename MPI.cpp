#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

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

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width, height, channels;
    vector<int> grayImage;

    double total_start, total_end;
    double load_time, convert_time, scatter_time, filter_time, gather_time;

    // -------------------------------
    // MASTER: Load image
    // -------------------------------
    if (rank == 0) {
        cout << "=================================================\n";
        cout << "       MPI Image Processing Pipeline\n";
        cout << "=================================================\n";
        cout << "Number of MPI processes: " << size << "\n";
        cout << "-------------------------------------------------\n";

        total_start = MPI_Wtime();

        string filename;
        cout << "Enter image name: ";
        getline(cin >> ws, filename);

        double load_start = MPI_Wtime();
        unsigned char* img = stbi_load(
            filename.c_str(),
            &width, &height, &channels, 3
        );
        channels = 3;

        if (!img) {
            cout << "Error: Failed to load image\n";
            cout << "Reason: " << stbi_failure_reason() << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        double load_end = MPI_Wtime();
        load_time = load_end - load_start;

        cout << "Image loaded: " << width << "x" << height
             << " | Channels: " << channels << endl;
        cout << "Image load time: " << load_time << " seconds\n";
        cout << "-------------------------------------------------\n";

        // Convert to grayscale
        double convert_start = MPI_Wtime();
        grayImage.resize(width * height);

        for (int i = 0; i < width * height; i++) {
            int r = img[i * 3];
            int g = img[i * 3 + 1];
            int b = img[i * 3 + 2];
            grayImage[i] = static_cast<int>(0.299*r + 0.587*g + 0.114*b);
        }

        stbi_image_free(img);
        double convert_end = MPI_Wtime();
        convert_time = convert_end - convert_start;

        cout << "Grayscale conversion time: " << convert_time << " seconds\n";
        cout << "-------------------------------------------------\n";
    }

    // Broadcast image dimensions
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rowsPerProc = height / size;
    int startRow = rank * rowsPerProc;
    int endRow = (rank == size - 1) ? height : startRow + rowsPerProc;

    int localRows = endRow - startRow;

    vector<int> localGray((localRows + 2) * width);
    vector<int> localLap(localRows * width);
    vector<int> localMed(localRows * width);

    // -------------------------------
    // Scatter image rows with halo
    // -------------------------------
    double scatter_start = MPI_Wtime();

    if (rank == 0) {
        cout << "Distributing image rows to processes...\n";

        for (int p = 1; p < size; p++) {
            int s = p * rowsPerProc;
            int e = (p == size - 1) ? height : s + rowsPerProc;
            int pRows = e - s;

            // Send halo rows: for first process we need row above (s-1), for last we need row below (e)
            int sendStart = (s > 0) ? (s - 1) : 0;
            int sendEnd = (e < height) ? e : height - 1;
            int sendSize = (sendEnd - sendStart + 1) * width;

            MPI_Send(&grayImage[sendStart * width],
                     sendSize,
                     MPI_INT, p, 0, MPI_COMM_WORLD);
        }

        // For rank 0, copy its portion with bottom halo
        int copySize = (localRows + 1) * width;
        if (localRows + 1 > height) copySize = height * width;
        copy(grayImage.begin(),
             grayImage.begin() + copySize,
             localGray.begin());
    } else {
        MPI_Recv(localGray.data(),
                 (localRows + 2) * width,
                 MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double scatter_end = MPI_Wtime();
    scatter_time = scatter_end - scatter_start;

    if (rank == 0) {
        cout << "Data distribution time: " << scatter_time << " seconds\n";
        cout << "-------------------------------------------------\n";
    }

    // -------------------------------
    // Apply Filters
    // -------------------------------
    double filter_start = MPI_Wtime();

    // Laplacian Filter
    int effectiveRows = (rank == size - 1 && localRows + 1 >= height) ? height - 1 : localRows;

    for (int y = 1; y <= effectiveRows && y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;

            // Ensure we don't go out of bounds
            if (y + 1 <= localRows || rank == size - 1) {
                int val =
                    -localGray[(y - 1) * width + x] +
                    -localGray[(y + 1) * width + x] +
                    -localGray[y * width + (x - 1)] +
                    -localGray[y * width + (x + 1)] +
                     4 * localGray[idx];

                localLap[(y - 1) * width + x] = clamp(abs(val));
            }
        }
    }

    // Median Filter
    int window[9];
    for (int y = 1; y <= effectiveRows && y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int k = 0;

            // Ensure we don't go out of bounds
            if (y + 1 <= localRows || rank == size - 1) {
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                        window[k++] = localGray[(y + dy) * width + (x + dx)];

                sort(window, window + 9);
                localMed[(y - 1) * width + x] = window[4];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double filter_end = MPI_Wtime();
    filter_time = filter_end - filter_start;

    if (rank == 0) {
        cout << "Applying Laplacian and Median filters...\n";
        cout << "Filter processing time: " << filter_time << " seconds\n";
        cout << "-------------------------------------------------\n";
    }

    // -------------------------------
    // Gather results
    // -------------------------------
    double gather_start = MPI_Wtime();

    vector<int> lapOut, medOut;
    if (rank == 0) {
        lapOut.resize(width * height);
        medOut.resize(width * height);
        cout << "Gathering results from all processes...\n";
    }

    MPI_Gather(localLap.data(), localRows * width, MPI_INT,
               lapOut.data(), localRows * width, MPI_INT,
               0, MPI_COMM_WORLD);

    MPI_Gather(localMed.data(), localRows * width, MPI_INT,
               medOut.data(), localRows * width, MPI_INT,
               0, MPI_COMM_WORLD);

    double gather_end = MPI_Wtime();
    gather_time = gather_end - gather_start;

    if (rank == 0) {
        cout << "Data gathering time: " << gather_time << " seconds\n";
        cout << "-------------------------------------------------\n";
    }

    // -------------------------------
    // Save output
    // -------------------------------
    if (rank == 0) {
        vector<unsigned char> lapImg(width * height);
        vector<unsigned char> medImg(width * height);

        for (int i = 0; i < width * height; i++) {
            lapImg[i] = (unsigned char)lapOut[i];
            medImg[i] = (unsigned char)medOut[i];
        }

        static int run_id = 0;
        string lapName = "mpi_laplacian_" + to_string(run_id) + ".png";
        string medName = "mpi_median_" + to_string(run_id) + ".png";
        run_id++;

        stbi_write_png(lapName.c_str(), width, height, 1,
                       lapImg.data(), width);
        stbi_write_png(medName.c_str(), width, height, 1,
                       medImg.data(), width);

        total_end = MPI_Wtime();
        double total_time = total_end - total_start;
        double parallel_time = filter_time;

        cout << "Output images saved successfully:\n";
        cout << "  - " << lapName << "\n";
        cout << "  - " << medName << "\n";
        cout << "=================================================\n";
        cout << "               Timing Summary\n";
        cout << "=================================================\n";
        cout << "Image loading:          " << load_time << " seconds\n";
        cout << "Grayscale conversion:   " << convert_time << " seconds\n";
        cout << "Data distribution:      " << scatter_time << " seconds\n";
        cout << "Parallel filter exec:   " << parallel_time << " seconds\n";
        cout << "Data gathering:         " << gather_time << " seconds\n";
        cout << "-------------------------------------------------\n";
        cout << "Total execution time:   " << total_time << " seconds\n";
        cout << "=================================================\n";
    }

    MPI_Finalize();
    return 0;
}
