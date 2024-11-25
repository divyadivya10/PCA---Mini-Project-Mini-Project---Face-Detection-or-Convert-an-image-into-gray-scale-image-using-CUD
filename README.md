# PCA---Mini-Project-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD
### Mini Project - Face Detection or Convert an image into gray scale image using CUDA GPU programming
## NAME:DIVYA R
## REG.NO:212222040040
### 1. Grayscale Image Conversion using CUDA
Grayscale conversion is a straightforward image processing task that can benefit from parallelization. Each pixel's intensity in a grayscale image is derived from the RGB values of the pixel, and this calculation can be done in parallel across all pixels of the image using CUDA.

Steps:
Load the Image: Use a library like OpenCV to load the image into memory.
Allocate Memory on the GPU: Use cudaMalloc to allocate memory for the image.
Launch CUDA Kernel: Write a kernel function that performs grayscale conversion for each pixel.
Synchronize and Copy Results: Transfer the result back to the CPU and display/save the image.
Here’s an example code to convert an image into grayscale using CUDA:
#include <stdio.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
### CODE
// CUDA Kernel for Grayscale Conversion
__global__ void rgbToGray(unsigned char *rgb, unsigned char *gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * 3;

    if (x < width && y < height) {
        // Grayscale conversion formula: Y = 0.299R + 0.587G + 0.114B
        unsigned char r = rgb[idx];
        unsigned char g = rgb[idx + 1];
        unsigned char b = rgb[idx + 2];
        gray[y * width + x] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

int main() {
    // Load the image using OpenCV
    cv::Mat img = cv::imread("input_image.jpg", cv::IMREAD_COLOR);
    if (img.empty()) {
        printf("Error loading image.\n");
        return -1;
    }

    int width = img.cols;
    int height = img.rows;

    // Allocate host memory for grayscale image
    cv::Mat gray_img(height, width, CV_8UC1);

    // Allocate device memory
    unsigned char *d_rgb, *d_gray;
    size_t rgb_size = width * height * 3 * sizeof(unsigned char);
    size_t gray_size = width * height * sizeof(unsigned char);
    cudaMalloc((void **)&d_rgb, rgb_size);
    cudaMalloc((void **)&d_gray, gray_size);

    // Copy data from host to device
    cudaMemcpy(d_rgb, img.data, rgb_size, cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Launch the kernel
    rgbToGray<<<grid, block>>>(d_rgb, d_gray, width, height);
    cudaDeviceSynchronize();

    // Copy grayscale image back to host
    cudaMemcpy(gray_img.data, d_gray, gray_size, cudaMemcpyDeviceToHost);

    // Save the grayscale image
    cv::imwrite("output_gray_image.jpg", gray_img);

    // Free device memory
    cudaFree(d_rgb);
    cudaFree(d_gray);

    return 0;
}
### 2. Face Detection using CUDA and OpenCV
For face detection, CUDA-accelerated libraries like OpenCV’s CUDA modules can be used. OpenCV provides the CascadeClassifier that can run on CUDA for real-time face detection.

Steps:
Load the Image: Read the image using OpenCV.
Convert to Grayscale: Most face detection algorithms work better on grayscale images.
Load the Haar Cascade for Face Detection: OpenCV has pre-trained models for face detection.
Perform Detection using CUDA: Use CUDA-accelerated face detection available in OpenCV's cv::cuda::CascadeClassifier.
Here’s an example code for CUDA-based face detection using OpenCV:
#include <opencv2/opencv.hpp>
#include <opencv2/cudaobjdetect.hpp>
### CODE
int main() {
    // Load the image
    cv::Mat img = cv::imread("input_image.jpg", cv::IMREAD_COLOR);
    if (img.empty()) {
        printf("Error loading image.\n");
        return -1;
    }

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Load the face detection model (Haar Cascade)
    cv::Ptr<cv::cuda::CascadeClassifier> face_cascade = 
        cv::cuda::CascadeClassifier::create("haarcascade_frontalface_alt.xml");

    // Transfer the grayscale image to the GPU
    cv::cuda::GpuMat d_gray;
    d_gray.upload(gray);

    // Detect faces
    cv::cuda::GpuMat faces_buf;
    face_cascade->detectMultiScale(d_gray, faces_buf);

    // Convert the result back to CPU
    std::vector<cv::Rect> faces;
    face_cascade->convert(faces_buf, faces);

    // Draw rectangles around detected faces
    for (size_t i = 0; i < faces.size(); i++) {
        cv::rectangle(img, faces[i], cv::Scalar(255, 0, 0), 2);
    }

    // Display the output
    cv::imshow("Detected Faces", img);
    cv::waitKey(0);

    return 0;
}
### Key Takeaways:
Grayscale Conversion: Each pixel’s RGB values are processed in parallel, using CUDA kernels for fast conversion.
Face Detection: Using CUDA-accelerated OpenCV, we can perform real-time face detection by leveraging the GPU.
These examples demonstrate how CUDA can be utilized for efficient image processing tasks.
