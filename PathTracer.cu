#pragma once

#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <vector>

#include "Math3D.cuh"

//Defining CUDA Utility function
#define cudaCheckErrors(x) cudaCheckError(x, #x, __FILE__, __LINE__)

//Display properties
#define WIDTH 1280
#define HEIGHT 720

//Path tracing properties
#define MAX_DEPTH 10
#define SAMPLES 1

//World properties
#define NUM_OBJECTS 3
#define FOV 90
#define ZNEAR 0.01
#define ZFAR 1000

//Utilities
#define PI 3.1415926535897932384626433832795

//CUDA Utility functions
void cudaCheckError(cudaError err, char const *const func, const char *const file, int const line)
{
	if (err != cudaSuccess)
	{
		std::cerr << "CUDA error has occured! The error: " << cudaGetErrorString(err) << ", which has occured at " <<
			file << ", on line " << line << ", in function: " << func << "' \n";
		cudaDeviceReset();
		exit(err);
	}
}

class Material
{
public:
	__host__ __device__ Material(const Material& m) { this->albedo = m.albedo; this->metallic = m.metallic; this->roughness = m.roughness; this->emittance = m.emittance; }
	__host__ __device__ Material(Vector3f albedo, float metallic, float roughness, Vector3f emittance) : albedo(albedo), metallic(metallic), roughness(roughness), emittance(emittance) {}
	__host__ __device__ Material(Vector3f albedo, float metallic, float roughness) : albedo(albedo), metallic(metallic), roughness(roughness), emittance(Vector3f(0)) {}
	__host__ __device__ Material() {}

	__host__ __device__ ~Material() {}

	Vector3f albedo;
	float metallic;
	float roughness;
	Vector3f emittance;
};

class Ray
{
public:
	__host__ __device__ Ray(Vector3f origin, Vector3f direction) : origin(origin), direction(direction) {}
	__host__ __device__ Ray() {}
	__host__ __device__ ~Ray() {}

	Vector3f origin;
	Vector3f direction;
};

class Sphere
{
public:
	__host__ __device__ Sphere(Vector3f center, float radius, Material material) : center(center), radius(radius), material(material) {}
	__host__ __device__ Sphere() {}

	__host__ __device__ ~Sphere() {}

	__host__ __device__ inline Vector3f getNormal(Vector3f pHit) const
	{
		return (pHit - center) / radius;
	}

	__host__ __device__ float intersect(Ray ray) const
	{
		float t0, t1;

		Vector3f L = center - ray.origin;
		float tca = L.dot(ray.direction);
		if (tca < 0)
			return -1;

		float d2 = L.dot(L) - tca * tca;
		if (d2 > radius * radius)
			return -1;

		float thc = sqrt(radius * radius - d2);
		t0 = tca - thc;
		t1 = tca + thc;

		if (t0 > t1)
			swap(t0, t1);

		if (t0 < 0)
		{
			swap(t0, t1);
			if (t0 < 0)
				return -1;
		}

		return t0;
	}

	__host__ __device__ inline void swap(float& a, float& b) const
	{
		float ob = b;
		b = a;
		a = ob;
	}

	Material material;
	Vector3f center;
	float radius;
};

__device__ 
Sphere* intersectScene(Sphere* spheres, int size, Ray ray, float& t)
{
	float nearestDistance = 10000;
	Sphere* ret = nullptr;

	for (int i = 0; i < size; i++)
	{
		float dist = spheres[i].intersect(ray);

		if (dist > 0)
		{
			if (dist < nearestDistance)
			{
				nearestDistance = dist;
				ret = &spheres[i];
			}
		}
	}

	t = nearestDistance;
	return ret;
}

__host__ __device__
Vector3f max(Vector3f value, Vector3f min)
{
	if (value < min)
		return min;
	else
		return value;
}

__host__ __device__
Vector3f mix(Vector3f a, Vector3f b, float factor)
{
	return a + (b - a) * factor;
}

__device__
Vector3f Hemisphere(float u0, float u1)
{
	float r = (float) sqrt(1 - u0 * u1);
	float phi = (float) (2 * PI * u1);

	return Vector3f(cos(phi) * r, sin(phi) * r, sqrt(1 - u0 > 0 ? 1 - u0 : 0));
}

__device__
float DistributionGGX(Vector3f N, Vector3f H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = fmax(N.dot(H), 0.0f);
	float NdotH2 = NdotH * NdotH;

	float num = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return num / denom;
}

__device__
float GeometrySchlickGGX(float NdotV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;

	float num = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return num / denom;
}

__device__
float GeometrySmith(Vector3f N, Vector3f V, Vector3f L, float roughness)
{
	float NdotV = fmax(N.dot(V), 0.0f);
	float NdotL = fmax(N.dot(L), 0.0f);

	float ggx2 = GeometrySchlickGGX(NdotV, roughness);
	float ggx1 = GeometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}

__device__
Vector3f fresnelSchlickRoughness(float cosTheta, Vector3f F0, float roughness)
{
	return F0 + (max(Vector3f(1 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

__device__
Vector3f fresnelSchlick(float cosTheta, Vector3f F0)
{
	return F0 + (Vector3f(1) - F0) * pow(1.0 - cosTheta, 5.0);
}


__device__
void createCoordinateSystem(const Vector3f& N, Vector3f& Nt, Vector3f& Nb)
{
	if (std::fabs(N.x) > std::fabs(N.y))
		Nt = Vector3f(N.z, 0, -N.x) / sqrtf(N.x * N.x + N.z * N.z);
	else
		Nt = Vector3f(0, -N.z, N.y) / sqrtf(N.y * N.y + N.z * N.z);

	//Nb = N.cross(Nt);
}

__device__
Vector3f trace(Ray ray, Sphere* spheres, curandState* randState, int size, int depth)
{
	//Depth clamp
	if (depth >= MAX_DEPTH)
		return Vector3f(0.0);

	//Intersection
	float t;
	Sphere* sphere = intersectScene(spheres, size, ray, t);

	if (sphere == nullptr)
	{
		return Vector3f(0.0);
	}

	//Some properties for the next ray, and it's generation
	Vector3f hitP = ray.origin + ray.direction * t;
	Vector3f normal = sphere->getNormal(hitP);
	Vector3f view = ray.direction;

	//Properties of the sphere we've hit
	Material material = sphere->material;

	Vector3f albedo = material.albedo;
	
	float metallicness = material.metallic;
	float roughness = material.roughness;

	Vector3f emittance = material.emittance;

	Vector3f F0(0.04);
	F0 = mix(F0, albedo, metallicness);

	//Rendering equation: light out = light emitted + light reflected towards camera
	Vector3f sample = Hemisphere(curand_uniform(randState++), curand_uniform(randState++));

	Vector3f N = normal;
	Vector3f V = view;
	//Vector3f R = view.reflect(normal);

	Vector3f Nt, Nb;
	createCoordinateSystem(N, Nt, Nb);

	Vector3f outV = Vector3f(
		sample.x * Nb.x + sample.y * N.x + sample.z * Nt.x,
		sample.x * Nb.y + sample.y * N.y + sample.z * Nt.y,
		sample.x * Nb.z + sample.y * N.z + sample.z * Nt.z
	);

	Vector3f H = (V + outV) / (V + outV).abs();

	float NdotV = fmax(N.dot(V), 0.0f);
	float HdotV = fmax(H.dot(V), 0.0f);

	float NDF = DistributionGGX(N, H, roughness);
	Vector3f F = fresnelSchlickRoughness(HdotV, F0, roughness);
	float G = GeometrySmith(N, V, outV, roughness);

	Vector3f outRadiance = (F * NDF * G * (PI / 2)) / fmax(N.dot(outV), 0.000001f);

	Vector3f incoming = trace(Ray(hitP, outV), spheres, randState, size, depth + 1);

	return emittance + (outRadiance * (1 - metallicness) + incoming * metallicness);
}

__device__
Vector3f renderPixel(Sphere* spheres, curandState* randState, int size, int x, int y)
{
	Vector3f totalColor;

	Vector3f position = Vector3f(x, y, 0);
	Vector3f direction = Vector3f(0, 0, 1);

	Ray ray = Ray(position, direction);

	printf("X: %d \n", position.x);

	for (int i = 0; i < SAMPLES; i++)
	{
		totalColor += trace(ray, spheres, randState, size, 0);
	}

	return totalColor / SAMPLES;
}

__global__ 
void render(Vector3f* pixels, Sphere* spheres, curandState* randState, int size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= WIDTH) || (j >= HEIGHT)) return;

	int index = j * WIDTH + i;

	pixels[index] = renderPixel(spheres, randState, size, i, j);
}

void createScene(Sphere* spheres)
{
	spheres[0] = Sphere(Vector3f(0, 0, 50), 40, Material(Vector3f(1), .3f, 1));
	spheres[1] = Sphere(Vector3f(-100, -100, 30), 100, Material(Vector3f(1, 0, 0), .4f, .1f));
	spheres[2] = Sphere(Vector3f(200, 400, -1000), 500, Material(Vector3f(1, 0, 0), .4f, .1f, Vector3f(100)));
}

__global__ 
void setupRandom(curandState* randState) 
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	curand_init(1234, idx, 0, &randState[idx]);
}

int main()
{
	//Create output file
	std::ofstream file("C:/Users/akmec/Desktop/out.ppm", std::ofstream::out);

	file << "P3\n" << WIDTH << "\n" << HEIGHT << "\n" << "255" << "\n";

	Vector3f v(PI);
	std::cout << v.x << std::endl;

	//Variables
	Vector3f* pixels_d;
	Sphere* spheres_d;
	curandState* randState;
	
	cudaCheckErrors(cudaMallocManaged((void**) &pixels_d, WIDTH * HEIGHT * sizeof(Vector3f)));
	cudaCheckErrors(cudaMallocManaged((void**) &spheres_d, NUM_OBJECTS * sizeof(Sphere)));
	cudaCheckErrors(cudaMalloc((void**) &randState, sizeof(curandState)));

	createScene(spheres_d);

	setupRandom<<<1, 1>>>(randState);

	cudaCheckErrors(cudaGetLastError());
	cudaCheckErrors(cudaDeviceSynchronize());
	
	std::cout << "Started rendering" << std::endl;

	int tx = 8;
	int ty = 8;

	dim3 blocks(WIDTH / tx + 1, HEIGHT / ty + 1, 1);
	dim3 threads(tx, ty, 1);

	render<<<blocks, threads>>>(pixels_d, spheres_d, randState, NUM_OBJECTS);

	cudaCheckErrors(cudaGetLastError());
	cudaCheckErrors(cudaDeviceSynchronize());

	std::cout << "Started writing file" << std::endl;

	for (int y = 0; y < HEIGHT; y++)
	{
		for (int x = 0; x < WIDTH; x++)
		{
			size_t index = y * WIDTH + x;

			file << int(pixels_d[index].x * 255.99) << " ";
			file << int(pixels_d[index].y * 255.99) << " ";
			file << int(pixels_d[index].z * 255.99) << " ";
		}

		file << "\n";
	}

	file.close();

	std::cout << "Finished everything" << std::endl;

	cudaCheckErrors(cudaFree(pixels_d));
	cudaCheckErrors(cudaFree(spheres_d));
	cudaCheckErrors(cudaFree(randState));

	system("PAUSE");
	return 0;
}