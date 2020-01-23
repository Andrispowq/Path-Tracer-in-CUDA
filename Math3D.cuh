#ifndef MATH3D_CUH_INCLUDED
#define MATH3D_CUH_INCLUDED

#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

#define MATH_PI 3.1415926535897932384626433832795

#define ToRadians(x) (x * MATH_PI) / 180.0f
#define ToDegrees(x) (x * 180.0f) / MATH_PI

template<typename T>
inline T Clamp(const T &a, const T &min, const T &max)
{
	if (a < min)
		return min;
	else if (a > max)
		return max;
	else return a;
}

class Vector2f
{
public:
	__host__ __device__ Vector2f(const Vector2f& v) { this->x = v.x; this->y = v.y; }
	__host__ __device__ Vector2f(float x, float y) : x(x), y(y) {}
	__host__ __device__ Vector2f(float v) : x(v), y(v) {}
	__host__ __device__ Vector2f() : x(0), y(0) {}
	__host__ __device__ ~Vector2f() {}

	__host__ __device__ inline Vector2f operator+(Vector2f v) const { return Vector2f(x + v.x, y + v.y); }
	__host__ __device__ inline Vector2f operator-(Vector2f v) const { return Vector2f(x - v.x, y - v.y); }
	__host__ __device__ inline Vector2f operator*(Vector2f v) const { return Vector2f(x * v.x, y * v.y); }
	__host__ __device__ inline Vector2f operator/(Vector2f v) const { return Vector2f(x / v.x, y / v.y); }

	__host__ __device__ inline Vector2f operator+=(Vector2f v) { return Vector2f(x += v.x, y += v.y); }
	__host__ __device__ inline Vector2f operator-=(Vector2f v) { return Vector2f(x -= v.x, y -= v.y); }
	__host__ __device__ inline Vector2f operator*=(Vector2f v) { return Vector2f(x *= v.x, y *= v.y); }
	__host__ __device__ inline Vector2f operator/=(Vector2f v) { return Vector2f(x /= v.x, y /= v.y); }

	__host__ __device__ inline Vector2f operator+(float v) const { return Vector2f(x + v, y + v); }
	__host__ __device__ inline Vector2f operator-(float v) const { return Vector2f(x - v, y - v); }
	__host__ __device__ inline Vector2f operator*(float v) const { return Vector2f(x * v, y * v); }
	__host__ __device__ inline Vector2f operator/(float v) const { return Vector2f(x / v, y / v); }

	__host__ __device__ inline Vector2f operator+=(float v) { return Vector2f(x += v, y += v); }
	__host__ __device__ inline Vector2f operator-=(float v) { return Vector2f(x -= v, y -= v); }
	__host__ __device__ inline Vector2f operator*=(float v) { return Vector2f(x *= v, y *= v); }
	__host__ __device__ inline Vector2f operator/=(float v) { return Vector2f(x /= v, y /= v); }

	__host__ __device__ inline bool operator==(Vector2f v) const { return x == v.x && y == v.y; }
	__host__ __device__ inline bool operator<(Vector2f v) const { return x < v.x && y < v.y; }
	__host__ __device__ inline bool operator>(Vector2f v) const { return x > v.x && y > v.y; }
	__host__ __device__ inline bool operator>=(Vector2f v) const { return x >= v.x && y >= v.y; }
	__host__ __device__ inline bool operator<=(Vector2f v) const { return x <= v.x && y <= v.y; }

	__host__ __device__ inline float length() const { return sqrt(x * x + y * y); }
	__host__ __device__ inline float lengthSquared() const { return x * x + y * y; }
	__host__ __device__ inline float dot(Vector2f v) const { return x * v.x + y * v.y; }
	__host__ __device__ inline Vector2f normalize() const { return *this / length(); }
	__host__ __device__ inline Vector2f normalized() { return *this /= length(); }

	__host__ __device__ inline Vector2f abs() { return Vector2f(fabs(x), fabs(y)); }

	float x;
	float y;
protected:
private:
};

class Vector3f
{
public:
	__host__ __device__ Vector3f(const Vector3f& v) { this->x = v.x; this->y = v.y; this->z = v.z; }
	__host__ __device__ Vector3f(float x, float y, float z) : x(x), y(y), z(z) {}
	__host__ __device__ Vector3f(float v) : x(v), y(v), z(v) {}
	__host__ __device__ Vector3f() : x(0), y(0), z(0) {}
	__host__ __device__ ~Vector3f() {}

	__host__ __device__ inline Vector3f operator+(Vector3f v) const { return Vector3f(x + v.x, y + v.y, z + v.z); }
	__host__ __device__ inline Vector3f operator-(Vector3f v) const { return Vector3f(x - v.x, y - v.y, z - v.z); }
	__host__ __device__ inline Vector3f operator*(Vector3f v) const { return Vector3f(x * v.x, y * v.y, z * v.z); }
	__host__ __device__ inline Vector3f operator/(Vector3f v) const { return Vector3f(x / v.x, y / v.y, z / v.z); }

	__host__ __device__ inline Vector3f operator+=(Vector3f v) { return Vector3f(x += v.x, y += v.y, z += v.z); }
	__host__ __device__ inline Vector3f operator-=(Vector3f v) { return Vector3f(x -= v.x, y -= v.y, z -= v.z); }
	__host__ __device__ inline Vector3f operator*=(Vector3f v) { return Vector3f(x *= v.x, y *= v.y, z *= v.z); }
	__host__ __device__ inline Vector3f operator/=(Vector3f v) { return Vector3f(x /= v.x, y /= v.y, z /= v.z); }

	__host__ __device__ inline Vector3f operator+(float v) const { return Vector3f(x + v, y + v, z + v); }
	__host__ __device__ inline Vector3f operator-(float v) const { return Vector3f(x - v, y - v, z - v); }
	__host__ __device__ inline Vector3f operator*(float v) const { return Vector3f(x * v, y * v, z * v); }
	__host__ __device__ inline Vector3f operator/(float v) const { return Vector3f(x / v, y / v, z / v); }

	__host__ __device__ inline Vector3f operator+=(float v) { return Vector3f(x += v, y += v, z += v); }
	__host__ __device__ inline Vector3f operator-=(float v) { return Vector3f(x -= v, y -= v, z -= v); }
	__host__ __device__ inline Vector3f operator*=(float v) { return Vector3f(x *= v, y *= v, z *= v); }
	__host__ __device__ inline Vector3f operator/=(float v) { return Vector3f(x /= v, y /= v, z /= v); }

	__host__ __device__ inline bool operator==(Vector3f v) const { return x == v.x && y == v.y && z == v.z; }
	__host__ __device__ inline bool operator<(Vector3f v) const { return x < v.x && y < v.y && z < v.z; }
	__host__ __device__ inline bool operator>(Vector3f v) const { return x > v.x && y > v.y && z > v.z; }
	__host__ __device__ inline bool operator>=(Vector3f v) const { return x >= v.x && y >= v.y && z >= v.z; }
	__host__ __device__ inline bool operator<=(Vector3f v) const { return x <= v.x && y <= v.y && z <= v.z; }

	__host__ __device__ inline float length() const { return sqrt(x * x + y * y + z * z); }
	__host__ __device__ inline float lengthSquared() const { return x * x + y * y + z * z; }
	__host__ __device__ inline float dot(Vector3f v) const { return x * v.x + y * v.y + z * v.z; }
	__host__ __device__ inline Vector3f normalize() const { return *this / length(); }
	__host__ __device__ inline Vector3f normalized() { return *this /= length(); }

	__host__ __device__ inline Vector3f abs() { return Vector3f(fabs(x), fabs(y), fabs(z)); }

	__host__ __device__ inline Vector3f cross(Vector3f v) const;
	__host__ __device__ inline Vector3f reflect(Vector3f normal) const;
	__host__ __device__ inline Vector3f refract(Vector3f normal, float refractIndexRatio) const;

	float x;
	float y;
	float z;
protected:
private:
};

class Vector4f
{
public:
	__host__ __device__ Vector4f(const Vector4f& v) { this->x = v.x; this->y = v.y; this->z = v.z; this->w = v.w; }
	__host__ __device__ Vector4f(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
	__host__ __device__ Vector4f(float v) : x(v), y(v), z(v), w(v) {}
	__host__ __device__ Vector4f() : x(0), y(0), z(0), w(0) {}
	__host__ __device__ ~Vector4f() {}

	__host__ __device__ inline Vector4f operator+(Vector4f v) const { return Vector4f(x + v.x, y + v.y, z + v.z, w + v.w); }
	__host__ __device__ inline Vector4f operator-(Vector4f v) const { return Vector4f(x - v.x, y - v.y, z - v.z, w - v.w); }
	__host__ __device__ inline Vector4f operator*(Vector4f v) const { return Vector4f(x * v.x, y * v.y, z * v.z, w * v.w); }
	__host__ __device__ inline Vector4f operator/(Vector4f v) const { return Vector4f(x / v.x, y / v.y, z / v.z, w / v.w); }

	__host__ __device__ inline Vector4f operator+=(Vector4f v) { return Vector4f(x += v.x, y += v.y, z += v.z, w += v.w); }
	__host__ __device__ inline Vector4f operator-=(Vector4f v) { return Vector4f(x -= v.x, y -= v.y, z -= v.z, w -= v.w); }
	__host__ __device__ inline Vector4f operator*=(Vector4f v) { return Vector4f(x *= v.x, y *= v.y, z *= v.z, w *= v.w); }
	__host__ __device__ inline Vector4f operator/=(Vector4f v) { return Vector4f(x /= v.x, y /= v.y, z /= v.z, w /= v.w); }

	__host__ __device__ inline Vector4f operator+(float v) const { return Vector4f(x + v, y + v, z + v, w + v); }
	__host__ __device__ inline Vector4f operator-(float v) const { return Vector4f(x - v, y - v, z - v, w - v); }
	__host__ __device__ inline Vector4f operator*(float v) const { return Vector4f(x * v, y * v, z * v, w * v); }
	__host__ __device__ inline Vector4f operator/(float v) const { return Vector4f(x / v, y / v, z / v, w / v); }

	__host__ __device__ inline Vector4f operator+=(float v) { return Vector4f(x += v, y += v, z += v, w += v); }
	__host__ __device__ inline Vector4f operator-=(float v) { return Vector4f(x -= v, y -= v, z -= v, w -= v); }
	__host__ __device__ inline Vector4f operator*=(float v) { return Vector4f(x *= v, y *= v, z *= v, w *= v); }
	__host__ __device__ inline Vector4f operator/=(float v) { return Vector4f(x /= v, y /= v, z /= v, w /= v); }

	__host__ __device__ inline bool operator==(Vector4f v) const { return x == v.x && y == v.y && z == v.z && w == v.w; }
	__host__ __device__ inline bool operator<(Vector4f v) const { return x < v.x && y < v.y && z < v.z && w < v.w; }
	__host__ __device__ inline bool operator>(Vector4f v) const { return x > v.x && y > v.y && z > v.z && w > v.w; }
	__host__ __device__ inline bool operator>=(Vector4f v) const { return x >= v.x && y >= v.y && z >= v.z && w >= v.w; }
	__host__ __device__ inline bool operator<=(Vector4f v) const { return x <= v.x && y <= v.y && z <= v.z && w <= v.w; }

	__host__ __device__ inline float length() const { return sqrt(x * x + y * y + z * z + w * w); }
	__host__ __device__ inline float lengthSquared() const { return x * x + y * y + z * z + w * w; }
	__host__ __device__ inline float dot(Vector4f v) const { return x * v.x + y * v.y + z * v.z + w * v.w; }
	__host__ __device__ inline Vector4f normalize() const { return *this / length(); }
	__host__ __device__ inline Vector4f normalized() { return *this /= length(); }

	__host__ __device__ inline Vector4f abs() { return Vector4f(fabs(x), fabs(y), fabs(z), fabs(w)); }

	float x;
	float y;
	float z;
	float w;
protected:
private:
};

class Matrix4f
{
	__host__ __device__ Matrix4f(Vector4f v);
	__host__ __device__ Matrix4f();

	__host__ __device__ inline Matrix4f operator+(Matrix4f v) const;
	__host__ __device__ inline Matrix4f operator-(Matrix4f v) const;

	__host__ __device__ inline Matrix4f operator+=(Matrix4f v);
	__host__ __device__ inline Matrix4f operator-=(Matrix4f v);

	__host__ __device__ inline Matrix4f operator+(Vector4f v) const;
	__host__ __device__ inline Matrix4f operator-(Vector4f v) const;

	__host__ __device__ inline Matrix4f operator+=(Vector4f v);
	__host__ __device__ inline Matrix4f operator-=(Vector4f v);

	__host__ __device__ inline Matrix4f operator+(float v) const;
	__host__ __device__ inline Matrix4f operator-(float v) const;

	__host__ __device__ inline Matrix4f operator+=(float v);
	__host__ __device__ inline Matrix4f operator-=(float v);

	__host__ __device__ inline Matrix4f operator*(Matrix4f v) const;
	__host__ __device__ inline Matrix4f operator*=(Matrix4f v);

	__host__ __device__ inline Vector4f operator*(Vector4f v) const;

	__host__ __device__ inline Matrix4f Identity() const;
	__host__ __device__ inline Matrix4f Zero() const;

	__host__ __device__ Matrix4f Translation(Vector3f position) const;
	__host__ __device__ Matrix4f Rotation(Vector3f rotation) const;
	__host__ __device__ Matrix4f Scaling(Vector3f scale) const;

	__host__ __device__ Matrix4f Transformation(Vector3f translation, Vector3f rotation, Vector3f scaling) const;
	__host__ __device__ Matrix4f PerspectiveProjection(float fov, float aspectRatio, float nearPlane, float farPlane) const;

	float m[4][4];
protected:
private:
};

#endif MATH3D_CUH_INCLUDED