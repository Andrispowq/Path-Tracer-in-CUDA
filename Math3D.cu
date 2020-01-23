#include "Math3d.cuh"

__host__ __device__ inline Vector3f Vector3f::cross(Vector3f v) const
{
	float x_ = y * v.z - z * v.y;
	float y_ = z * v.x - x * v.z;
	float z_ = x * v.y - y * v.x;

	return Vector3f(x_, y_, z_);
}

__host__ __device__ inline Vector3f Vector3f::reflect(Vector3f normal) const
{
	return *this - normal * 2 * this->dot(normal);
}

__host__ __device__ inline Vector3f Vector3f::refract(Vector3f normal, float refractIndexRatio) const
{
	float k = 1 - refractIndexRatio * refractIndexRatio * (1 - this->dot(normal) * this->dot(normal));

	if (k < 0)
		return Vector3f(0);
	else
		return *this * refractIndexRatio - normal * (refractIndexRatio * this->dot(normal) + sqrt(k));
}

__host__ __device__ Matrix4f::Matrix4f(Vector4f v)
{
	for (int i = 0; i < 4; i++)
	{
		m[i][0] = v.x;
		m[i][1] = v.y;
		m[i][2] = v.z;
		m[i][3] = v.w;
	}
}

__host__ __device__ Matrix4f::Matrix4f()
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			m[i][j] = 0;
		}
	}
}

__host__ __device__ inline Matrix4f Matrix4f::operator+(Matrix4f v) const
{
	Matrix4f res = Matrix4f();

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			res.m[i][j] = m[i][j] + v.m[i][j];
		}
	}

	return res;
}

__host__ __device__ inline Matrix4f Matrix4f::operator-(Matrix4f v) const
{
	Matrix4f res = Matrix4f();

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			res.m[i][j] = m[i][j] - v.m[i][j];
		}
	}

	return res;
}

__host__ __device__ inline Matrix4f Matrix4f::operator+=(Matrix4f v)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			m[i][j] += v.m[i][j];
		}
	}

	return *this;
}

__host__ __device__ inline Matrix4f Matrix4f::operator-=(Matrix4f v)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			m[i][j] -= v.m[i][j];
		}
	}

	return *this;
}

__host__ __device__ inline Matrix4f Matrix4f::operator+(Vector4f v) const
{
	Matrix4f res = Matrix4f();

	for (int i = 0; i < 4; i++)
	{
		res.m[i][0] = m[i][0] + v.x;
		res.m[i][1] = m[i][1] + v.y;
		res.m[i][2] = m[i][2] + v.z;
		res.m[i][3] = m[i][3] + v.w;
	}

	return res;
}

__host__ __device__ inline Matrix4f Matrix4f::operator-(Vector4f v) const
{
	Matrix4f res = Matrix4f();

	for (int i = 0; i < 4; i++)
	{
		res.m[i][0] = m[i][0] - v.x;
		res.m[i][1] = m[i][1] - v.y;
		res.m[i][2] = m[i][2] - v.z;
		res.m[i][3] = m[i][3] - v.w;
	}

	return res;
}

__host__ __device__ inline Matrix4f Matrix4f::operator+=(Vector4f v)
{
	for (int i = 0; i < 4; i++)
	{
		m[i][0] += v.x;
		m[i][1] += v.y;
		m[i][2] += v.z;
		m[i][3] += v.w;
	}

	return *this;
}

__host__ __device__ inline Matrix4f Matrix4f::operator-=(Vector4f v)
{
	for (int i = 0; i < 4; i++)
	{
		m[i][0] -= v.x;
		m[i][1] -= v.y;
		m[i][2] -= v.z;
		m[i][3] -= v.w;
	}

	return *this;
}

__host__ __device__ inline Matrix4f Matrix4f::operator+(float v) const
{
	Matrix4f res = Matrix4f();

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			res.m[i][j] = m[i][j] + v;
		}
	}

	return res;
}

__host__ __device__ inline Matrix4f Matrix4f::operator-(float v) const
{
	Matrix4f res = Matrix4f();

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			res.m[i][j] = m[i][j] - v;
		}
	}

	return res;
}

__host__ __device__ inline Matrix4f Matrix4f::operator+=(float v)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			m[i][j] += v;
		}
	}

	return *this;
}

__host__ __device__ inline Matrix4f Matrix4f::operator-=(float v)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			m[i][j] -= v;
		}
	}

	return *this;
}

__host__ __device__ inline Matrix4f Matrix4f::operator*(Matrix4f v) const
{
	Matrix4f res = Matrix4f();

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			res.m[i][j] = m[i][0] * v.m[0][j] + m[i][1] * v.m[1][j] + m[i][2] * v.m[2][j] + m[i][3] * v.m[3][j];
		}
	}

	return res;
}

__host__ __device__ inline Matrix4f Matrix4f::operator*=(Matrix4f v)
{
	Matrix4f res = Matrix4f();

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			res.m[i][j] = m[i][0] * v.m[0][j] + m[i][1] * v.m[1][j] + m[i][2] * v.m[2][j] + m[i][3] * v.m[3][j];
		}
	}

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			this->m[i][j] = res.m[i][j];
		}
	}

	return *this;
}

__host__ __device__ inline Vector4f Matrix4f::operator*(Vector4f v) const
{
	Vector4f res = Vector4f();

	res.x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w;
	res.y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w;
	res.z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w;
	res.w = m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w;

	return res;
}

__host__ __device__ inline Matrix4f Matrix4f::Identity() const
{
	Matrix4f res = Matrix4f();

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (i == j)
				res.m[i][j] = 1;
			else
				res.m[i][j] = 0;
		}
	}

	return res;
}

__host__ __device__ inline Matrix4f Matrix4f::Zero() const
{
	Matrix4f res = Matrix4f();

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			res.m[i][j] = 0;
		}
	}

	return res;
}

__host__ __device__ Matrix4f Matrix4f::Translation(Vector3f position) const
{
	Matrix4f res = Matrix4f().Identity();

	res.m[0][3] = position.x;
	res.m[1][3] = position.y;
	res.m[2][3] = position.z;

	return res;
}

__host__ __device__ Matrix4f Matrix4f::Rotation(Vector3f rotation) const
{
	Matrix4f rx = Matrix4f().Identity();
	Matrix4f ry = Matrix4f().Identity();
	Matrix4f rz = Matrix4f().Identity();

	float x = ToRadians(rotation.x);
	float y = ToRadians(rotation.y);
	float z = ToRadians(rotation.z);

	rx.m[1][1] = cos(x);
	rx.m[1][2] = -sin(x);
	rx.m[2][1] = sin(x);
	rx.m[2][2] = cos(x);

	ry.m[0][0] = cos(y);
	ry.m[0][2] = sin(y);
	ry.m[2][0] = -sin(y);
	ry.m[2][2] = cos(y);

	rz.m[0][0] = cos(z);
	rz.m[0][1] = -sin(z);
	rz.m[1][0] = sin(z);
	rz.m[1][1] = cos(z);

	return ry * rx * rz;
}

__host__ __device__ Matrix4f Matrix4f::Scaling(Vector3f scale) const
{
	Matrix4f res = Matrix4f().Identity();

	res.m[0][0] = scale.x;
	res.m[1][1] = scale.y;
	res.m[2][2] = scale.z;

	return res;
}

__host__ __device__ Matrix4f Matrix4f::Transformation(Vector3f translation, Vector3f rotation, Vector3f scaling) const
{
	Matrix4f Translation = Matrix4f().Translation(translation);
	Matrix4f Rotation = Matrix4f().Rotation(rotation);
	Matrix4f Scaling = Matrix4f().Scaling(scaling);

	return Scaling * Rotation * Translation;
}

__host__ __device__ Matrix4f Matrix4f::PerspectiveProjection(float fov, float aspectRatio, float nearPlane, float farPlane) const
{
	Matrix4f res = Matrix4f().Identity();

	float tanFOV = tan(ToRadians(fov / 2));
	float frustumLength = farPlane - nearPlane;

	res.m[0][0] = 1 / (tanFOV * aspectRatio);
	res.m[1][1] = 1 / tanFOV;
	res.m[2][2] = (farPlane + nearPlane) / frustumLength;
	res.m[2][3] = -(2 * farPlane * nearPlane) / frustumLength;
	res.m[3][2] = 1;
	res.m[3][3] = 0;

	return res;
}