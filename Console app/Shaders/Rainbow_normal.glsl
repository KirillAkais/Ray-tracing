#version 460

#if 0

#define Iterations 8
#define RaysPerPixel 128

#else

#define Iterations 5
#define RaysPerPixel 1550

#endif
#define MaxDistance 10000.0
#define N_IN 1.33
#define N_OUT 1.0

#define SphereCount 1

#define SunBrightness 5000

#define PI2 6.2831853071795864769
#define PI  3.1415926535897932385

out vec4 fragColor;

uniform vec2 iResolution;
uniform vec3 iPos;
uniform vec2 iMouse;

uniform sampler2D iSpectrum;
uniform sampler2D iGradient;
uniform sampler2D iPrevFrame;

uniform float iAngle;
uniform float iSamplePart;
uniform float iMatrixSize;
uniform float iFov;
uniform float iFocus;

uniform vec2 iSeed1;
uniform vec2 iSeed2;

uvec4 R_STATE;

//Random direction generation

uint TausStep(uint z, int S1, int S2, int S3, uint M)
{
	uint b = (((z << S1) ^ z) >> S2);
	return (((z & M) << S3) ^ b);	
}

uint LCGStep(uint z, uint A, uint C)
{
	return (A * z + C);	
}

vec2 hash22(vec2 p)
{
	p += iSeed1.x;
	vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
	p3 += dot(p3, p3.yzx+33.33);
	return fract((p3.xx+p3.yz)*p3.zy);
}

float random()
{
	R_STATE.x = TausStep(R_STATE.x, 13, 19, 12, uint(4294967294));
	R_STATE.y = TausStep(R_STATE.y, 2, 25, 4, uint(4294967288));
	R_STATE.z = TausStep(R_STATE.z, 3, 11, 17, uint(4294967280));
	R_STATE.w = LCGStep(R_STATE.w, uint(1664525), uint(1013904223));
	return 2.3283064365387e-10 * float((R_STATE.x ^ R_STATE.y ^ R_STATE.z ^ R_STATE.w));
}

vec3 random_sphere() {
	vec3 rand = vec3(random(), random(), random());
	float theta = rand.x * 2.0 * 3.14159265;
	float v = rand.y;
	float phi = acos(2.0 * v - 1.0);
	float r = pow(rand.z, 1.0 / 3.0);
	float x = r * sin(phi) * cos(theta);
	float y = r * sin(phi) * sin(theta);
	float z = r * cos(phi);
	return vec3(x, y, z);
}

vec3 random_normal_hemisphere(vec3 n)
{
    vec3 v = random_sphere();
    return dot(v, n) < 0.0 ? -v : v;
}

float gauss_dist(float m, float s)
{
	float u1 = random();
	float u2 = random();
	float mag = s * sqrt(-2.0 * log(u1));
	float z0  = mag * cos(PI2 * u2) + m;
	return z0;
}


mat2 rot(float a)
 {
	float s = sin(a);
	float c = cos(a);
	return mat2(c, -s, s, c);
}

mat3 rot(float x, float y, float z)
{
	float sx = sin(x);
	float sy = sin(y);
	float sz = sin(z);
	float cx = cos(x);
	float cy = cos(y);
	float cz = cos(z);
	return mat3(cy*cz, sx*sy*cz-cx*sz, cx*sy*cz+sx*sz,
				cy*sz, sx*sy*sz+cx*cz, cx*sy*sz-sx*cz,
				-sy, sx*cy, cx*cy);
}

mat3 rot(vec3 v)
{
	float sx = sin(v.x);
	float sy = sin(v.y);
	float sz = sin(v.z);
	float cx = cos(v.x);
	float cy = cos(v.y);
	float cz = cos(v.z);
	return mat3(cy*cz, sx*sy*cz-cx*sz, cx*sy*cz+sx*sz,
				cy*sz, sx*sy*sz+cx*cz, cx*sy*sz-sx*cz,
				-sy, sx*cy, cx*cy);
}

struct Ray
{
	vec3 origin;
	vec3 direction;
	vec3 color;
	vec3 light;
	bool is_inside;
	int wave_length;
	vec3 pos;
};

struct Material
{
    vec3 emmition_color;
    vec3 diffusion_color;
	vec3 specular_color;
	vec3 transmissive_color;
	float emmitance;
    float roughness;
	float matt;
    float transparency;
};

struct Sphere
{
    Material material;
    vec3 position;
    float radius;
};

Sphere spheres[SphereCount];

float n[31] = float[31](
1.313831671,
1.313033277,
1.312296981,
1.311616362,
1.310985827,
1.310400482,
1.309856026,
1.309348669,
1.308875055,
1.308432208,
1.308017476,
1.307628492,
1.307263137,
1.306919507,
1.30659589,
1.306290741,
1.306002663,
1.305730392,
1.305472777,
1.305228773,
1.304997428,
1.304777871,
1.304569307,
1.304371006,
1.304182301,
1.304002575,
1.303831262,
1.303667842,
1.303511833,
1.303362788,
1.303220298
);

vec3 wl_col[31] = vec3[31](
vec3(131, 0, 181),//400
vec3(126, 0, 219),
vec3(106, 0, 255),
vec3(61, 0, 255),
vec3(0, 0, 255),//440
vec3(0, 70, 255),
vec3(0, 123, 255),
vec3(0, 169, 255),
vec3(0, 213, 255),
vec3(0, 255, 255),//490
vec3(0, 255, 146),
vec3(0, 255, 0),//510
vec3(54, 255, 0),
vec3(94, 255, 0),
vec3(129, 255, 0),
vec3(163, 255, 0),
vec3(195, 255, 0),
vec3(225, 255, 0),
vec3(255, 255, 0),//580
vec3(255, 223, 0),
vec3(255, 190, 0),
vec3(255, 155, 0),
vec3(255, 119, 0),
vec3(255, 79, 0),
vec3(255, 33, 0),
vec3(255, 0, 0),
vec3(255, 0, 0),
vec3(255, 0, 0),
vec3(255, 0, 0),
vec3(255, 0, 0),
vec3(255, 0, 0)
);

vec3 ang[3] = vec3[3](
vec3(1,1,0),
vec3(1,0,1),
vec3(0,1,1)
);

bool IntersectRaySphere(Ray r, Sphere sphere, out float fraction, out vec3 normal)
{
	sphere.position = r.pos;
    vec3 L = r.origin - sphere.position;
    float a = dot(r.direction, r.direction);
    float b = 2.0 * dot(L, r.direction);
    float c = dot(L, L) - sphere.radius * sphere.radius;
    float D = b * b - 4 * a * c;

    if (D < 0.0) return false;

    float r1 = (-b - sqrt(D)) / (2.0 * a);
    float r2 = (-b + sqrt(D)) / (2.0 * a);

    if (r1 > 0.0)
        fraction = r1;
    else if (r2 > 0.0)
        fraction = r2;
    else
        return false;
    normal = normalize(r.direction * fraction + L);

    return true;
}

vec3 get_sun(vec3 rd)
{
	vec2 uv = vec2(asin(rd.z) * 2, 1.5) / 3.14159265;
	uv = uv * 0.5 + 0.5;
	vec3 light = normalize(vec3(-1, 0, 0.3));
	vec3 col = texture2D(iGradient, uv).rgb;
	vec3 sun = vec3(1., 1., 1.0);
	if(dot(rd, light)<0.99995)
	{
		sun *= max(0, pow(dot(rd, light), 8192))/100000;
	}
	return sun * SunBrightness;
}

vec3 get_sky(vec3 rd)
{
	vec2 uv = vec2(asin(rd.z) * 2, 1.5) / 3.14159265;
	uv = uv * 0.5 + 0.5;
	vec3 col = texture2D(iGradient, uv).rgb;
	return col;
}


bool CastRay(Ray r, out float fraction, out vec3 normal, out Material material)
{
    float minDistance = MaxDistance;

    float D;
    vec3 N;
    if (IntersectRaySphere(r, spheres[0], D, N) && D < minDistance)
    {
        minDistance = D;
        normal = N;
        material = spheres[0].material;
    }
    fraction = minDistance;
    return minDistance != MaxDistance;
}


float fresnel(float n_in, float n_out, vec3 direction, vec3 normal)
{
    float R0 = ((n_out - n_in) * (n_out - n_in)) / ((n_out + n_in) * (n_out + n_in));
    float fresnel = R0 + (1.0 - R0) * pow((1.0 - abs(dot(direction, normal))), 5.0);
    return fresnel;
}

vec3 ideal_refract(vec3 direction, vec3 normal, float n_in, float n_out)
{
    bool from_outside = dot(normal, direction) < 0.0;
    float ratio = from_outside ? n_out / n_in : n_in / n_out;

    vec3 refraction, reflection;
    refraction = from_outside ? refract(direction, normal, ratio) : refract(direction, -normal, ratio);
	reflection = from_outside ? reflect(direction, normal) : reflect(direction, -normal);
    return refraction == vec3(0.0) ? reflection : refraction;
}

bool is_refracted(vec3 direction, vec3 normal, float transparency, float n_in, float n_out)
{
    float fresnel = fresnel(n_in, n_out, direction, normal);
	float rand = random();
    return transparency > rand && fresnel < rand;
}

vec3 trace_ray(vec3 ro, vec3 rd, int wl)
{
	Ray ray;
	ray.origin = ro;
	ray.direction = rd;
	
	ray.light = vec3(0.0);
	ray.is_inside = false;
	ray.wave_length = wl;
	ray.pos = ro + rd*1000.0 + random_sphere()*1.01;
	ray.color = wl_col[ray.wave_length]*vec3(1.33,1.54,1.0)/255.0;
	
	Material material;
	vec3 normal;
	float distance;

	for (int i = 0; i < Iterations; i++)
	{
		if(!CastRay(ray, distance, normal, material))
		{
			ray.light += ray.color;
			return ray.light * get_sun(ray.direction) + get_sky(ray.direction);
		}
		vec3 new_origin = ray.origin + ray.direction * distance;
		
		vec3 random_vec = normalize(normal + random_sphere());
		
		vec3 new_direction = random_vec;
		
		bool refracted = is_refracted(ray.direction, normal, material.transparency, n[ray.wave_length], N_OUT);
		if (refracted)
        {
			ray.light += material.emmitance * material.emmition_color * ray.color;
			ray.color *= material.transmissive_color;
            vec3 ideal_refraction = ideal_refract(ray.direction, normal, n[ray.wave_length], N_OUT);
			if(random() > material.roughness)
			{
				new_direction = normalize(mix(ideal_refraction, -new_direction, material.matt));
			}
			else
			{
				new_direction *= -1.0;
			}
            new_origin += normal * (dot(new_direction, normal) < 0.0 ? -0.001 : 0.001);
			ray.is_inside = dot(new_direction, normal) < 0;
        }
        else
        {
			if(ray.is_inside)	normal *= -1;
			vec3 ideal_reflection = reflect(ray.direction, normal);
			new_origin += normal * 0.001;
			if(random() > material.roughness)
			{
				ray.light += material.emmitance * material.emmition_color * ray.color;
				ray.color *= material.specular_color;
				new_direction = normalize(mix(ideal_reflection, new_direction, material.matt));
				
			}
			else
			{
				ray.light += material.emmitance * material.emmition_color * ray.color;
				ray.color *= material.diffusion_color;
				new_direction = normalize(mix(normalize(-new_origin), new_direction, 0.1));
			}
        }
		
		
		ray.origin = new_origin;
		ray.direction = new_direction;
	}
	return ray.light;
}

void scene_init()
{
	Material drop;
	drop.diffusion_color = vec3(1.0, 1.0, 1.0);
	drop.specular_color = vec3(1.0, 1.0, 1.0);
	drop.transmissive_color = vec3(1, 1, 1.0);
	drop.emmitance = 0.0;
	drop.roughness = 0.0;
	drop.matt = 0.0;
	drop.transparency = 1.0;
	
	spheres[0].material = drop;
	spheres[0].position = vec3(0);
	spheres[0].radius = 1;
}

void main (void)
{
	vec2 coord = gl_FragCoord.xy / iResolution - vec2(0.5);
	coord.x /= iResolution.y / iResolution.x;
	
	vec2 uvRes = hash22(coord + 1.0) * iResolution + iResolution;
	R_STATE.x = uint(iSeed1.x + uvRes.x);
	R_STATE.y = uint(iSeed1.y + uvRes.x);
	R_STATE.z = uint(iSeed2.x + uvRes.y);
	R_STATE.w = uint(iSeed2.y + uvRes.y);
	
	float t = random() * 6.283185307;
	float u = random() + random();
	float r = 0;
	if(u > 1)
	{
		r = 2 - u;
	}
	else
	{
		r = u;
	}
	r *= iMatrixSize;
	
	vec3 d = vec3(0.0, cos(t) * r, sin(t) * r);
	d.zx *= rot(iMouse.y);
	d.xy *= rot(iMouse.x);
	
	vec3 ray_origin = iPos + d;
	vec3 ray_direction = normalize(vec3(iFocus, coord *  tan(iFov)) - d);
	
	ray_direction.zx *= rot(iMouse.y);
	ray_direction.xy *= rot(iMouse.x);
	
	ray_direction.y = sin(coord.x*PI);
	ray_direction.x = cos(coord.x*PI);
	ray_direction.z = tan(coord.y*PI/2);
	
	ray_direction = normalize(ray_direction);
	
	vec3 col = vec3(0);
	
	scene_init();
	
	for(int i = 0; i < RaysPerPixel; i++)
	{
		col += trace_ray(ray_origin, ray_direction, i%31);
	}
	col /= RaysPerPixel;
	
	vec3 texturecol = texture2D(iPrevFrame, gl_FragCoord.xy / iResolution).rgb;
	
	col = mix(texturecol, col, iSamplePart);
	
	fragColor = vec4(col, 1);
}