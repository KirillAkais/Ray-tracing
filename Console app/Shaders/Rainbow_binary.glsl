#version 460

#if 0

#define Iterations 8
#define RaysPerPixel 128

#else

#define Iterations 5
#define RaysPerPixel 93

#endif
#define MaxDistance 100.0
#define N_IN 1.33
#define N_OUT 1.0

#define SphereCount 1
#define BoxCount 0
#define PlaneCount 1

#define MARCH_STEPS 20
#define HIT_DIST 0.0001

#define SunBrightness 9300

out vec4 fragColor;

uniform vec2 iResolution;
uniform vec3 iPos;
uniform vec2 iMouse;

uniform sampler2D iSpectrum;
uniform sampler2D iGradient;
uniform sampler2D iPrevFrame;

uniform float iSamplePart;
uniform float iMatrixSize;
uniform float iFov;
uniform float iFocus;

uniform vec2 iSeed1;
uniform vec2 iSeed2;

struct Ray
{
	vec3 origin;
	vec3 direction;
	vec3 color;
	vec3 light;
	bool is_inside;
	int wave_length;
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

struct Box
{
    Material material;
    vec3 halfSize;
    mat3 rotation;
    vec3 position;
};

struct Sphere
{
    Material material;
    vec3 position;
    float radius;
};

struct Plane
{
    Material material;
    vec3 position;
    vec3 normal;
};

Sphere spheres[SphereCount];
Plane planes[PlaneCount];
//Box boxes[BoxCount];

float n[31] = float[31](
1.343556681,
1.342644593,
1.341800458,
1.341017091,
1.340288242,
1.339608449,
1.33897292,
1.338377434,
1.337818259,
1.337292088,
1.336795977,
1.336327298,
1.3358837,
1.335463073,
1.335063521,
1.334683329,
1.334320952,
1.333974986,
1.333644159,
1.333327313,
1.333023391,
1.33273143,
1.332450546,
1.332179931,
1.331918842,
1.331666596,
1.331422563,
1.331186163,
1.330956859,
1.330734155,
1.330517591
);

bool IntersectRaySphere(Ray r, Sphere sphere, out float fraction, out vec3 normal)
{
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

float sdSphere(vec3 pos, Sphere s)
{
	vec3 p = s.position - pos;
	float d = length(p) - s.radius;
	//if (d < 0 && r.is_inside)	d = -d;
	return d;
}


bool IntersectRayBox(Ray r, Box box, out float fraction, out vec3 normal)
{
	vec3 rd = r.direction * box.rotation;
	vec3 ro = (r.origin - box.position) * box.rotation;
	
	
    vec3 m = 1.0 / rd;
    vec3 n = m * ro;
    vec3 k = abs(m) * box.halfSize;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    float tN = max( max( t1.x, t1.y ), t1.z );
    float tF = min( min( t2.x, t2.y ), t2.z );
	
	mat3 txi = transpose(box.rotation);
	
    if(tN > tF || tF < 0.0) 
		return false;
	
	if(tN > 0.0)
		normal = -sign(rd) * step(t1.yzx, t1.xyz) * step(t1.zxy, t1.xyz) * txi;
	else
	{
		t1 = n - k;
     	t2 = n + k;
		normal = sign(rd) * step(t1.yzx, t1.xyz) * step(t1.zxy, t1.xyz) * txi;
	}
	
	if(tN > 0.0)
		fraction = tN;
	else
		fraction = tF;
	
    return true;
}

bool IntersectRayPlane(Ray r, Plane p, out float fraction, out vec3 normal)
{
	normal = p.normal;
    float d = -(dot(r.origin - p.position, p.normal))/dot(r.direction, p.normal);
	fraction = d;
	//if(d < 0 || length(r.origin+d*r.direction-p.position) < 1)	return false;
	if(d < 0 || dot(r.direction, vec3(1,0,0))>0.99995 || dot(r.direction, normalize(vec3(1, 0.067, -0.1)))>0.99995)	return false;
	return true;
}


vec3 get_sky(vec3 rd)
{
	vec2 uv = vec2(asin(rd.z) * 2, 1.5) / 3.14159265;
	uv = uv * 0.5 + 0.5;
	vec3 light = normalize(vec3(1, 0, 0));
	vec3 light2 = normalize(vec3(1, 0.067, -0.1));
	vec3 col = texture2D(iGradient, uv).rgb * 1.;
	vec3 sun = vec3(1., 1., 1.0);
	vec3 sun2 = vec3(1.0, 1.0, 0.0);
	//sun *= max(0, pow(dot(rd, light), 1024));
	if(dot(rd, light)<0.99995)
	{
		sun *= 0;
	}
	if(dot(rd, light2)<0.99995)
	{
		sun2 *= 0;
	}
	return (sun+sun2) * SunBrightness + col*0;
}

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

bool march_ray(Ray r, Sphere s, out float fraction, out vec3 normal)
{
	float dist = 0;
	normal = vec3(0, 1, 0);
	vec3 pos = r.origin;
	vec3 rd = r.direction;
	float d = 0;
	for(int i = 0; i < 15; i++)
	{
		d = sdSphere(pos, s);
		dist += d;
		if(d < 0.0002)
		{
			//color = vec3(0.6);
            //vec2 e = vec2(0.0001, 0);
            normal = normalize(pos - s.position);
			//col = color;
			fraction = dist;
			return true;
		}
		else if(d > 1000)
		{
			return false;
		}
		pos += rd * d;
	}
	return false;
}

bool CastRay(Ray r, out float fraction, out vec3 normal, out Material material)
{
    float minDistance = MaxDistance;

    for (int i = 0; i < SphereCount; i++)
    {
        float D;
        vec3 N;
        if (IntersectRaySphere(r, spheres[i], D, N) && D < minDistance)
        {
            minDistance = D;
            normal = N;
            material = spheres[i].material;
        }
    }

    /*for (int i = 0; i < BoxCount; i++)
    {
        float D;
        vec3 N;
        if (IntersectRayBox(r, boxes[i], D, N) && D < minDistance)
        {
            minDistance = D;
            normal = N;
            material = boxes[i].material;
        }
    }*/

	for (int i = 0; i < PlaneCount; i++)
    {
        float D;
        vec3 N;
        if (IntersectRayPlane(r, planes[i], D, N) && D < minDistance)
        {
            minDistance = D;
            normal = N;
            material = planes[i].material;
        }
    }
    fraction = minDistance;
    return minDistance != MaxDistance;
}

bool MarchRay(Ray r, out float fraction, out vec3 normal, out Material material)
{
    float minDistance = MaxDistance;
	vec3 origin = r.origin;
	vec3 N;
	float D;
	int min_i = 0;
	for(int k = 0; k < MARCH_STEPS; k++)
	{
		for (int i = 0; i < SphereCount; i++)
		{
			//D = sdSphere(r, spheres[i]);
			if (D < minDistance)
			{
				minDistance = D;
				min_i = i;
			}
		}
		if(minDistance >= MaxDistance)
		{
			return false;
		}
		if(minDistance < HIT_DIST)
		{
			fraction = length(r.origin - origin);
			normal = normalize(r.origin - spheres[min_i].position);
			material = spheres[min_i].material;
			return true;
		}
		r.origin += r.direction * minDistance;
	}
    return false;
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
    //reflection = reflect(direction, normal);
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
	//ray.color = vec3(1.0);
	ray.light = vec3(0.0);
	ray.is_inside = false;
	ray.wave_length = wl;
	ray.color = texture2D(iSpectrum, vec2(wl / 30.0, 0.5)).rgb;

	Material material;
	vec3 normal;
	float distance;

	for (int i = 0; i < Iterations; i++)
	{
		if(!CastRay(ray, distance, normal, material))
		{
			ray.light += ray.color;
			return ray.light * get_sky(ray.direction);
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
	//return normal;
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
	
	Material white;
	white.diffusion_color = vec3(1, 1, 1);
	white.specular_color = vec3(1);
	white.emmitance = 0.0;
	white.roughness = 1.0;
	white.matt = 1.0;
	
	Material black;
	black.diffusion_color = vec3(0);
	black.specular_color = vec3(0);
	black.emmitance = 0.0;
	black.roughness = 0.0;
	black.matt = 0.0;

	spheres[0].material = drop;
	spheres[0].position = vec3(0.0, 0.0, 0.0);
	spheres[0].radius = 1.0;
	/*
	boxes[0].material = black;
	boxes[0].position = vec3(0.0, 10.0, 0.0);
	boxes[0].halfSize = vec3(10, 1, 10);
	boxes[0].rotation = rot(0, 0, 0);
	
	boxes[1].material = black;
	boxes[1].position = vec3(-11.0, 0.0, 0.0);
	boxes[1].halfSize = vec3(1, 10, 10);
	boxes[1].rotation = rot(0, 0, 0);
	
	boxes[2].material = black;
	boxes[2].position = vec3(0.0, 0.0, 11.0);
	boxes[2].halfSize = vec3(10, 10, 1);
	boxes[2].rotation = rot(0, 0, 0);
	
	boxes[3].material = black;
	boxes[3].position = vec3(0.0, 0.0, -11.0);
	boxes[3].halfSize = vec3(10, 10, 1);
	boxes[3].rotation = rot(0, 0, 0);
	
	boxes[4].material = black;
	boxes[4].position = vec3(0.0, -10.0, 0.0);
	boxes[4].halfSize = vec3(10, 1, 10);
	boxes[4].rotation = rot(0, 0, 0);
	
	boxes[0].material = white;
	boxes[0].position = vec3(11.0, -1001.0, 0.0);
	boxes[0].halfSize = vec3(1, 1000, 1000);
	boxes[0].rotation = rot(0, 0, 0);
	
	boxes[1].material = white;
	boxes[1].position = vec3(11.0, 1001.0, 0.0);
	boxes[1].halfSize = vec3(1, 1000, 1000);
	boxes[1].rotation = rot(0, 0, 0);
	
	boxes[2].material = white;
	boxes[2].position = vec3(11.0, 0.0, -1001.0);
	boxes[2].halfSize = vec3(1, 1000, 1000);
	boxes[2].rotation = rot(0, 0, 0);
	
	boxes[3].material = white;
	boxes[3].position = vec3(11.0, 0.0, 1001.0);
	boxes[3].halfSize = vec3(1, 1000, 1000);
	boxes[3].rotation = rot(0, 0, 0);*/
	
	planes[0].material = white;
	planes[0].position = vec3(11.0, 0.0, 0.0);
	planes[0].normal = vec3(-1, 0, 0);
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
	
	vec3 col = vec3(0);
	
	scene_init();
	
	for(int i = 0; i < RaysPerPixel; i++)
	{
		col += trace_ray(ray_origin, ray_direction, i%31);
	}
	col /= RaysPerPixel;
	
	col = mix(texture2D(iPrevFrame, gl_FragCoord.xy / iResolution).rgb, col, iSamplePart);
	
	//col += texture2D(iPrevFrame, gl_FragCoord.xy / iResolution).rgb;
	
	fragColor = vec4(col, 1);
}