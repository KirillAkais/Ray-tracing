#version 460

#if 0

#define Iterations 8
#define RaysPerPixel 128

#else

#define Iterations 8
#define RaysPerPixel 310

#endif
#define MaxDistance 10000.0
#define N_IN 1.33
#define N_OUT 1.0

#define TriPrismCount 1
#define OctahedronCount 1

#define SunBrightness 1000

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
	int population;
	float h;
	float rad;
	mat3 rotation;
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

struct Triangle
{
    vec3 v0;
	vec3 v1;
	vec3 v2;
	bool inv_normal;
};

struct Octahedron
{
    Material material;
    vec3 position;
	mat3 rotation;
    float radius;
	Triangle[8] faces;
};

struct TriPrism
{
    Material material;
    vec3 position;
	mat3 rotation;
    float radius;
	float height;
	Triangle[8] faces;
};

Octahedron octahedrons[OctahedronCount];
TriPrism triprisms[TriPrismCount];
//Box boxes[BoxCount];

Octahedron make_oct(float r)
{
	Octahedron o;
	o.faces[0].v0 = vec3(0,0,r);
	o.faces[0].v1 = vec3(0,r,0);
	o.faces[0].v2 = vec3(r,0,0);
	o.faces[0].inv_normal = true;
	o.faces[1].v0 = vec3(0,0,-r);
	o.faces[1].v1 = vec3(0,r,0);
	o.faces[1].v2 = vec3(r,0,0);
	o.faces[1].inv_normal = false;
	o.faces[2].v0 = vec3(0,0,r);
	o.faces[2].v1 = vec3(0,-r,0);
	o.faces[2].v2 = vec3(r,0,0);
	o.faces[2].inv_normal = false;
	o.faces[3].v0 = vec3(0,0,-r);
	o.faces[3].v1 = vec3(0,-r,0);
	o.faces[3].v2 = vec3(r,0,0);
	o.faces[3].inv_normal = true;
	o.faces[4].v0 = vec3(0,0,r);
	o.faces[4].v1 = vec3(0,r,0);
	o.faces[4].v2 = vec3(-r,0,0);
	o.faces[4].inv_normal = false;
	o.faces[5].v0 = vec3(0,0,-r);
	o.faces[5].v1 = vec3(0,r,0);
	o.faces[5].v2 = vec3(-r,0,0);
	o.faces[5].inv_normal = true;
	o.faces[6].v0 = vec3(0,0,r);
	o.faces[6].v1 = vec3(0,-r,0);
	o.faces[6].v2 = vec3(-r,0,0);
	o.faces[6].inv_normal = true;
	o.faces[7].v0 = vec3(0,0,-r);
	o.faces[7].v1 = vec3(0,-r,0);
	o.faces[7].v2 = vec3(-r,0,0);
	o.faces[7].inv_normal = false;
	o.radius = r;
	return o;
}

TriPrism make_triprism(float r, float h)
{
	TriPrism o;
	float h2 = h / 2.0;
	float rt = r * 0.8660254;
	float r2 = r / 2.0;
	o.faces[0].v0 = vec3(-r2,-rt,h2);
	o.faces[0].v1 = vec3(-r2,rt,h2);
	o.faces[0].v2 = vec3(r,0,h2);
	o.faces[0].inv_normal = true;
	o.faces[1].v0 = vec3(-r2,rt,-h2);
	o.faces[1].v1 = vec3(-r2,-rt,-h2);
	o.faces[1].v2 = vec3(r,0,-h2);
	o.faces[1].inv_normal = true;
	o.faces[2].v0 = vec3(-r2,-rt,h2);
	o.faces[2].v1 = vec3(-r2,rt,h2);
	o.faces[2].v2 = vec3(-r2,-rt,-h2);
	o.faces[2].inv_normal = false;
	o.faces[3].v0 = vec3(-r2,rt,-h2);
	o.faces[3].v1 = vec3(-r2,-rt,-h2);
	o.faces[3].v2 = vec3(-r2,rt,h2);
	o.faces[3].inv_normal = false;
	o.faces[4].v0 = vec3(-r2,rt,h2);
	o.faces[4].v1 = vec3(-r2,rt,-h2);
	o.faces[4].v2 = vec3(r,0,h2);
	o.faces[4].inv_normal = true;
	o.faces[5].v0 = vec3(r,0,-h2);
	o.faces[5].v1 = vec3(r,0,h2);
	o.faces[5].v2 = vec3(-r2,rt,-h2);
	o.faces[5].inv_normal = true;
	o.faces[6].v0 = vec3(-r2,-rt,h2);
	o.faces[6].v1 = vec3(-r2,-rt,-h2);
	o.faces[6].v2 = vec3(r,0,h2);
	o.faces[6].inv_normal = false;
	o.faces[7].v0 = vec3(r,0,-h2);
	o.faces[7].v1 = vec3(r,0,h2);
	o.faces[7].v2 = vec3(-r2,-rt,-h2);
	o.faces[7].inv_normal = false;
	o.radius = r;
	return o;
}

float n[31] = float[31](
2.464126948,
2.459756371,
2.455728373,
2.452007123,
2.448561471,
2.445364223,
2.442391531,
2.439622406,
2.437038306,
2.434622791,
2.432361245,
2.430240628,
2.428249274,
2.426376723,
2.424613564,
2.422951316,
2.421382316,
2.419899624,
2.418496941,
2.417168542,
2.41590921,
2.414714183,
2.413579108,
2.412499998,
2.411473195,
2.410495338,
2.409563335,
2.408674337,
2.407825713,
2.407015033,
2.40624005
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

bool IntersectRayTri(Ray r, Triangle tr, out float fraction, out vec3 normal)
{
	vec3 ro = r.origin;
	vec3 rd = r.direction;
	
    vec3 v1v0 = tr.v1 - tr.v0;
    vec3 v2v0 = tr.v2 - tr.v0;
    vec3 rov0 = ro - tr.v0;
    vec3  n = cross( v1v0, v2v0 );
    vec3  q = cross( rov0, rd );
    float d = 1.0/dot( rd, n );
    float u = d*dot( -q, v2v0 );
    float v = d*dot(  q, v1v0 );
    float t = d*dot( -n, rov0 );
	fraction = t;
	normal = normalize(n);
	if (tr.inv_normal)	normal = -normal;
    if( u<0.0 || v<0.0 || (u+v)>1.0 || t < 0) return false;
    return true;
}

bool IntersectRayOct(Ray r, Octahedron o, out float fraction, out vec3 normal)
{
	//r.pos = vec3(0);
	o.rotation = r.rotation;
	o.position = r.pos;
	//o.radius = r.rad;
	//o.height = r.h;
	r.direction = r.direction * o.rotation;
	r.origin = (r.origin - o.position) * o.rotation;
	float minDistance = MaxDistance;

    for (int i = 0; i < 8; i++)
    {
        float D;
        vec3 N;
        if (IntersectRayTri(r, o.faces[i], D, N) && D < minDistance)
        {
            minDistance = D;
            normal = N;
        }
    }
	normal *= transpose(o.rotation);
    fraction = minDistance;
    return minDistance != MaxDistance;
}

bool IntersectRayTriPrism(Ray r, TriPrism tp, out float fraction, out vec3 normal)
{
	tp.rotation = r.rotation;
	tp.position = r.pos;
	//tp.radius = r.rad;
	//tp.height = r.h;
	r.direction = r.direction * tp.rotation;
	r.origin = (r.origin - tp.position) * tp.rotation;
	float minDistance = MaxDistance;

    for (int i = 0; i < 8; i++)
    {
        float D;
        vec3 N;
        if (IntersectRayTri(r, tp.faces[i], D, N) && D < minDistance)
        {
            minDistance = D;
            normal = N;
        }
    }
	normal *= transpose(tp.rotation);
    fraction = minDistance;
    return minDistance != MaxDistance;
}

vec3 get_sun(vec3 rd)
{
	vec2 uv = vec2(asin(rd.z) * 2, 1.5) / 3.14159265;
	uv = uv * 0.5 + 0.5;
	vec3 light = normalize(vec3(1, 0, 0.1));
	vec3 col = texture2D(iGradient, uv).rgb;
	vec3 sun = vec3(1., 1., 1.0);
	if(dot(rd, light)<0.99995)
	{
		sun *= max(0, pow(dot(rd, light), 8192))/1000;
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
	if(r.population == 0)
	{
		if (IntersectRayOct(r, octahedrons[0], D, N) && D < minDistance)
		{
			minDistance = D;
			normal = N;
			material = octahedrons[0].material;
		}
	}
	else
	{
		if (IntersectRayTriPrism(r, triprisms[0], D, N) && D < minDistance)
		{
			minDistance = D;
			normal = N;
			material = triprisms[0].material;
		}
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
	ray.pos = ro + rd*10.0 + random_sphere()*1.045;
	ray.population = int(random()*198)%2;
	//ray.population = 1;
	
	
	ray.rotation = rot(random()*PI2,random()*PI2,0) * rot(random()*PI2,0,random()*PI2)*rot(0,random()*PI2,random()*PI2);
	
	if(ray.population==1)	ray.rotation = rot(gauss_dist(0,0.03),0,random()*PI2);
	//else if(ray.population==2)	ray.rotation = rot(random()*PI2,random()*PI2,random()*PI2);
	
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
	//return normal;
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

	octahedrons[0] = make_oct(1);
	octahedrons[0].material = drop;
	octahedrons[0].position = vec3(0);
	octahedrons[0].rotation = rot(0,0,0);
	
	triprisms[0] = make_triprism(1,0.2);
	triprisms[0].material = drop;
	triprisms[0].position = vec3(0);
	triprisms[0].rotation = rot(0,0,0);
}

void main (void)
{
	vec2 coord = gl_FragCoord.xy / iResolution - vec2(0.5);
	//coord.x /= iResolution.y / iResolution.x;
	
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