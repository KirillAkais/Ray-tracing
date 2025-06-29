#include "iostream"
#include "fstream"
#include "random"
#include "SFML/Graphics.hpp"

/*
Sorry there are no comments here, I'm too lazy
Basically the programm just loads a shader, draws it in a window and lets you control the camera.
There is a bunch of different scenes in Shaders folder, the current one is hardcoded
To load a scene change the file name in line 93.
Most of the controls are done through the uniforms.

Controls:
WASD to move, mouse to look around
Space/LShift - move up/down
F5 - screenshot (a screenshot is done automatically every 5 minutes)

Arrows:
up/down - aperture
left/right - focus

F/R - FOV
G/T - exposure
H/Y - white
Delete - focus on origin (0,0,0)
Mouse left - reset render progress
*/

#if 0 //1 for high-res, 0 for low

//In case you want to render in one of standart resolutions:
//4k: 3840 2160
//6k: 3240 5760
//8k: 4320 7680
#define WINDOW_H 2048
#define WINDOW_W 2048

#else

#define WINDOW_H 512
#define WINDOW_W 512

#endif

using namespace sf;
using namespace std;

Vector2f operator* (Vector2f vec, double x)
{
	return Vector2f(vec.x * x, vec.y * x);
}

Vector3f operator* (Vector3f vec, double x)
{
	return Vector3f(vec.x * x, vec.y * x, vec.z * x);
}

void main()
{
	int frame_still = 1;

	float matrix_size = 0.00175;
	float fov = 1.51902;//1.525
	float focus = 9;
	//fov = atan(focus);

	float white = 4;
	float exposure = 16;

	int max_samples = 5000;

	ofstream out;
	ifstream in;

	random_device rd;
	mt19937 e2(rd());
	uniform_real_distribution<> dist(0.0f, 1.0f);

	Texture sky;
	sky.loadFromFile("shaders/sky.jpg");

	Texture gradient;
	gradient.loadFromFile("shaders/gradient_sky.png");
	Texture spectrum;
	spectrum.loadFromFile("shaders/spectrum.png");

	Vector3f pos(0, 0, 0.0);
	Vector2f mouse(2*3.1415, -0.5975);

	double sensivity = 0.0005;

	Shader shader;
	shader.loadFromFile("Shaders/Rainbow_ice.glsl", Shader::Fragment);
	Shader post_processing;
	post_processing.loadFromFile("Shaders/post processing.glsl", Shader::Fragment);
	RectangleShape rect;
	rect.setFillColor(Color::Green);
	rect.setSize(Vector2f(WINDOW_W, WINDOW_H));

	RenderWindow window(VideoMode(WINDOW_W, WINDOW_H), "Ray tracing");
	window.setFramerateLimit(60);
	window.setMouseCursorVisible(false);
	//window.setPosition(Vector2i(0,0));

	RenderTexture texture1;
	texture1.create(WINDOW_W, WINDOW_H);
	RenderTexture texture2;
	texture2.create(WINDOW_W, WINDOW_H);
	RenderTexture texture_screenshot;
	texture_screenshot.create(WINDOW_W, WINDOW_H);

	Sprite sprite1;
	sprite1.setTexture(texture1.getTexture());
	Sprite sprite2;
	sprite2.setTexture(texture2.getTexture());

	Clock render_clock;

	while (window.isOpen())
	{
		Event e;
		while (window.pollEvent(e))
		{
			if (e.type == Event::Closed || Keyboard::isKeyPressed(Keyboard::Escape))
			{
				window.close();
			}
			else if (e.type == Event::MouseMoved)
			{
				Vector2f delta = Vector2f(Mouse::getPosition().x - WINDOW_W / 2 - window.getPosition().x, Mouse::getPosition().y - WINDOW_H / 2 - window.getPosition().y) * sensivity;

				if (abs(delta.x) > 0 && abs(delta.y) > 0)
				{
					frame_still = 1;
				}

				mouse += delta;
				Mouse::setPosition(Vector2i(WINDOW_W / 2, WINDOW_H / 2) + window.getPosition());

				if (mouse.x >= 6.283185307)
				{
					mouse.x -= 6.283185307;
				}
				else if (mouse.x <= -6.283185307)
				{
					mouse.x += 6.283185307;
				}
				if (mouse.y >= 1.570796327)
				{
					mouse.y = 1.570796327;
				}
				else if (mouse.y <= -1.570796327)
				{
					mouse.y = -1.570796327;
				}
			}
		}

		if (Keyboard::isKeyPressed(Keyboard::F6))
		{
			out.open("scene.txt");
			out << matrix_size;
			out << " ";
			out << fov;
			out << " ";
			out << focus;
			out << " ";
			out << pos.x;
			out << " ";
			out << pos.y;
			out << " ";
			out << pos.z;
			out << " ";
			out << mouse.x;
			out << " ";
			out << mouse.y;
			out << " ";
			out.close();
		}
		if (Keyboard::isKeyPressed(Keyboard::F7))
		{
			in.open("scene.txt");
			in >> matrix_size;
			in >> fov;
			in >> focus;
			in >> pos.x;
			in >> pos.y;
			in >> pos.z;
			in >> mouse.x;
			in >> mouse.y;
			in.close();
			frame_still = 1;
		}
		if (Keyboard::isKeyPressed(Keyboard::W))
		{
			pos += Vector3f(cos(mouse.x), sin(mouse.x), 0) * 0.1;
			frame_still = 1;
		}
		else if (Keyboard::isKeyPressed(Keyboard::S))
		{
			pos -= Vector3f(cos(mouse.x), sin(mouse.x), 0) * 0.1;
			frame_still = 1;
		}
		else if (Keyboard::isKeyPressed(Keyboard::D))
		{
			pos += Vector3f(cos(mouse.x + 1.570796327), sin(mouse.x + 1.570796327), 0) * 0.1;
			frame_still = 1;
		}
		else if (Keyboard::isKeyPressed(Keyboard::A))
		{
			pos -= Vector3f(cos(mouse.x + 1.570796327), sin(mouse.x + 1.570796327), 0) * 0.1;
			frame_still = 1;
		}
		else if (Keyboard::isKeyPressed(Keyboard::Space))
		{
			pos += Vector3f(0, 0, 1) * 0.1;
			frame_still = 1;
		}
		else if (Keyboard::isKeyPressed(Keyboard::LShift))
		{
			pos -= Vector3f(0, 0, 1) * 0.1;
			frame_still = 1;
		}
		else if (Keyboard::isKeyPressed(Keyboard::Up))
		{
			matrix_size += 0.0005;
			frame_still = 1;
		}
		else if (Keyboard::isKeyPressed(Keyboard::Down))
		{
			matrix_size -= 0.0005;
			frame_still = 1;
		}
		else if (Keyboard::isKeyPressed(Keyboard::Left))
		{
			focus -= 0.05;
			fov = atan(focus);
			frame_still = 1;
		}
		else if (Keyboard::isKeyPressed(Keyboard::Right))
		{
			focus += 0.05;
			fov = atan(focus);
			frame_still = 1;
		}
		else if (Keyboard::isKeyPressed(Keyboard::R))
		{
			fov += 0.005 / focus;
			frame_still = 1;
		}
		else if (Keyboard::isKeyPressed(Keyboard::F))
		{
			fov -= 0.005 / focus;
			frame_still = 1;
		}
		else if (Keyboard::isKeyPressed(Keyboard::G))
		{
			exposure -= 0.05;
			frame_still = 1;
		}
		else if (Keyboard::isKeyPressed(Keyboard::T))
		{
			exposure += 0.05;
			frame_still = 1;
		}
		else if (Keyboard::isKeyPressed(Keyboard::H))
		{
			white -= 0.05;
			frame_still = 1;
		}
		else if (Keyboard::isKeyPressed(Keyboard::Y))
		{
			white += 0.05;
			frame_still = 1;
		}
		else if (Keyboard::isKeyPressed(Keyboard::Delete))
		{
			focus = sqrt(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z);
			frame_still = 1;
		}

		if (Mouse::isButtonPressed(Mouse::Left))
		{
			frame_still = 1;
		}

		if (frame_still < 2)
		{
			render_clock.restart();
		}

		shader.setUniform("iResolution", Vector2f(WINDOW_W, WINDOW_H));
		post_processing.setUniform("iResolution", Vector2f(WINDOW_W, WINDOW_H));
		shader.setUniform("iPos", pos);
		shader.setUniform("iMouse", mouse);
		shader.setUniform("iSky", sky);
		shader.setUniform("iGradient", gradient);
		shader.setUniform("iSpectrum", spectrum);
		shader.setUniform("iSeed1", Vector2f((float)dist(e2), (float)dist(e2)) * 999.0f);
		shader.setUniform("iSeed2", Vector2f((float)dist(e2), (float)dist(e2)) * 999.0f);
		shader.setUniform("iSamplePart", 1.0f / frame_still);
		shader.setUniform("iMatrixSize", matrix_size);
		shader.setUniform("iFov", fov);
		shader.setUniform("iFocus", focus);
		shader.setUniform("iWhite", white);
		shader.setUniform("iExposure", exposure);
		shader.setUniform("iAngle", 0.001f * frame_still);
		//if (frame_still == 1)
		{
			//shader.setUniform("iSamplePart", 1.0f);
			//shader.setUniform("iPos", Vector3f(-1000.0f,0.0f,0.0f));
		}
		window.clear(Color::White);

		if (frame_still % 2 == 1)
		{
			shader.setUniform("iPrevFrame", texture2.getTexture());
			texture1.clear();
			texture1.draw(rect, &shader);
			texture1.display();
			post_processing.setUniform("iImage", texture1.getTexture());
			window.draw(sprite1, &post_processing);
		}
		else
		{
			shader.setUniform("iPrevFrame", texture1.getTexture());
			texture2.clear();
			texture2.draw(rect, &shader);
			texture2.display();
			post_processing.setUniform("iImage", texture2.getTexture());
			window.draw(sprite2, &post_processing);
		}
		//sleep(seconds(8));
		if (Keyboard::isKeyPressed(Keyboard::F5))
		{
			texture_screenshot.clear();
			texture_screenshot.draw(sprite1, &post_processing);
			texture_screenshot.display();
			texture_screenshot.getTexture().copyToImage().saveToFile("screenshot_" + to_string(frame_still) + ".png");
			cout << frame_still << endl;
		}
		
		if (render_clock.getElapsedTime().asSeconds() > 300 || frame_still == max_samples)
		{
			texture_screenshot.clear();
			texture_screenshot.draw(sprite1, &post_processing);
			texture_screenshot.display();
			texture_screenshot.getTexture().copyToImage().saveToFile("screenshot_" + to_string(frame_still) + ".png");
			render_clock.restart();
			cout << frame_still << endl;
		}
		//if (frame_still >= 1000)
		//{
		//	texture1.getTexture().copyToImage().saveToFile("screenshot_" + to_string(frame_still) + ".png");
		//	window.close();
		//}

		window.display();
		frame_still++;
	}
}