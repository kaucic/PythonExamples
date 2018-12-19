# -*- coding: utf-8 -*-
"""
Asteroids game

@author: Kimberly
"""

# program template for Spaceship
try:
    import simplegui
except ImportError:
    import SimpleGUICS2Pygame.simpleguics2pygame as simplegui

import math
import random

# globals for user interface
WIDTH = 800
HEIGHT = 600
score = 0
lives = 3
time = 0

class ImageInfo:
    def __init__(self, center, size, radius = 0, lifespan = None, animated = False):
        self.center = center
        self.size = size
        self.radius = radius
        if lifespan:
            self.lifespan = lifespan
        else:
            self.lifespan = float('inf')
        self.animated = animated

    def get_center(self):
        return self.center

    def get_size(self):
        return self.size

    def get_radius(self):
        return self.radius

    def get_lifespan(self):
        return self.lifespan

    def get_animated(self):
        return self.animated

    
# art assets created by Kim Lathrop, may be freely re-used in non-commercial projects, please credit Kim
    
# debris images - debris1_brown.png, debris2_brown.png, debris3_brown.png, debris4_brown.png
#                 debris1_blue.png, debris2_blue.png, debris3_blue.png, debris4_blue.png, debris_blend.png
debris_info = ImageInfo([320, 240], [640, 480])
debris_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/debris2_blue.png")

# nebula images - nebula_brown.png, nebula_blue.png
nebula_info = ImageInfo([400, 300], [800, 600])
nebula_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/nebula_blue.f2014.png")

# splash image
splash_info = ImageInfo([200, 150], [400, 300])
splash_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/splash.png")

# ship image
ship_info = ImageInfo([45, 45], [90, 90], 35)
ship_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/double_ship.png")

# missile image - shot1.png, shot2.png, shot3.png
missile_info = ImageInfo([5,5], [10, 10], 3, 50)
missile_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/shot2.png")

# asteroid images - asteroid_blue.png, asteroid_brown.png, asteroid_blend.png
asteroid_info = ImageInfo([45, 45], [90, 90], 40)
asteroid_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/asteroid_blue.png")

# animated explosion - explosion_orange.png, explosion_blue.png, explosion_blue2.png, explosion_alpha.png
explosion_info = ImageInfo([64, 64], [128, 128], 17, 24, True)
explosion_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/explosion_alpha.png")

# sound assets purchased from sounddogs.com, please do not redistribute
soundtrack = simplegui.load_sound("http://commondatastorage.googleapis.com/codeskulptor-assets/sounddogs/soundtrack.mp3")
missile_sound = simplegui.load_sound("http://commondatastorage.googleapis.com/codeskulptor-assets/sounddogs/missile.mp3")
missile_sound.set_volume(.5)
ship_thrust_sound = simplegui.load_sound("http://commondatastorage.googleapis.com/codeskulptor-assets/sounddogs/thrust.mp3")
explosion_sound = simplegui.load_sound("http://commondatastorage.googleapis.com/codeskulptor-assets/sounddogs/explosion.mp3")

# helper functions to handle transformations
def angle_to_vector(ang):
    return [math.cos(ang), math.sin(ang)]

def dist(p,q):
    return math.sqrt((p[0] - q[0]) ** 2+(p[1] - q[1]) ** 2)


# Ship class
class Ship:
    def __init__(self, pos, vel, angle, image, info):
        self.pos = [pos[0],pos[1]]
        self.vel = [vel[0],vel[1]]
        self.thrust = False
        self.angle = angle
        self.angle_vel = 0.0
        self.image = image
        self.image_center = info.get_center()
        self.image_size = info.get_size()
        self.radius = info.get_radius()
        
    def draw(self,canvas):
        #canvas.draw_circle(self.pos, self.radius, 1, "White", "White")
        if (self.thrust):
            ship_thrust_sound.play()            
            new_x = self.image_center[0] + self.image_size[0]
            new_y = self.image_center[1]
            canvas.draw_image(self.image,[new_x, new_y],self.image_size,self.pos,self.image_size,self.angle)
        else:
            ship_thrust_sound.rewind()
            canvas.draw_image(self.image,self.image_center,self.image_size,self.pos,self.image_size,self.angle)

    def update(self):
        DRAG_PERCENTAGE = 0.004
        self.pos[0] = (self.pos[0] + self.vel[0]) % WIDTH
        self.pos[1] = (self.pos[1] + self.vel[1]) % HEIGHT
        self.vel[0] *= (1.0 - DRAG_PERCENTAGE)
        self.vel[1] *= (1.0 - DRAG_PERCENTAGE)
        self.angle += self.angle_vel
        
    def turn_thrusters_on(self):
        self.thrust = True;
        THRUST_MAG = 4
        dx,dy = angle_to_vector(self.angle)
        self.vel[0] += (THRUST_MAG * dx)
        self.vel[1] += (THRUST_MAG * dy)
    
    def turn_thrusters_off(self):
        self.thrust = False;
    
    def shoot(self):
        global a_missile
        MISSILE_SPEED_MULTIPLIER = 2
        dx,dy = angle_to_vector(self.angle)
        pos_x = self.pos[0] + self.image_center[0] * dx
        pos_y = self.pos[1] + self.image_center[1] * dy
        vel_x = self.vel[0] + MISSILE_SPEED_MULTIPLIER * dx
        vel_y = self.vel[1] + MISSILE_SPEED_MULTIPLIER * dy
        a_missile = Sprite([pos_x,pos_y], [vel_x,vel_y],0,0, missile_image,missile_info,missile_sound)
        
    def increase_rotation(self,angle_increment):
        self.angle_vel += angle_increment
        
    def stop_rotating(self):
        self.angle_vel = 0.0;
    
# Sprite class
class Sprite:
    def __init__(self, pos, vel, ang, ang_vel, image, info, sound = None):
        self.pos = [pos[0],pos[1]]
        self.vel = [vel[0],vel[1]]
        self.angle = ang
        self.angle_vel = ang_vel
        self.image = image
        self.image_center = info.get_center()
        self.image_size = info.get_size()
        self.radius = info.get_radius()
        self.lifespan = info.get_lifespan()
        self.animated = info.get_animated()
        self.age = 0
        if sound:
            sound.rewind()
            sound.play()
   
    def draw(self, canvas):
        #canvas.draw_circle(self.pos, self.radius, 1, "Red", "Red")
        canvas.draw_image(self.image,self.image_center,self.image_size,self.pos,self.image_size,self.angle)
    
    def update(self):
        self.pos[0] = (self.pos[0] + self.vel[0]) % WIDTH
        self.pos[1] = (self.pos[1] + self.vel[1]) % HEIGHT
        self.angle += self.angle_vel
           
def draw(canvas):
    global time
    
    # animiate background
    time += 1
    wtime = (time / 4) % WIDTH
    center = debris_info.get_center()
    size = debris_info.get_size()
    canvas.draw_image(nebula_image, nebula_info.get_center(), nebula_info.get_size(), [WIDTH / 2, HEIGHT / 2], [WIDTH, HEIGHT])
    canvas.draw_image(debris_image, center, size, (wtime - WIDTH / 2, HEIGHT / 2), (WIDTH, HEIGHT))
    canvas.draw_image(debris_image, center, size, (wtime + WIDTH / 2, HEIGHT / 2), (WIDTH, HEIGHT))

    # draw ship and sprites
    my_ship.draw(canvas)
    a_rock.draw(canvas)
    a_missile.draw(canvas)
    
    # update ship and sprites
    my_ship.update()
    a_rock.update()
    a_missile.update()
    
    # print out the score and remaining lives
    canvas.draw_text("Score: %d" % score,[10, 20], 18, "Blue")
    canvas.draw_text("Lives: %d" % lives,[10, 40], 18, "Red")

def key_pressed_down(key):
    print "Key down=", key
    ANGLE_INCREMENT = 0.1
    if (key == 37): # "left"
        my_ship.increase_rotation(-ANGLE_INCREMENT)
    elif (key == 39): # "right"
        my_ship.increase_rotation(ANGLE_INCREMENT)
    elif (key == 38): # "up"
        my_ship.turn_thrusters_on()
    elif (key == 32): # "space bar"
        my_ship.shoot()
    else:
        pass

def key_up(key):
    print "Key up=", key
    if (key == 37): # "left"
        my_ship.stop_rotating()
    elif (key == 39): # "right"
        my_ship.stop_rotating()
    elif (key == 38): # "up"
        my_ship.turn_thrusters_off()
    else:
        pass

# timer handler that spawns a rock    
def rock_spawner():
    global a_rock
    ROCK_VELOCITY = 6.0
    ROCK_ANGLE_VEL = 0.4
    pos_x = random.randint(50,WIDTH-50)
    pos_y = random.randint(50,HEIGHT-50)
    vel_x = random.random()*ROCK_VELOCITY - 0.5*ROCK_VELOCITY
    vel_y = random.random()*ROCK_VELOCITY - 0.5*ROCK_VELOCITY
    ang_vel = random.random()*ROCK_ANGLE_VEL - 0.5*ROCK_ANGLE_VEL
    a_rock = Sprite([pos_x,pos_y],[vel_x, vel_y],0,ang_vel,asteroid_image,asteroid_info)
    
# initialize frame
frame = simplegui.create_frame("Asteroids", WIDTH, HEIGHT)

# initialize ship and two sprites
my_ship = Ship([WIDTH / 2, HEIGHT / 2], [-1, 2.5], -math.pi/2, ship_image, ship_info)
a_rock = Sprite([WIDTH / 3, HEIGHT / 3], [2.5, -1], 0, -0.1, asteroid_image, asteroid_info)
a_missile = Sprite([2 * WIDTH / 3, 2 * HEIGHT / 3], [-1,1], 0, 0, missile_image, missile_info)

# register handlers
frame.set_draw_handler(draw)
frame.set_keydown_handler(key_pressed_down)
frame.set_keyup_handler(key_up)

timer = simplegui.create_timer(1000.0, rock_spawner)

# get things rolling
timer.start()
frame.start()
