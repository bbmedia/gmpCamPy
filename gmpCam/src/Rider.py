from random import randint


class MyRider:
    tracks = []

    def __init__(self, i, xi, yi, age):
        self.i = i
        self.pos = xi, yi
        self.age = age
        self.done = False
        self.rgb = randint(0, 255), randint(0, 255), randint(0, 255)

    def update_position(self, xn, yn):
        self.age = 0
        self.tracks.append(self.pos)
        self.pos = xn, yn

    def get_x(self):
        return self.pos[0]

    def get_y(self):
        return self.pos[1]

    def is_done(self):
        self.done = True

    def get_age(self):
        return self.age

    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True

    def timed_out(self):
        return self.done

    def get_rgb(self):
        return self.rgb
