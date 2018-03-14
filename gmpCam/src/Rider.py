from random import randint


class MyRider:

    def __init__(self, i, xi, yi, max_age):
        self.pid = i
        self.pos = xi, yi
        self.max_age = max_age
        self.done = False
        self.age = 0
        self.rgb = randint(0, 255), randint(0, 255), randint(0, 255)
        self.tracks = []
        self.tracks.append(self.pos)

    def get_pid(self):
        return self.pid

    def update_position(self, xn, yn):
        self.age = 0
        self.pos = xn, yn
        self.tracks.append(self.pos)

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
