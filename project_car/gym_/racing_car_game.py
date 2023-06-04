import pygame
import math
from enum import Enum
import logging

class CarAction(Enum):
    NO_ACTION = 1
    TURN_RIGHT = 2
    TURN_LEFT = 3

screen_width = 720
screen_hight = 480

class Car:
    def __init__(self, center_position : tuple) -> None:
        self._car_origin = pygame.image.load("car.png")
        self._car_origin_center = center_position
        self._car_origin_offset = self._calculate_image_offset(self._car_origin)
        self._angle = 0
        self._speed = 1

        image, rect = self.rotate(self._car_origin, self._car_origin_center,
                                    self._car_origin_offset, self._angle)
        self.car_moved = image
        self.car_moved_center = rect.center
        self.car_moved_offset = self._calculate_image_offset(self.car_moved)

        self.position = self._car_origin_center

    def _get_top_left(self, origin_center, offset):
        return (origin_center[0] - offset[0], origin_center[1] - offset[1])

    def _calculate_image_offset(self, image):
        center = image.get_rect().center
        topleft = image.get_rect().topleft
        return (center[0] - topleft[0], center[1] - topleft[1])

    def _calculate_center_position(self, right_top_position : tuple) -> tuple:
        half_width = self._car_origin.get_width() / 2
        half_height = self._car_origin.get_height() / 2
        return right_top_position[0] + half_width, right_top_position[1] + half_height

    def rotate(self, image, pivot, originPos, _angle):
        # offset from pivot to center
        image_rect = image.get_rect(topleft = (pivot[0] - originPos[0], pivot[1]-originPos[1]))
        offset_center_to_pivot = pygame.math.Vector2(pivot) - image_rect.center

        # roatated offset from pivot to center
        rotated_offset = offset_center_to_pivot.rotate(-_angle)

        # roatetd image center
        rotated_image_center = (pivot[0] - rotated_offset.x, pivot[1] - rotated_offset.y)

        # get a rotated image
        rotated_image = pygame.transform.rotate(image, _angle)
        rotated_image_rect = rotated_image.get_rect(center = rotated_image_center)

        return rotated_image, rotated_image_rect

    def move(self, action):
        if action == 1:
            self._angle += 5
        elif action == 2:
            self._angle -= 5

        if action == 1 or action == 2:
            image, rect = self.rotate(self._car_origin, self._car_origin_center, self._car_origin_offset, self._angle)
            self.car_moved = image
            self.car_moved_center = rect.center
            self.car_moved_offset = self._calculate_image_offset(self.car_moved)

        # if action == 3 and self._speed < 7:
        #     self._speed += 1

        x = math.sin(math.radians(self._angle)) * self._speed
        y = math.cos(math.radians(self._angle)) * self._speed
        self.position = (self.position[0] + x, self.position[1] + y)

    def draw(self, screen : pygame.Surface):
        screen.blit(self.car_moved, self._get_top_left(self.position, self.car_moved_offset))

    def get_corners(self) -> list[tuple]:
        w, h = self._car_origin.get_size()
        half_diagonal = math.sqrt(w** 2 + h ** 2) / 2
        tang = w / h
        alpha = math.atan(tang)


        x1 = math.sin(alpha + math.radians(self._angle)) * half_diagonal + self.position[0]
        y1 = math.cos(alpha + math.radians(self._angle)) * half_diagonal + self.position[1]

        x2 = math.sin(-alpha + math.radians(self._angle)) * half_diagonal + self.position[0]
        y2 = math.cos(-alpha + math.radians(self._angle)) * half_diagonal + self.position[1]

        x3 = math.sin(math.radians(180 + self._angle) - alpha) * half_diagonal + self.position[0]
        y3 = math.cos(math.radians(180 + self._angle) - alpha) * half_diagonal + self.position[1]

        x4 = math.sin(math.radians(180 + self._angle) + alpha) * half_diagonal + self.position[0]
        y4 = math.cos(math.radians(180 + self._angle) + alpha) * half_diagonal + self.position[1]

        return  [(x1, y1), (x2, y2), (x3, y3),(x4, y4)]

    def reset_to_origin(self):
        self._angle = 0

        image, rect = self.rotate(self._car_origin, self._car_origin_center,
                                    self._car_origin_offset, self._angle)
        self.car_moved = image
        self.car_moved_center = rect.center
        self.car_moved_offset = self._calculate_image_offset(self.car_moved)

        self.position = self._car_origin_center

    def get_speed(self):
        return self._speed

    @property
    def angle(self):
        return self._angle


class RaceTrack:
    def __init__(self, path_to_track) -> None:
        self._track = pygame.image.load(path_to_track)

    def draw(self, screen : pygame.Surface):
        screen.blit(self._track, (0, 0))

    def get_size(self) -> tuple:
        return (self._track.get_width(), self._track.get_height())

    def get_start(self):
        for i in range(0 , self._track.get_width()):
            for j in range(0, self._track.get_height()):
                if (self._track.get_at((i,j)) == (0, 255, 255, 255)):
                    return (i, j)

    def get_point(self, position: tuple) -> tuple:
        return self._track.get_at((int(position[0]), int(position[1])))

    def get_surface(self):
        return self._track

class CarRadars:
    MAX_LEN = 50
    def __init__(self, ) -> None:
        self._radars = [
            [(0,0), 0, -90],
            # [(0,0), 0, -45],
            [(0,0), 0,   0],
            # [(0,0), 0,  45],
            [(0,0), 0,  90],
        ]

    def update(self, car_center, car_angle, track: pygame.Surface):
        for radar in self._radars:
            len = 0
            _, _, degree = radar
            x = int(car_center[0] + math.sin(math.radians(car_angle + degree)) * len)
            y = int(car_center[1] + math.cos(math.radians(car_angle + degree)) * len)
            while track.get_at((x, y)) == (255, 255, 255, 255) and len < self.MAX_LEN:
                len += 1
                x = int(car_center[0] + math.sin(math.radians(car_angle + degree)) * len)
                y = int(car_center[1] + math.cos(math.radians(car_angle + degree)) * len)

            radar[0] = (x, y)
            radar[1] = len

    def draw(self, surface: pygame.Surface, car_center):
        for radar in self._radars:
            pos, _, _ = radar
            pygame.draw.line(surface, (0, 0, 255), car_center, pos, 2)
            pygame.draw.circle(surface, (0, 150, 200), pos, 4)

    def get_radars_length(self) -> list:
        return [radar[1] for radar in self._radars]

    def reset(self):
        self._radars = [
            [(0,0), 0, -90],
            [(0,0), 0, -45],
            [(0,0), 0,   0],
            [(0,0), 0,  45],
            [(0,0), 0,  90],
        ]



class CarGame2D:
    def __init__(self) -> None:
        self.track = RaceTrack("mapa3.png")
        self.car = Car(self.track.get_start())
        self.radars = CarRadars()
        screen_size = self.track.get_size()
        self.reward_text_pos = (screen_size[0] - 150, 10)
        self._colision = False
        self._finish = False
        self._reward = 0


    def start_visualization(self):
        if pygame.get_init() is False:
            pygame.init()
            self.font = pygame.font.SysFont("comicsansms", 24)
            self.clock = pygame.time.Clock()
            self.screen = pygame.display.set_mode(self.track.get_size())


    def check_exit(self):
        if self._colision is True:
            logging.info("Colsion :C")
            return True

        if self._finish is True:
            logging.info("Finish")
            return True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

        return False

    def check_colistion(self) -> None:
        for point in self.car.get_corners():
            if self.track.get_point(point) == (0, 0, 0, 255):
                self._colision = True

    def check_end(self) -> None:
        for point in self.car.get_corners():
            if self.track.get_point(point) == (255, 0, 0, 255):
                self._finish = True

    def action(self, action):
        self.car.move(action)
        self.radars.update(self.car.position, self.car._angle, self.track.get_surface())
        self.check_colistion()
        self.check_end()

    def evaluate(self):
        radars_len = self.radars.get_radars_length()
        reward = (sum(radars_len) / 1000)   # sum of radar lengths (max 150) / 100 -> max 1.5
        reward += self.car.get_speed() * 10

        if self._finish is True:
            self._reward += 1000
        self._reward += reward

        if self._colision is True:
            self._reward -= 1000

        return reward

    def observe(self):
        radars_len = self.radars.get_radars_length()
        return tuple(radars_len)

    def is_done(self):
        if self._finish is True or self._colision is True:
            return True

        return False

    def reset(self):
        self.car.reset_to_origin()
        self.radars.reset()
        self._reward = 0
        self._finish = False
        self._colision = False

    def view(self):
        self.track.draw(self.screen)
        self.car.draw(self.screen)
        self.radars.draw(self.screen, self.car.position)

        corner = self.car.get_corners()
        for i in corner:
            pygame.draw.circle(self.screen, (123, 200, 255), i, 1)

        reward_text = self.font.render("Reward: %.2f" % self._reward, False, (255, 123, 123))
        speed_text = self.font.render("Speed: %.2f" % self.car.get_speed(), False, (255, 123, 123))
        self.screen.blit(reward_text, self.reward_text_pos)
        self.screen.blit(speed_text, (self.reward_text_pos[0], 40))
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if pygame.get_init() is True:
            pygame.quit()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    game = CarGame2D()
    while game.check_exit() == False:
        game.action(0)
        game.evaluate()
        game.view()
        game.is_done()