import pygame
import math

def rotate(surface, angle, pivot, offset):
    """Rotate the surface around the pivot point.

    Args:
        surface (pygame.Surface): The surface that is to be rotated.
        angle (float): Rotate by this angle.
        pivot (tuple, list, pygame.math.Vector2): The pivot point.
        offset (pygame.math.Vector2): This vector is added to the pivot.
    """
    rotated_image = pygame.transform.rotozoom(surface, -angle, 1)  # Rotate the image.
    rotated_offset = offset.rotate(angle)  # Rotate the offset vector.
    # Add the offset vector to the center/pivot point to shift the rect.
    rect = rotated_image.get_rect(center=pivot+rotated_offset)
    return rotated_image, rect  # Return the rotated image and shifted rect.

def blitRotate(surf, image, pivot, originPos, angle):
    # offset from pivot to center
    image_rect = image.get_rect(topleft = (pivot[0] - originPos[0], pivot[1]-originPos[1]))
    offset_center_to_pivot = pygame.math.Vector2(pivot) - image_rect.center

    # roatated offset from pivot to center
    rotated_offset = offset_center_to_pivot.rotate(-angle)

    # roatetd image center
    rotated_image_center = (pivot[0] - rotated_offset.x, pivot[1] - rotated_offset.y)

    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)
    rotated_image_rect = rotated_image.get_rect(center = rotated_image_center)

    return rotated_image, rotated_image_rect

def calculate_image_offset(image):
    center = image.get_rect().center
    topleft = image.get_rect().topleft
    return (center[0] - topleft[0], center[1] - topleft[1])

def get_top_left(origin_center, offset):
    return (origin_center[0] - offset[0], origin_center[1] - offset[1])

pygame.init()
screen = pygame.display.set_mode((300, 300))
clock = pygame.time.Clock()

image = pygame.image.load('car.png')
w, h = image.get_size()
dupa = image.get_rect()

angle = 45
done = False
pos = (screen.get_width()/4, screen.get_height()/4)

origin_center = pos

w, h = image.get_size()
print(w / 2, h / 2)
topleft = image.get_rect().topleft
center = image.get_rect().center

print(center[0] - topleft[0], center[1] - topleft[1])

clock.tick(45)
for event in pygame.event.get():
    if event.type == pygame.QUIT:
        done = True

screen.fill(0)
offset = calculate_image_offset(image)
screen.blit(image, get_top_left(origin_center, offset))
pygame.draw.circle(screen, (255, 0, 120), (origin_center[0] + offset[0], origin_center[1] + offset[1]), 5)
pygame.draw.circle(screen, (255, 120, 120), origin_center, 5)

pygame.display.flip()
pygame.time.wait(500)

a, b = blitRotate(screen, image, origin_center, (w/2, h/2), 45)
screen.blit(a, b)
pygame.display.flip()
pygame.time.wait(500)

move = (origin_center[0] + 25, origin_center[1] + 25)
offset = calculate_image_offset(a)
screen.blit(a, get_top_left(move, offset))
pygame.draw.circle(screen, (120, 120, 120), move, 5)

pygame.display.flip()
pygame.time.wait(500)

a, b = blitRotate(screen, image, origin_center, (w/2, h/2), 90)
move = (origin_center[0] + 50, origin_center[1] + 50)
offset = calculate_image_offset(a)
screen.blit(a, get_top_left(move, offset))
pygame.draw.circle(screen, (120, 120, 120), move, 5)
pygame.display.flip()
pygame.time.wait(500)

a, b = blitRotate(screen, image, origin_center, (w/2, h/2), 45)
move = (origin_center[0] + 50, origin_center[1] + 50)
offset = calculate_image_offset(a)
screen.blit(a, get_top_left(move, offset))
pygame.draw.circle(screen, (120, 120, 120), move, 5)
pygame.display.flip()
pygame.time.wait(500)



for i in range(0, 100):
    # x = math.cos(math.radians(angle)) * 1
    # # y = math.sin(math.radians(angle)) * 1
    # # xsum += x
    # # ysum += y
    pass
    # screen.fill(0)
    # screen.blit(a, (xsum, ysum))
    # # print(b.topright, b.center)
    # pygame.draw.circle(screen, (120, 0, 255), pos, 5)
    # pygame.draw.circle(screen, (255, 0, 120), (xsum, ysum), 5)
    # pygame.display.flip()
    # pygame.time.wait(10)

pygame.time.wait(5000)

pygame.quit()
exit()