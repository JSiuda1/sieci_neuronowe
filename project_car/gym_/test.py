
import pygame
import math

pygame.init()
screen = pygame.display.set_mode((300, 300))
clock = pygame.time.Clock()

def blitRotate(surf, image, pos, originPos, angle):
    # pos - aktualna pozycja on surface
    # originPos - (polowa szerokosci, polowa wysokosci) -> const
    # image - aktualny obrazek

    # offset from pivot to center
    image_rect = image.get_rect(topleft = (pos[0] - originPos[0], pos[1]-originPos[1]))
    offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center

    # roatated offset from pivot to center
    rotated_offset = offset_center_to_pivot.rotate(-angle)

    # roatetd image center
    rotated_image_center = (pos[0] - rotated_offset.x, pos[1] - rotated_offset.y)

    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)
    rotated_image_rect = rotated_image.get_rect(center = rotated_image_center)

    # rotate and blit the image
    surf.blit(rotated_image, rotated_image_rect)

    # draw rectangle around the image
    pygame.draw.rect(surf, (255, 0, 0), (*rotated_image_rect.topleft, *rotated_image.get_size()),2)

    return rotated_image, rotated_image_rect

def blitRotate2(surf, image, topleft, angle):

    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)

    surf.blit(rotated_image, new_rect.topleft)
    pygame.draw.rect(surf, (255, 0, 0), new_rect, 2)

image = pygame.image.load('car.png')
w, h = image.get_size()
dupa = image.get_rect()

angle = 45
done = False
pos = (screen.get_width()/4, screen.get_height()/4)

clock.tick(45)
for event in pygame.event.get():
    if event.type == pygame.QUIT:
        done = True

screen.fill(0)
pygame.time.wait(500)
origin_center = image.get_rect().center
top_left = image.get_rect().topleft
print(top_left)
center = image.get_rect().center
offset = (top_left[0] - center[0], top_left[1] - center[1])

print(offset)

a, b = blitRotate(screen, image, pos, (w/2, h/2), 45)
top_left = a.get_rect().topleft
center = a.get_rect().center
offset = (top_left[0] - center[0], top_left[1] - center[1])
print(f"Offset after rotation {offset}")
print(f"Origin center {origin_center}")
print(top_left)
xsum = b.topleft[0]
ysum = b.topleft[1]
for i in range(0, 100):
    # x = math.cos(math.radians(angle)) * 1
    # y = math.sin(math.radians(angle)) * 1
    # xsum += x
    # ysum += y

    screen.fill(0)
    screen.blit(a, (xsum, ysum))
    # print(b.topright, b.center)
    pygame.draw.circle(screen, (120, 0, 255), pos, 5)
    pygame.draw.circle(screen, (255, 0, 120), (xsum, ysum), 5)
    pygame.display.flip()
    pygame.time.wait(10)

pygame.time.wait(500)

pygame.quit()
exit()