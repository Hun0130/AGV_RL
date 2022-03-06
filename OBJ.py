import pygame

class obj():
    rows = 20
    w = 500
    def __init__(self, start, dirnx = 1, dirny = 0, color = (255,0,0)):
        self.pos = start

        self.dirnx = dirnx
        self.dirny = dirny 
        self.color = color
        
        # the number of products produced
        self.product = 0

    def move(self, dirnx, dirny):
        self.dirnx = dirnx
        self.dirny = dirny
        self.pos  = (self.pos[0] + self.dirnx, self.pos[1] + self.dirny)

    def draw(self, surface, eyes = False):
        dis = self.w // self.rows
        i = self.pos[0]
        j = self.pos[1]
        
        pygame.draw.rect(surface, self.color, (i * dis+1, j * dis+1, dis-2, dis-2))
        if eyes:
            centre = dis // 2
            radius = 3
            circleMiddle = (i * dis + centre - radius, j * dis + 8)
            circleMiddle2 = (i * dis + dis - radius * 2, j * dis + 8)
            pygame.draw.circle(surface, (0,0,0), circleMiddle, radius)
            pygame.draw.circle(surface, (0,0,0), circleMiddle2, radius)
            
    def produce(self):
        self.product = self.product + 1