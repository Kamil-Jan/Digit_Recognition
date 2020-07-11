import pygame
import numpy as np
import matplotlib.pyplot as plt
import cv2
pygame.init()


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)

class Canvas(object):
    def __init__(self, model, width=560, height=560):
        self.window = pygame.display.set_mode((width, height))
        self.background = pygame.Surface(self.window.get_size())
        pygame.display.set_caption("Py Paint")
        self.clear_screen()

        self.model = model
        self.brush_color = BLACK
        self.brush_size = 20

    def clear_screen(self):
        self.background.fill(WHITE)

    def get_image(self):
        # Get the image from Canvas
        imgdata = pygame.surfarray.pixels3d(self.background)
        imgdata = imgdata.swapaxes(1, 0)
        img = cv2.resize(imgdata, (28, 28), cv2.INTER_AREA)

        # Convert the image to gray scale
        gray_scale = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        gray_scale = 255 - gray_scale
        gray_scale = gray_scale / 255.0

        # Reshape image
        pred_image = gray_scale.reshape((1, -1))
        pred_image = np.expand_dims(pred_image, 1)

        prediction = self.model.predict(pred_image)
        pred_label = np.argmax(prediction)

        pred_string = f"Predict: {pred_label}"
        print(pred_string, end="", flush=True)
        print("\b" * len(pred_string), end="", flush=True)

    def catch_events(self):
        pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    self.brush_size += 5
                if event.button == 5:
                    self.brush_size -= 5
                    if self.brush_size < 0:
                        self.brush_size = 1

        if pygame.mouse.get_pressed() == (1, 0, 0):
            self.brush_color = BLACK
            pygame.draw.circle(self.background, self.brush_color,
                               pos, self.brush_size)

        if pygame.mouse.get_pressed() == (0, 0, 1):
            self.get_image()

        if pygame.mouse.get_pressed() == (0, 1, 0):
            self.clear_screen()

    def run(self):
        running = True

        while running:
            self.catch_events()
            # self.get_image()
            self.window.blit(self.background, (0, 0))
            pygame.display.update()

