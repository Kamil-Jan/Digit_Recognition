from Canvas import Canvas
from tensorflow.keras.models import load_model


def main():
    model = load_model("model.h5")
    canvas = Canvas(model)
    canvas.run()

if __name__ == "__main__":
    main()

