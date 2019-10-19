import json
import tensorflow as tf

class SpellClassifier:
    """
    The Spell Classifier contains TensorFlow model information for classifying spells.
    """

    def __init__(self, model_filename, classes_filename):
        self.model = self.load_model(model_filename)
        self.classes = self.load_classes(classes_filename)

    def load_model(self, filename):
        """
        Load TensorFlow model (.h5).
        """

        model = tf.keras.models.load_model(filename)
        model.summary()

        return model

    def load_classes(self, filename):
        """
        Load class labels & indices from a JSON file.
        """
        
        with open(filename) as json_data:
            classes = json.load(json_data)

            # Convert keys to ints
            classes = {int(key): value for key, value in classes.items()}

        return classes