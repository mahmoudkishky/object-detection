import argparse
import importlib
import sys

MODEL_CHOICES = ('yolo', 'ssd', 'rcnn')


def load_modules(model_name):
    """Import the configs/model/trainer of the selected model.

    The trainers and data generator do `import configs`, so the selected
    configs module is registered under that name before they are imported.
    """
    sys.modules['configs'] = importlib.import_module(f'configs.{model_name}_configs')
    models = importlib.import_module(f'models.{model_name}_model')
    trainer = importlib.import_module(f'trainers.{model_name}_trianer')
    data_generator = importlib.import_module('dataGenerator')
    return sys.modules['configs'], models, trainer.Trainer, data_generator.VOCDataGenerator


def build_model(model_name, configs, models):
    if model_name == 'yolo':
        model_classes = {'YOLO': models.YOLO, 'YOLO_Simple': models.YOLO_Simple}
        if configs.MODEL not in model_classes:
            raise ValueError(f"Unsupported model type: {configs.MODEL}")
        return model_classes[configs.MODEL](
            input_shape=(configs.IMAGE_SIZE_X, configs.IMAGE_SIZE_Y, 3),
            num_classes=len(configs.VOC_LABELS),
            grid_size=configs.GRID_SIZE,
            anchors=configs.YOLO_ANCHORS,
        )

    if model_name == 'rcnn':
        if configs.MODEL != 'RCNN':
            raise ValueError(f"Unsupported model type: {configs.MODEL}")
        return models.RCNN(
            input_shape=(configs.IMAGE_SIZE_X, configs.IMAGE_SIZE_Y, 3),
            num_class=len(configs.VOC_LABELS),
        )

    if configs.MODEL != 'SSD_Simple':
        raise ValueError(f"Unsupported model type: {configs.MODEL}")
    return models.SSD_Simple(
        input_shape=(None, configs.IMAGE_SIZE_X, configs.IMAGE_SIZE_Y, 3),
        num_classes=len(configs.VOC_LABELS),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES, help="model to train")
    parser.add_argument("--epochs", dest="epochs", type=int, default=10, help="number of training epochs")
    args = parser.parse_args()

    configs, models, Trainer, VOCDataGenerator = load_modules(args.model)

    training_generator = VOCDataGenerator(configs.DATA_PATH, configs.VOC_LABELS, configs.BATCH_SIZE, True)
    val_generator = VOCDataGenerator(configs.DATA_PATH, configs.VOC_LABELS, configs.BATCH_SIZE, False)

    model = build_model(args.model, configs, models)
    trainer = Trainer(model, training_generator, val_generator, args.epochs, configs.LEARNING_RATE)
    trainer.train()


if __name__ == "__main__":
    main()
