import logging
import os

from models.classifier import ChartClassifier
from models.resnet import ResNetClassifier
from config import Config
from argparse import ArgumentParser


logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

logger = logging.getLogger('__file__')

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='cnn', help='cnn, svm')

    parser.add_argument("--save", action='store_true', help="whether to save the model")
    parser.add_argument('--train', action='store_true', help='whether to train the model')
    parser.add_argument('--predict', action='store_true', help='whether to predict data')

    parser.add_argument("--default_path", type=str, default='/content/drive/My Drive')

    parser.add_argument("--X_train_path", type=str, default='data/figureqa/train',
                        help="path to training data divided into categories")
    parser.add_argument("--X_val_path", type=str, default='data/figureqa/validation',
                        help="path to validation data divided into categories")
    parser.add_argument("--X_test_path", type=str, default='data/figureqa/validation_1k',
                        help="path to test data divided into categories")

    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epoch", type=int, default=5, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--color_mode", type=str, default="rgb", help="color mode: 1) 'rgb' \
                                                                                   2) 'grayscale'")

    parser.add_argument("--n_hidden_layers", type=int, default=3, help="number of hidden layers")
    parser.add_argument("--target_size", type=tuple, default=(164, 164), help="image size")
    parser.add_argument("--steps_per_epoch", default=None, help="steps per epoch")
    parser.add_argument("--validation_steps", default=None, help="validation steps")
    parser.add_argument("--train_frac", type=float, default=1.0, help="training set fraction used for traning")

    parser.add_argument("--num_classes", type=int, default=4, help="number of categories")

    args = parser.parse_args()

    config = Config(model=args.model,
                    X_train_path=os.path.join(args.default_path, args.X_train_path),
                    X_val_path=os.path.join(args.default_path, args.X_val_path),
                    default_path=args.default_path,
                    target_size=args.target_size,
                    batch_size=args.batch_size,
                    epoch=args.epoch,
                    color_mode=args.color_mode,
                    steps_per_epoch=args.steps_per_epoch,
                    validation_steps=args.validation_steps,
                    train_frac=args.train_frac,
                    n_hidden_layers=args.n_hidden_layers,
                    num_classes=args.num_classes)

    task_catalog = os.path.join(args.default_path, args.model)
    if not os.path.exists(task_catalog):
        os.makedirs(task_catalog)

    prefix = "_".join(["color", str(args.color_mode),
                       "frac", str(args.train_frac),
                       "layers", str(args.n_hidden_layers),
                       "epoch", str(args.epoch)])

    if not os.path.exists(os.path.join(task_catalog, prefix)):
        os.makedirs(os.path.join(task_catalog, prefix))

    if args.model == "svm" or args.model == "cnn":
        model = ChartClassifier(config=config, prefix=prefix)
    elif args.model == "resnet":
        model = ResNetClassifier(config=config, prefix=prefix)

    if args.train:
        model.train()
    else:
        model.load_model()

    if args.save:
        model.save_model()

    if args.predict:
        model.predict(os.path.join(args.default_path, args.X_test_path))
