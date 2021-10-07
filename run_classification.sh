for n_hidden_layers in 2 3 4 5
do
    for color in 'grayscale' 'rgb'
    do
        for train_frac in 0.01 0.02 0.05 0.1 0.2 0.3 0.5 0.7 1 
        do
            echo "FigureQA Loop: $color $train_frac $n_hidden_layers."
            CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --train --predict --save --model "cnn" --num_classes 4 --train_frac $train_frac --color_mode $color --n_hidden_layers $n_hidden_layers --default_path "/home/charts-description" --X_train_path "data/figureqa/train" --X_val_path "data/figureqa/validation" --X_test_path "data/figureqa/test" --epoch 20
            CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --predict --model "cnn" --num_classes 4 --train_frac $train_frac --color_mode $color --n_hidden_layers $n_hidden_layers --default_path "/home/charts-description" --X_test_path "data/plotqa/test" --epoch 20
            CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --predict --model "cnn" --num_classes 4 --train_frac $train_frac --color_mode $color --n_hidden_layers $n_hidden_layers --default_path "/home/charts-description" --X_test_path "data/chata" --epoch 20
        done
    done
done
echo "FigureQA SVM."
for color in 'grayscale' 'rgb'
    CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --train --predict --save --model "svm" --num_classes 4 --train_frac 1 --color_mode $color --n_hidden_layers 3 --default_path "/home/charts-description" --X_train_path "data/figureqa/train" --X_val_path "data/figureqa/validation" --X_test_path "data/figureqa/test" --epoch 20
    CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --predict --model "svm" --num_classes 4 --train_frac 1 --color_mode $color --n_hidden_layers 3 --default_path "/home/charts-description" --X_test_path "data/plotqa/test" --epoch 20
    CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --predict --model "svm" --num_classes 4 --train_frac 1 --color_mode $color --n_hidden_layers 3 --default_path "/home/charts-description" --X_test_path "data/chata" --epoch 20
done
echo "FigureQA ResNet."
CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --train --predict --save --model "resnet" --num_classes 4 --train_frac 1 --color_mode "rgb" --n_hidden_layers 3 --default_path "/home/charts-description" --X_train_path "data/figureqa/train" --X_val_path "data/figureqa/validation" --X_test_path "data/figureqa/test" --epoch 20
CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --predict --model "resnet" --num_classes 4 --train_frac 1 --color_mode "rgb" --n_hidden_layers 3 --default_path "/home/charts-description" --X_test_path "data/plotqa/test" --epoch 20
CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --predict --model "resnet" --num_classes 4 --train_frac 1 --color_mode "rgb" --n_hidden_layers 3 --default_path "/home/charts-description" --X_test_path "data/chata" --epoch 20

for n_hidden_layers in 2 3 4 5
do
    for color in 'grayscale' 'rgb'
    do
        for train_frac in 0.01 0.02 0.05 0.1 0.2 0.3 0.5 0.7 1 
        do
            echo "PlotQA Loop: $color $train_frac $n_hidden_layers."
            CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --train --predict --save --model "cnn" --num_classes 4 --train_frac $train_frac --color_mode $color --n_hidden_layers $n_hidden_layers --default_path "/home/charts-description" --X_train_path "data/plotqa/train" --X_val_path "data/plotqa/validation" --X_test_path "data/plotqa/test" --epoch 20
            CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --predict --model "cnn" --num_classes 4 --train_frac $train_frac --color_mode $color --n_hidden_layers $n_hidden_layers --default_path "/home/charts-description" --X_test_path "data/figureqa/test" --epoch 20
            CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --predict --model "cnn" --num_classes 4 --train_frac $train_frac --color_mode $color --n_hidden_layers $n_hidden_layers --default_path "/home/charts-description" --X_test_path "data/chata" --epoch 20
        done
    done
done
echo "PlotQA SVM."
for color in 'grayscale' 'rgb'
    CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --train --predict --save --model "svm" --num_classes 4 --train_frac 1 --color_mode $color --n_hidden_layers 3 --default_path "/home/charts-description" --X_train_path "data/plotqa/train" --X_val_path "data/plotqa/validation" --X_test_path "data/plotqa/test" --epoch 20
    CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --predict --model "svm" --num_classes 4 --train_frac 1 --color_mode $color --n_hidden_layers 3 --default_path "/home/charts-description" --X_test_path "data/figureqa/test" --epoch 20
    CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --predict --model "svm" --num_classes 4 --train_frac 1 --color_mode $color --n_hidden_layers 3 --default_path "/home/charts-description" --X_test_path "data/chata" --epoch 20
done
echo "PlotQA ResNet."
CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --train --predict --save --model "resnet" --num_classes 4 --train_frac 1 --color_mode "rgb" --n_hidden_layers 3 --default_path "/home/charts-description" --X_train_path "data/plotqa/train" --X_val_path "data/plotqa/validation" --X_test_path "data/plotqa/test" --epoch 20
CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --predict --model "resnet" --num_classes 4 --train_frac 1 --color_mode "rgb" --n_hidden_layers 3 --default_path "/home/charts-description" --X_test_path "data/figureqa/test" --epoch 20
CUDA_VISIBLE_DEVICES=0 python ChaDes/run.py --predict --model "resnet" --num_classes 4 --train_frac 1 --color_mode "rgb" --n_hidden_layers 3 --default_path "/home/charts-description" --X_test_path "data/chata" --epoch 20
