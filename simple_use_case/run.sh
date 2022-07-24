PROJECT_ROOT='/home/sxr/code/xconfig_dev'

CONFIG_FILE='config.yaml'
RESULTS_ROOT='/home/sxr/code/xconfig_dev/simple_use_case/results'

mkdir -p $RESULTS_ROOT

python main.py $PROJECT_ROOT $CONFIG_FILE \
    'results_root:str:'$RESULTS_ROOT \
    'model|scale_net|lr:float:0.0005' \
