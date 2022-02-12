#!/bin/bash
IMAGENET1K_DIR=$1
IMAGENET100_DIR=$2

if [ -z "$IMAGENET1K_DIR" ]
then
    echo "Imagenet1k directory has not been provided (should be first argument)"
    exit 1
fi

if [ -z "$IMAGENET100_DIR" ]
then
    echo "Imagenet100 directory has not been provided (should be second argument)"
    exit 1
fi

if [ ! -d $IMAGENET1K_DIR ]
then
    echo "Imagenet1k directory provided $IMAGENET1K_DIR DOES NOT exists."
    exit 1
fi

if [ ! -d $IMAGENET100_DIR ]
then
    echo "Imagenet100 directory provided $IMAGENET100_DIR DOES NOT exists. You need to create it before running this script"
    exit 1
fi

imagenet100_classes=('n01558993' \
                     'n01773797' \
                     'n01983481' \
                     'n02086910' \
                     'n02091831' \
                     'n02105505' \
                     'n02113799' \
                     'n02123045' \
                     'n02326432' \
                     'n02788148' \
                     'n02974003' \
                     'n03259280' \
                     'n03530642' \
                     'n03764736' \
                     'n03794056' \
                     'n03947888' \
                     'n04127249' \
                     'n04418357' \
                     'n04517823' \
                     'n07753275' \
                     'n01692333' \
                     'n01820546' \
                     'n02009229' \
                     'n02087046' \
                     'n02093428' \
                     'n02106550' \
                     'n02113978' \
                     'n02138441' \
                     'n02396427' \
                     'n02804414' \
                     'n03017168' \
                     'n03379051' \
                     'n03584829' \
                     'n03775546' \
                     'n03837869' \
                     'n04026417' \
                     'n04136333' \
                     'n04429376' \
                     'n04589890' \
                     'n07831146' \
                     'n01729322' \
                     'n01855672' \
                     'n02018207' \
                     'n02089867' \
                     'n02099849' \
                     'n02107142' \
                     'n02114855' \
                     'n02172182' \
                     'n02483362' \
                     'n02859443' \
                     'n03032252' \
                     'n03424325' \
                     'n03594734' \
                     'n03777754' \
                     'n03891251' \
                     'n04067472' \
                     'n04229816' \
                     'n04435653' \
                     'n04592741' \
                     'n07836838' \
                     'n01735189' \
                     'n01978455' \
                     'n02085620' \
                     'n02089973' \
                     'n02100583' \
                     'n02108089' \
                     'n02116738' \
                     'n02231487' \
                     'n02488291' \
                     'n02869837' \
                     'n03062245' \
                     'n03492542' \
                     'n03637318' \
                     'n03785016' \
                     'n03903868' \
                     'n04099969' \
                     'n04238763' \
                     'n04485082' \
                     'n07714571' \
                     'n13037406' \
                     'n01749939' \
                     'n01980166' \
                     'n02086240' \
                     'n02090622' \
                     'n02104029' \
                     'n02109047' \
                     'n02119022' \
                     'n02259212' \
                     'n02701002' \
                     'n02877765' \
                     'n03085013' \
                     'n03494278' \
                     'n03642806' \
                     'n03787032' \
                     'n03930630' \
                     'n04111531' \
                     'n04336792' \
                     'n04493381' \
                     'n07715103' \
                     'n13040303')

for SPLIT in train val
do
    mkdir -p $IMAGENET100_DIR/$SPLIT
    for CLASS in ${imagenet100_classes[@]}
    do
        IMAGENET1K_CLASS_DIR=$IMAGENET1K_DIR/$SPLIT/$CLASS
        if [ ! -d $IMAGENET1K_CLASS_DIR ]
        then
            echo "Imagenet1k class dir $IMAGENET1K_CLASS_DIR not found! Check the paths you have provided and that you have correctly downloaded Imagenet1k!"
            exit 1
        fi
        IMAGENET100_CLASS_DIR=$IMAGENET100_DIR/$SPLIT/$CLASS
        echo "Creating symlink for $SPLIT/$CLASS."
        if [ ! -d $IMAGENET100_CLASS_DIR ]
        then
            ln -s $IMAGENET1K_CLASS_DIR $IMAGENET100_CLASS_DIR
        fi
    done
done