# Faster-RCNN from scratch

## Download Pascal VOC dataset

```bash
mkdir -p ./data

curl -L -o VOCtrainval_06-Nov-2007.tar http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar -xf VOCtrainval_06-Nov-2007.tar
mv VOCdevkit/VOC2007 ./data/VOC2007

curl -L -o VOCtest_06-Nov-2007.tar http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
mv VOCdevkit/VOC2007 ./data/VOC2007-test

rm -rf VOCdevkit
rm VOCtrainval_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar
```

