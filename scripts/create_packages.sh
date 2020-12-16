#!/bin/bash

DRESDEN_BRANCH="ebenjamin/lightgbm-io-speedup"
BUZZSAW_VERSION=0.13.0


WORK_DIR=$(pwd)
PACKAGES_OUTPUT_DIR="${WORK_DIR}/packages"
rm -rf $PACKAGES_OUTPUT_DIR

git clone  --single-branch --branch "$DRESDEN_BRANCH" "https://github.etsycorp.com/Engineering/dresden.git" /tmp/dresden

mkdir -p $PACKAGES_OUTPUT_DIR

#Get all needed package distributions for dresden
declare -a dresden_packages=("dresden-common" "dresden-hp" "dresden-lightgbm")
for package in "${dresden_packages[@]}"
do
   cd /tmp/dresden/${package}
   python setup.py sdist
   cp dist/*.tar.gz $PACKAGES_OUTPUT_DIR
done

#Get the neccesary Buzzsaw wheel
gsutil cp gs://etsy-mlinfra-prod-libraries-4kin/buzzsaw/Buzzsaw-"$BUZZSAW_VERSION"-*.whl $PACKAGES_OUTPUT_DIR

#Get the moltr package
#git clone --single-branch --branch "master" "https://github.com/akurennoy/moltr.git" /tmp/moltr
cd $WORK_DIR/lightgbm_ai_platform/moltr
python setup.py sdist
cp dist/*.tar.gz $PACKAGES_OUTPUT_DIR

#Clean up
rm -rf /tmp/dresden
