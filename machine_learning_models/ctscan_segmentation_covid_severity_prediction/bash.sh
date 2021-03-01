filename=$1
outputfile=${1}_png

for i in $filename/*
do
	python niitopng.py -i $i -o $outputfile
done
