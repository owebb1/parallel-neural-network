OUTFILE=/home/owebb1/cs87/project/source/sequential/exp2

echo "epochs=30; batch size=10" &>> $OUTFILE
echo "" &>> $OUTFILE
cd /home/owebb1/cs87/project/source/sequential/ProjectCode/bin/
./neuralNet >> $OUTFILE

