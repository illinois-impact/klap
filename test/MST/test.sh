
NAME=mst
VERSIONS='base aw ab ag'
RUNS=10
FILE=$NAME.csv

echo "benchmark, version, runID, time" > $FILE
for version in $VERSIONS; do
    if [ -f $NAME.$version ]; then
        for r in `seq 1 $RUNS`; do
            TIMES=`./$NAME.$version -o 0`
            FTIME=`echo $TIMES | cut -d , -f 1`
            VTIME=`echo $TIMES | cut -d , -f 2`
            echo ${NAME}f, $version, $r, $FTIME | tee -a $FILE
            echo ${NAME}v, $version, $r, $VTIME | tee -a $FILE
        done
    fi
done

