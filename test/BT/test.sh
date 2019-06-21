
NAME=bt
VERSIONS='base aw ab ag'
RUNS=10
FILE=$NAME.csv

echo "benchmark, version, runID, time" > $FILE
for version in $VERSIONS; do
    if [ -f $NAME.$version ]; then
        for r in `seq 1 $RUNS`; do
            echo $NAME, $version, $r, `./$NAME.$version -o 0` | tee -a $FILE
        done
    fi
done

