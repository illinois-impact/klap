
base:
	for d in */; do $(MAKE) -C $$d; done

a%:
	for d in */; do $(MAKE) `echo $$d | tr '[:upper:]' '[:lower:]' | tr -d /`.$@ -C $$d; done

all:
	for d in */; do $(MAKE) all -C $$d; done

clean:
	for d in */; do $(MAKE) clean -C $$d; done
	rm -f *.csv

test:
	for d in */; do $(MAKE) test -C $$d; done
	echo "benchmark, version, runID, time" > all.csv
	for d in */; do grep -v benchmark $$d`echo $$d | tr '[:upper:]' '[:lower:]' | tr -d /`.csv | tee -a all.csv; done

test-all:
	for d in */; do $(MAKE) test-all -C $$d; done
	echo "benchmark, version, runID, time" > all.csv
	for d in */; do grep -v benchmark $$d`echo $$d | tr '[:upper:]' '[:lower:]' | tr -d /`.csv | tee -a all.csv; done

